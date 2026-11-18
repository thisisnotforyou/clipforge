"""ClipForge: paste a link (or upload a video), get ready-to-post 9:16 clips.

Pipeline per job:
  1. Read the video's info (YouTube "Most replayed" heatmap when it exists).
  2. Find the best moments: heatmap peaks, else loudness spikes in the audio.
  3. Download only those parts (yt-dlp download sections).
  4. Transcribe each part with faster-whisper (word timings) and trim the end on a sentence.
  5. Render 1080x1920 with captions, a hook plate, the chosen style and -14 LUFS sound.
  6. Write a caption + hashtags per clip, and a zip of everything.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
# yt-dlp needs Deno (installed into the venv by `pip install deno`) to solve YouTube's JS challenges.
os.environ["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", ROOT / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# --- Limits for a public deployment (override with env vars) ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1"))          # jobs rendered at once
MAX_SOURCE_SECONDS = int(os.environ.get("MAX_SOURCE_SECONDS", str(8 * 3600)))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "6"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
ALLOWED_HOSTS = ("youtube.com", "youtu.be", "twitch.tv", "kick.com")

app = FastAPI(title="ClipForge")
# Sites allowed to call this engine from the browser, e.g. "https://clipforge.vercel.app".
# Comma separated; "*" allows any site.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
active_by_ip: dict[str, str] = {}
_whisper = None
_whisper_lock = threading.Lock()


# ----------------------------------------------------------------- helpers
def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-800:]}")
    return p.stderr + p.stdout


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out or 0)


def update(job_id: str, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)
        snapshot = dict(jobs[job_id])
    (JOBS_DIR / job_id / "job.json").write_text(json.dumps(snapshot, indent=1))


def host_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def load_wav(path: Path):
    """16 kHz mono PCM wav -> float32 array (skips PyAV decoding inside faster-whisper)."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def ytdl(opts: dict, url: str, download: bool = True, tries: int = 3):
    """yt-dlp with retries: YouTube sometimes answers 403 on the first try."""
    import yt_dlp
    extra = {}
    if os.environ.get("YTDLP_COOKIES") and Path(os.environ["YTDLP_COOKIES"]).exists():
        extra["cookiefile"] = os.environ["YTDLP_COOKIES"]  # helps on cloud servers YouTube flags as bots
    if os.environ.get("YTDLP_PROXY"):
        extra["proxy"] = os.environ["YTDLP_PROXY"]
    opts = {**extra, **opts}
    last = None
    for attempt in range(tries):
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True, **opts}) as y:
                return y.extract_info(url, download=download)
        except Exception as ex:  # noqa: BLE001
            last = ex
            if "Unsupported URL" in str(ex) or "Private video" in str(ex):
                break
            time.sleep(2 + attempt * 3)
    raise last


def get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")
    return _whisper


# --------------------------------------------------------- moment finding
def pick_windows(scores: list[tuple[float, float]], duration: float, n: int,
                 length: float, skip_intro: float) -> list[tuple[float, float, float]]:
    """scores: (time, score). Returns up to n non-overlapping (start, end, score) windows.
    The peak sits ~60% into the window so the setup plays before the payoff."""
    chosen: list[tuple[float, float, float]] = []
    for t, sc in sorted(scores, key=lambda x: -x[1]):
        if t < skip_intro:
            continue
        start = max(0.0, t - length * 0.6)
        end = min(duration, start + length)
        start = max(0.0, end - length)
        if any(not (end + 4 <= s or start >= e + 4) for s, e, _ in chosen):
            continue
        chosen.append((start, end, sc))
        if len(chosen) >= n:
            break
    return sorted(chosen)


def moments_from_heatmap(heatmap: list[dict]) -> list[tuple[float, float]]:
    return [((h["start_time"] + h["end_time"]) / 2, float(h["value"])) for h in heatmap]


def moments_from_loudness(audio: Path) -> list[tuple[float, float]]:
    """Per-second loudness, smoothed, scored against the local (2 min) median,
    so a shout in a quiet part counts as much as a shout in a loud part."""
    log = run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-vn", "-ac", "1",
        "-af", "aresample=8000,asetnsamples=n=8000,astats=metadata=1:reset=1,"
               "ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ])
    levels = []
    for m in re.finditer(r"RMS_level=(-?[\d.]+|-inf)", log):
        v = m.group(1)
        levels.append(-90.0 if v == "-inf" else max(-90.0, float(v)))
    if not levels:
        return []
    k = 3
    smooth = [statistics.fmean(levels[max(0, i - k): i + k + 1]) for i in range(len(levels))]
    out = []
    for i in range(0, len(smooth)):
        lo, hi = max(0, i - 60), min(len(smooth), i + 60)
        base = statistics.median(smooth[lo:hi:4] or smooth[lo:hi])
        out.append((float(i), smooth[i] - base))
    return out


# ------------------------------------------------------------- captions
def ass_color(hex_rgb: str, alpha: str = "00") -> str:
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


STYLES = {
    # Neon: big uppercase word-by-word captions, active word lit, neon frame.
    "neon": dict(font="Arial Black", size=88, primary="FFFFFF", active="22F0FF",
                 outline="000000", outline_w=7, cap_y=1380, upper=True, chunk=3,
                 hook_bg="FF2BD6", hook_fg="FFFFFF", border="FF2BD6"),
    # Clean: sentence captions at the bottom, calm hook, no frame.
    "clean": dict(font="Arial", size=64, primary="FFFFFF", active="FFE600",
                  outline="000000", outline_w=5, cap_y=1640, upper=False, chunk=7,
                  hook_bg="000000", hook_fg="FFFFFF", border=None),
}


def ass_escape(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def fmt_t(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(words: list[dict], hook: str, length: float, style: str) -> str:
    st = STYLES[style]
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},{ass_color(st['primary'])},{ass_color(st['primary'])},{ass_color(st['outline'])},&H80000000,1,0,0,0,100,100,0,0,1,{st['outline_w']},2,8,60,60,0,1
Style: Hook,{st['font']},58,{ass_color(st['hook_fg'])},{ass_color(st['hook_fg'])},{ass_color(st['hook_bg'])},{ass_color(st['hook_bg'], '20')},1,0,0,0,100,100,0,0,3,18,0,8,70,70,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook:
        h = ass_escape(hook.upper() if st["upper"] else hook)
        lines.append(f"Dialogue: 1,{fmt_t(0)},{fmt_t(min(length, 4.5))},Hook,,0,0,0,,{{\\fad(150,250)}}{h}")
    act = ass_color(st["active"])
    pos = f"{{\\an8\\pos(540,{st['cap_y']})}}"
    chunk = st["chunk"]
    groups = [words[i:i + chunk] for i in range(0, len(words), chunk)]
    for g in groups:
        for j, w in enumerate(g):
            start = w["start"]
            end = g[j + 1]["start"] if j + 1 < len(g) else w["end"]
            if end - start < 0.05:
                end = start + 0.05
            parts = []
            for k, x in enumerate(g):
                txt = ass_escape(x["word"].strip())
                if st["upper"]:
                    txt = txt.upper()
                if k == j:
                    pop = "\\fscx112\\fscy112" if style == "neon" else ""
                    parts.append(f"{{\\c{act}{pop}}}{txt}{{\\r}}")
                else:
                    parts.append(txt)
            lines.append(f"Dialogue: 0,{fmt_t(start)},{fmt_t(end)},Cap,,0,0,0,,{pos}{' '.join(parts)}")
    return hdr + "\n".join(lines) + "\n"


# ------------------------------------------------------------- rendering
def render_clip(src: Path, offset: float, length: float, ass_name: str, out: Path,
                style: str, layout: str, cwd: Path):
    st = STYLES[style]
    if layout == "crop":
        vf = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[base]"
    else:  # blur: whole frame in the middle, blurred copy fills the screen
        vf = ("[0:v]split=2[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              "boxblur=24:2,eq=brightness=-0.10:saturation=1.2[bg];"
              "[b]scale=1080:1920:force_original_aspect_ratio=decrease,"
              "scale='if(gt(a,1),1180,iw)':-2,crop='min(iw,1080)':ih[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]")
    tail = "[base]"
    if st["border"]:
        vf += f";[base]drawbox=x=0:y=0:w=iw:h=ih:color=0x{st['border']}@0.95:t=14[framed]"
        tail = "[framed]"
    vf += f";{tail}ass={ass_name}[v]"
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset:.3f}", "-t", f"{length:.3f}", "-i", str(src),
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", str(out),
    ], cwd=cwd)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{min(1.5, length / 2):.2f}",
         "-i", str(out), "-frames:v", "1", "-vf", "scale=360:-2", str(out.with_suffix(".jpg"))])


# --------------------------------------------------------------- posting
STOP = set("the a an and or of to in on for with is are was this that my your our i you it at by from vs".split())


def hashtags_for(title: str, extra: list[str]) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if w.lower() not in STOP and len(w) > 2]
    tags, seen = [], set()
    for w in words[:5] + extra:
        t = "#" + w.lower()
        if t not in seen:
            seen.add(t)
            tags.append(t)
    return tags[:8]


def make_hook(title: str, text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
    hook = first if 12 <= len(first) <= 60 else title
    hook = re.sub(r"\s+", " ", hook).strip()
    return (hook[:57] + "...") if len(hook) > 60 else hook


# ------------------------------------------------------------------ job
def process(job_id: str, opts: dict):
    jdir = JOBS_DIR / job_id
    try:
        n, length = opts["num_clips"], opts["clip_len"]
        style, layout, captions = opts["style"], opts["layout"], opts["captions"]
        info: dict = {}
        upload: Path | None = opts.get("upload")
        update(job_id, status="running", stage="Reading the video", progress=3)

        if upload:
            duration = probe_duration(upload)
            title = opts.get("upload_name") or "My video"
            uploader, page_url = "", ""
        else:
            info = ytdl({"skip_download": True}, opts["url"], download=False)
            if info.get("_type") == "playlist":
                raise RuntimeError("That link is a playlist or channel. Paste a single video or stream link.")
            if info.get("is_live"):
                raise RuntimeError("This stream is live right now. Paste it again after it ends (live clipping is not supported yet).")
            duration = float(info.get("duration") or 0)
            title = info.get("title") or "Video"
            uploader = info.get("uploader") or info.get("channel") or ""
            page_url = info.get("webpage_url") or opts["url"]
        if not duration:
            raise RuntimeError("Could not read the video's length.")
        if duration > MAX_SOURCE_SECONDS:
            raise RuntimeError(f"Video is longer than {MAX_SOURCE_SECONDS // 3600} hours, the limit on this site.")
        length = min(length, max(10.0, duration))
        update(job_id, title=title, uploader=uploader, source=page_url, duration=duration)

        # 2. best moments
        heat = info.get("heatmap") if info else None
        skip_intro = min(30.0, duration * 0.03)
        if heat:
            update(job_id, stage="Finding the most replayed moments", progress=10)
            scores = moments_from_heatmap(heat)
            method = "YouTube Most replayed"
        else:
            update(job_id, stage="Listening for the loudest reactions", progress=8)
            if upload:
                audio = upload
            else:
                ytdl({"format": "worstaudio/bestaudio/worst",
                      "outtmpl": str(jdir / "audio.%(ext)s")}, page_url)
                audio = next(jdir.glob("audio.*"))
            scores = moments_from_loudness(audio)
            method = "loudness spikes"
            if not upload:
                audio.unlink(missing_ok=True)
        windows = pick_windows(scores, duration, n, length, skip_intro)
        if not windows:  # short video: just take it from the top
            windows = [(0.0, min(duration, length), 1.0)]
        update(job_id, method=method, stage=f"Found {len(windows)} moments", progress=15)

        clips = []
        per = 80 / len(windows)
        for i, (s, e, sc) in enumerate(windows, 1):
            base = 15 + per * (i - 1)
            # 3. get the part
            update(job_id, stage=f"Clip {i}/{len(windows)}: downloading the moment", progress=int(base))
            pad = 3.0
            if upload:
                src, offset, avail = upload, s, min(duration - s, e - s + pad)
            else:
                from yt_dlp.utils import download_range_func
                for f in jdir.glob(f"raw_{i}.*"):
                    f.unlink()
                ytdl({
                    "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
                    "merge_output_format": "mp4",
                    "download_ranges": download_range_func(None, [(s, min(duration, e + pad))]),
                    "outtmpl": str(jdir / f"raw_{i}.%(ext)s"),
                }, page_url)
                src = next(f for f in jdir.glob(f"raw_{i}.*") if not f.name.endswith(".part"))
                offset, avail = 0.0, probe_duration(src)
            want = e - s

            # 4. transcribe + trim the end on a finished sentence
            words, text = [], ""
            if captions:
                update(job_id, stage=f"Clip {i}/{len(windows)}: writing captions", progress=int(base + per * 0.35))
                wav = jdir / f"clip_{i}.wav"
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}",
                     "-t", f"{avail:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
                pcm = load_wav(wav)
                segs = list(get_whisper().transcribe(pcm, word_timestamps=True, vad_filter=True)[0])
                if not segs:  # VAD drops singing / speech over loud game audio: retry without it
                    segs = list(get_whisper().transcribe(pcm, word_timestamps=True)[0])
                wav.unlink(missing_ok=True)
                ends = [sg.end for sg in segs if want * 0.7 <= sg.end <= avail]
                if ends:
                    want = min(ends, key=lambda x: abs(x - want)) + 0.35
                want = min(want, avail)
                for sg in segs:
                    for w in sg.words or []:
                        if w.start < want and w.word.strip():
                            words.append({"word": w.word, "start": w.start, "end": min(w.end, want)})
                text = " ".join(sg.text.strip() for sg in segs if sg.start < want)
            want = max(3.0, min(want, avail))

            # 5. render
            update(job_id, stage=f"Clip {i}/{len(windows)}: editing 9:16", progress=int(base + per * 0.6))
            hook = make_hook(title, text) if opts["hook"] else ""
            (jdir / f"clip_{i}.ass").write_text(build_ass(words, hook, want, style), encoding="utf-8")
            out = jdir / f"clip_{i}.mp4"
            render_clip(src, offset, want, f"clip_{i}.ass", out, style, layout, jdir)
            if not upload:
                src.unlink(missing_ok=True)

            ts = int(s)
            credit = f"Credit: {uploader} ({page_url}{'&' if '?' in page_url else '?'}t={ts})" if uploader else ""
            tags = hashtags_for(title, ["shorts", "clips", "gaming"])
            caption = hook or title
            clips.append({
                "n": i, "file": out.name, "thumb": out.with_suffix(".jpg").name,
                "start": round(s, 1), "length": round(want, 1), "score": round(sc, 3),
                "hook": hook, "caption": caption, "hashtags": tags, "credit": credit,
                "transcript": text,
            })
            update(job_id, clips=clips)

        # 6. posting notes + zip, best first
        ranked = sorted(clips, key=lambda c: -c["score"])
        md = [f"# Clips from: {title}", f"Moments picked by {method}.", ""]
        for r, c in enumerate(ranked, 1):
            md += [f"## #{r} - {c['file']}  (starts {int(c['start'] // 60)}:{int(c['start'] % 60):02d} in the source)",
                   "", c["caption"], "", " ".join(c["hashtags"]), ""]
            if c["credit"]:
                md += [c["credit"], ""]
        (jdir / "POSTING.md").write_text("\n".join(md), encoding="utf-8")
        with zipfile.ZipFile(jdir / "clips.zip", "w") as z:
            for c in clips:
                z.write(jdir / c["file"], c["file"])
            z.write(jdir / "POSTING.md", "POSTING.md")
        update(job_id, status="done", stage="Done", progress=100, clips=clips)
    except Exception as ex:  # noqa: BLE001 - shown to the user
        msg = str(ex)
        if "Sign in to confirm" in msg or "bot" in msg.lower():
            msg = "YouTube blocked the download from this server (bot check). Try again later, or upload the video file."
        update(job_id, status="error", stage="Failed", error=msg[:500])
    finally:
        if opts.get("upload"):
            Path(opts["upload"]).unlink(missing_ok=True)
        with jobs_lock:
            for ip, jid in list(active_by_ip.items()):
                if jid == job_id:
                    del active_by_ip[ip]


# ------------------------------------------------------------------ API
def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return (fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?"))


@app.post("/api/jobs")
async def create_job(
    request: Request,
    url: str = Form(""),
    file: Optional[UploadFile] = File(None),
    num_clips: int = Form(3),
    clip_len: int = Form(30),
    style: str = Form("neon"),
    layout: str = Form("blur"),
    captions: bool = Form(True),
    hook: bool = Form(True),
):
    ip = client_ip(request)
    with jobs_lock:
        if ip in active_by_ip:
            raise HTTPException(429, "You already have a job running. Wait for it to finish.")
        queued = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
        if queued >= MAX_QUEUE:
            raise HTTPException(503, "The site is busy right now. Try again in a few minutes.")
    if style not in STYLES or layout not in ("blur", "crop"):
        raise HTTPException(400, "Unknown style or layout.")
    num_clips = max(1, min(8, num_clips))
    clip_len = max(15, min(90, clip_len))

    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    jdir.mkdir()
    opts = dict(num_clips=num_clips, clip_len=float(clip_len), style=style, layout=layout,
                captions=captions, hook=hook)
    if file is not None and file.filename:
        dest = jdir / ("upload" + Path(file.filename).suffix.lower()[:6])
        size, cap = 0, MAX_UPLOAD_MB * 1024 * 1024
        with dest.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > cap:
                    fh.close()
                    shutil.rmtree(jdir, ignore_errors=True)
                    raise HTTPException(413, f"File is bigger than {MAX_UPLOAD_MB} MB.")
                fh.write(chunk)
        opts["upload"] = dest
        opts["upload_name"] = Path(file.filename).stem
    else:
        url = url.strip()
        if not url.startswith(("http://", "https://")) or not host_allowed(url):
            shutil.rmtree(jdir, ignore_errors=True)
            raise HTTPException(400, "Paste a YouTube, Twitch or Kick link, or upload a video file.")
        opts["url"] = url

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Waiting in line", "progress": 0,
                        "clips": [], "created": time.time(), "options": {k: v for k, v in opts.items()
                                                                          if k not in ("upload",)}}
        active_by_ip[ip] = job_id
    update(job_id)
    executor.submit(process, job_id, opts)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job and job["status"] == "queued":
            ahead = sum(1 for j in jobs.values()
                        if j["status"] in ("queued", "running") and j["created"] < job["created"])
            job = dict(job, ahead=ahead)
    if not job:
        p = JOBS_DIR / job_id / "job.json"
        if re.fullmatch(r"[0-9a-f]{12}", job_id) and p.exists():
            job = json.loads(p.read_text())
        else:
            raise HTTPException(404, "Job not found (clips are deleted after a few hours).")
    return JSONResponse(job)


@app.get("/files/{job_id}/{name}")
def get_file(job_id: str, name: str, dl: int = 0):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not re.fullmatch(
            r"(clip_\d+\.(mp4|jpg)|POSTING\.md|clips\.zip)", name):
        raise HTTPException(404)
    p = JOBS_DIR / job_id / name
    if not p.exists():
        raise HTTPException(404)
    # dl=1 forces "save file": the <a download> attribute is ignored across domains.
    attach = dl or name.endswith((".zip", ".md"))
    return FileResponse(p, filename=name if attach else None)


def janitor():
    while True:
        cutoff = time.time() - JOB_TTL_HOURS * 3600
        for d in JOBS_DIR.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                with jobs_lock:
                    j = jobs.get(d.name)
                    if j and j["status"] in ("queued", "running"):
                        continue
                    jobs.pop(d.name, None)
                shutil.rmtree(d, ignore_errors=True)
        time.sleep(600)


threading.Thread(target=janitor, daemon=True).start()
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
