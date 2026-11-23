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


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip().split(",")
    return (int(out[0]), int(out[1])) if len(out) == 2 and out[0] else (1920, 1080)


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
                 length: float, skip_intro: float, skip_outro: float = 0.0) -> list[tuple[float, float, float]]:
    """scores: (time, score). Returns up to n non-overlapping (start, end, score) windows.
    The peak sits ~60% into the window so the setup plays before the payoff.
    The intro and outro (end screens, "subscribe" music) are left out."""
    chosen: list[tuple[float, float, float]] = []
    last = max(length, duration - skip_outro)
    for t, sc in sorted(scores, key=lambda x: -x[1]):
        if t < skip_intro or t > last:
            continue
        start = max(0.0, t - length * 0.6)
        end = min(last, start + length)
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


# ------------------------------------------------------------- facecam
def find_facecam(src: Path, offset: float, length: float):
    """Find a streamer's face-cam overlay. Returns (x, y, w, h) as fractions of the frame, or None.

    1. Sample frames; a face that sits in the same spot in most of them is the cam
       (faces in the gameplay move around).
    2. Grow a box out from that face over pixels that barely change between frames:
       the streamer's room behind them is still, the gameplay around the cam is not.
    """
    import cv2
    import numpy as np

    W, H, N = 640, 360, 16
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{offset:.2f}", "-t", f"{length:.2f}", "-i", str(src),
        "-vf", f"fps={N / max(length, 1):.4f},scale={W}:{H}", "-frames:v", str(N),
        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    n = len(raw) // (W * H)
    if n < 6:
        return None
    frames = np.frombuffer(raw[: n * W * H], np.uint8).reshape(n, H, W)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []
    for i, f in enumerate(frames):
        for (x, y, w, h) in cascade.detectMultiScale(f, 1.1, 5, minSize=(16, 16)):
            faces.append((i, x + w / 2, y + h / 2, w))
    best, best_frames = None, set()
    for _, cx, cy, w in faces:
        near = [(i, a, b, c) for i, a, b, c in faces if abs(a - cx) < 0.5 * w and abs(b - cy) < 0.5 * w]
        fr = {i for i, *_ in near}
        if len(fr) > len(best_frames):
            best, best_frames = near, fr
    if not best or len(best_frames) < max(4, n * 0.35):
        return None
    fx = float(np.median([a for _, a, _, _ in best]))
    fy = float(np.median([b for _, _, b, _ in best]))
    fw = float(np.median([c for *_, c in best]))

    still = frames.astype(np.float32).std(axis=0) < 10  # True where nothing moves
    x0, x1 = int(fx - fw * 0.8), int(fx + fw * 0.8)
    y0, y1 = int(fy - fw * 0.8), int(fy + fw * 0.8)
    lim = dict(l=max(0, int(fx - 6 * fw)), r=min(W, int(fx + 6 * fw)),
               t=max(0, int(fy - 4 * fw)), b=min(H, int(fy + 4 * fw)))
    grew = True
    while grew:
        grew = False
        if x0 > lim["l"] and still[y0:y1, x0 - 1].mean() > 0.55:
            x0 -= 1; grew = True
        if x1 < lim["r"] and still[y0:y1, x1].mean() > 0.55:
            x1 += 1; grew = True
        if y0 > lim["t"] and still[y0 - 1, x0:x1].mean() > 0.55:
            y0 -= 1; grew = True
        if y1 < lim["b"] and still[y1, x0:x1].mean() > 0.55:
            y1 += 1; grew = True
    w, h = x1 - x0, y1 - y0
    if not (2.2 * fw <= w <= 12 * fw and 1.6 * fw <= h <= 9 * fw):
        # Room behind the streamer wasn't still enough: take a sensible box around the face.
        w, h = 4.5 * fw, 2.9 * fw
        x0, y0 = max(0, fx - w / 2), max(0, fy - h * 0.45)
        w, h = min(w, W - x0), min(h, H - y0)
    return (x0 / W, y0 / H, w / W, h / H)


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


def build_ass(words: list[dict], hook: str, length: float, style: str, seam: int | None = None) -> str:
    """seam: y of the line between face cam and gameplay (facecam layout). When set, the hook
    becomes a big yellow headline on that line for the whole clip and captions move lower."""
    st = dict(STYLES[style])
    if seam:
        st["cap_y"] = max(st["cap_y"], seam + 420)
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
Style: Head,Arial Black,84,{ass_color('FFD21F')},{ass_color('FFD21F')},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,7,3,5,50,50,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook and seam:
        h = ass_escape(hook.upper())
        lines.append(f"Dialogue: 2,{fmt_t(0)},{fmt_t(length)},Head,,0,0,0,,{{\\pos(540,{seam + 90})}}{h}")
    elif hook:
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
                style: str, layout: str, cwd: Path, cam=None, seam: int = 0, viral=None):
    st = STYLES[style]
    if layout == "facecam" and cam:
        x, y, w, h = cam
        bot = 1920 - seam
        vf = ("[0:v]split=2[a][b];"
              f"[a]crop=w=iw*{w:.4f}:h=ih*{h:.4f}:x=iw*{x:.4f}:y=ih*{y:.4f},"
              f"scale=1080:{seam}:force_original_aspect_ratio=increase,crop=1080:{seam},setsar=1[top];"
              f"[b]scale=1080:{bot}:force_original_aspect_ratio=increase,crop=1080:{bot},setsar=1[bot];"
              "[top][bot]vstack=2[base]")
        st = dict(st, border=None)
    elif layout in ("crop", "facecam"):
        vf = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[base]"
    else:  # blur: whole frame in the middle, blurred copy fills the screen
        vf = ("[0:v]split=2[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              "boxblur=24:2,eq=brightness=-0.10:saturation=1.2[bg];"
              "[b]scale=1080:1920:force_original_aspect_ratio=decrease,"
              "scale='if(gt(a,1),1180,iw)':-2,crop='min(iw,1080)':ih[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]")
    if viral is not None:
        render_viral(src, offset, length, vf, ass_name, out, st, cwd, seam, viral)
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{min(1.5, length / 2):.2f}",
             "-i", str(out), "-frames:v", "1", "-vf", "scale=360:-2", str(out.with_suffix(".jpg"))])
        return
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


# ------------------------------------------------------------ viral edit
FPS = 30
PUNCH_ZOOM, PUNCH_IN, PUNCH_HOLD, PUNCH_OUT = 1.10, 0.35, 0.8, 0.45
INTRO_ZOOM, INTRO_DUR = 1.08, 0.7
BAR_COLOR = (255, 210, 31)  # same yellow as the facecam headline


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def zoom_at(t: float, punches: list[float]) -> float:
    """Zoom factor at time t: a gentle settle from 1.08 at the start, plus eased punch-ins."""
    z = 1.0 + (INTRO_ZOOM - 1.0) * (1 - smoothstep(t / INTRO_DUR))
    for p in punches:
        if p <= t < p + PUNCH_IN:
            z = max(z, 1 + (PUNCH_ZOOM - 1) * smoothstep((t - p) / PUNCH_IN))
        elif p + PUNCH_IN <= t < p + PUNCH_IN + PUNCH_HOLD:
            z = max(z, PUNCH_ZOOM)
        elif p + PUNCH_IN + PUNCH_HOLD <= t < p + PUNCH_IN + PUNCH_HOLD + PUNCH_OUT:
            k = (t - p - PUNCH_IN - PUNCH_HOLD) / PUNCH_OUT
            z = max(z, PUNCH_ZOOM - (PUNCH_ZOOM - 1) * smoothstep(k))
    return z


def find_punches(pcm, sr: int, length: float, limit: int = 3) -> list[float]:
    """Loudest moments of the clip (shouts, explosions), at least 5 s apart, as punch-in times."""
    import numpy as np
    hop = int(sr * 0.1)
    n = len(pcm) // hop
    if n < 30:
        return []
    rms = np.sqrt((pcm[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    db = 20 * np.log10(rms)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    score = db - np.median(db)
    out: list[float] = []
    for i in np.argsort(-score):
        t = i * 0.1 - 0.15  # start pushing in just before the peak
        if score[i] < 6 or len(out) >= limit:
            break
        if 1.2 <= t <= length - 1.8 and all(abs(t - o) >= 5 for o in out):
            out.append(round(float(t), 2))
    return sorted(out)


def make_whoosh(path: Path):
    if not path.exists():
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "anoisesrc=d=0.5:c=pink:r=48000:a=0.6",
             "-af", "highpass=f=400,lowpass=f=4500,afade=t=in:d=0.3:curve=exp,"
                    "afade=t=out:st=0.3:d=0.2,volume=-14dB", "-ac", "2", str(path)])


def has_audio(path: Path) -> bool:
    return bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                "stream=index", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True).stdout.strip())


def render_viral(src: Path, offset: float, length: float, vf: str, ass_name: str, out: Path,
                 st: dict, cwd: Path, seam: int, viral: dict):
    """Layout (ffmpeg) -> smooth zooms + progress bar (Python, sub-pixel) -> captions, sound (ffmpeg)."""
    import cv2
    import numpy as np

    W, H = 1080, 1920
    punches = viral.get("punches", [])
    vf1 = vf + ";[base]eq=contrast=1.06:saturation=1.18,fps=30[pre]"
    p1 = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}", "-t", f"{length:.3f}",
         "-i", str(src), "-filter_complex", vf1, "-map", "[pre]", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    vchain = "[0:v]"
    if st["border"]:
        vchain += f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{st['border']}@0.95:t=14,"
    vchain += f"ass={ass_name}[v]"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    audio = has_audio(src)
    fc = vchain
    if audio:
        cmd += ["-ss", f"{offset:.3f}", "-t", f"{length:.3f}", "-i", str(src)]
        fc += ";[1:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[ga]"
        if punches:
            whoosh = cwd / "whoosh.wav"
            make_whoosh(whoosh)
            cmd += ["-i", str(whoosh)]
            fc += f";[2:a]asplit={len(punches)}" + "".join(f"[w{k}]" for k in range(len(punches)))
            for k, p in enumerate(punches):
                ms = max(0, int((p - 0.2) * 1000))
                fc += f";[w{k}]adelay={ms}|{ms}[d{k}]"
            fc += ";[ga]" + "".join(f"[d{k}]" for k in range(len(punches)))
            fc += f"amix=inputs={len(punches) + 1}:normalize=0:duration=first[a]"
        else:
            fc += ";[ga]anull[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]"]
    if audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]
    p2 = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Zoom only the gameplay when the face cam sits on top, so the streamer's face stays still.
    y0 = seam if seam else 0
    rh = H - y0
    cx, cy = W / 2, rh / 2
    frame_bytes = W * H * 3
    total = max(1, int(round(length * FPS)))
    i = 0
    try:
        while True:
            buf = p1.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
            t = i / FPS
            z = zoom_at(t, punches)
            if z > 1.0005:
                region = fr[y0:]
                m = np.float32([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy]])
                fr[y0:] = cv2.warpAffine(region, m, (W, rh), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
            bar = int(W * min(1.0, (i + 1) / total))
            fr[H - 12:, :bar] = BAR_COLOR
            p2.stdin.write(fr.tobytes())
            i += 1
    finally:
        p2.stdin.close()
        e2 = p2.stderr.read().decode(errors="replace")
        p2.wait()
        p1.stdout.close()
        e1 = p1.stderr.read().decode(errors="replace")
        p1.wait()
    if p2.returncode != 0 or i == 0:
        raise RuntimeError(f"viral render failed: {(e2 or e1)[-600:]}")


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
    """Headline: a short line said in the clip; else its first words; the title only as a last resort."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    good = [s for s in sentences if 12 <= len(s) <= 60]
    if good:
        hook = good[0]
    elif text.strip():
        words = text.split()
        hook = " ".join(words[:8]) + ("..." if len(words) > 8 else "")
    else:
        hook = re.sub(r"[_\-]+", " ", title)
    hook = re.sub(r"\s+", " ", hook).strip()
    return (hook[:57] + "...") if len(hook) > 60 else hook


# ------------------------------------------------------------------ job
def process(job_id: str, opts: dict):
    jdir = JOBS_DIR / job_id
    try:
        n, length = opts["num_clips"], opts["clip_len"]
        style, layout, captions = opts["style"], opts["layout"], opts["captions"]
        viral = opts.get("viral", False)
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
        skip_outro = min(40.0, duration * 0.05) if duration > 120 else 0.0
        windows = pick_windows(scores, duration, n, length, skip_intro, skip_outro)
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
                    # Prefer plain https H.264 <=1080p: HLS / VP9-HDR streams can't be section-cut reliably.
                    "format": "bv*+ba/b",
                    "format_sort": ["res:1080", "proto:https", "vcodec:avc", "acodec:aac"],
                    "merge_output_format": "mp4",
                    "download_ranges": download_range_func(None, [(s, min(duration, e + pad))]),
                    "outtmpl": str(jdir / f"raw_{i}.%(ext)s"),
                }, page_url)
                src = next(f for f in jdir.glob(f"raw_{i}.*") if not f.name.endswith(".part"))
                offset, avail = 0.0, probe_duration(src)
            want = e - s

            # 4. transcribe + trim the end on a finished sentence
            words, text, pcm = [], "", None
            if captions or viral:
                wav = jdir / f"clip_{i}.wav"
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}",
                     "-t", f"{avail:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
                pcm = load_wav(wav)
                wav.unlink(missing_ok=True)
            if captions:
                update(job_id, stage=f"Clip {i}/{len(windows)}: writing captions", progress=int(base + per * 0.35))
                segs = list(get_whisper().transcribe(pcm, word_timestamps=True, vad_filter=True)[0])
                if not segs:  # VAD drops singing / speech over loud game audio: retry without it
                    segs = list(get_whisper().transcribe(pcm, word_timestamps=True)[0])
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

            # 4b. viral edit: open on the first word (no dead air), find punch-in moments
            vplan = None
            if viral:
                lead = words[0]["start"] - 0.25 if words else 0.0
                if lead > 0.6 and want - lead >= max(10.0, want * 0.6):
                    offset += lead
                    want -= lead
                    for w in words:
                        w["start"] -= lead
                        w["end"] -= lead
                    if pcm is not None:
                        pcm = pcm[int(lead * 16000):]
                vplan = {"punches": find_punches(pcm, 16000, want) if pcm is not None else []}

            # 5. render
            update(job_id, stage=f"Clip {i}/{len(windows)}: editing 9:16", progress=int(base + per * 0.6))
            hook = make_hook(title, text) if opts["hook"] else ""
            cam, seam = None, 0
            if layout == "facecam":
                cam = find_facecam(src, offset, want)
                if cam:
                    sw, sh = probe_size(src)
                    aspect = (cam[2] * sw) / max(1.0, cam[3] * sh)
                    seam = int(round(min(820, max(560, 1080 / aspect)) / 2) * 2)
            (jdir / f"clip_{i}.ass").write_text(build_ass(words, hook, want, style, seam or None),
                                                encoding="utf-8")
            out = jdir / f"clip_{i}.mp4"
            render_clip(src, offset, want, f"clip_{i}.ass", out, style, layout, jdir, cam, seam, vplan)
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
    num_clips: int = Form(10),
    clip_len: int = Form(30),
    style: str = Form("neon"),
    layout: str = Form("blur"),
    captions: bool = Form(True),
    hook: bool = Form(True),
    viral: bool = Form(True),
):
    ip = client_ip(request)
    with jobs_lock:
        if ip in active_by_ip:
            raise HTTPException(429, "You already have a job running. Wait for it to finish.")
        queued = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
        if queued >= MAX_QUEUE:
            raise HTTPException(503, "The site is busy right now. Try again in a few minutes.")
    if style not in STYLES or layout not in ("blur", "crop", "facecam"):
        raise HTTPException(400, "Unknown style or layout.")
    num_clips = max(1, min(15, num_clips))
    clip_len = max(15, min(90, clip_len))

    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    jdir.mkdir()
    opts = dict(num_clips=num_clips, clip_len=float(clip_len), style=style, layout=layout,
                captions=captions, hook=hook, viral=viral)
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
