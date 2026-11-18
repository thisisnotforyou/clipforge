---
title: ClipForge
emoji: 🎬
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Paste a link, get edited 9:16 clips with captions
---

# ClipForge

Paste a YouTube / Twitch / Kick link (or upload a video) and get ready-to-post 9:16 clips.

- Finds moments from YouTube's "Most replayed" graph, otherwise from loudness spikes.
- Downloads only those moments, transcribes them (faster-whisper) and ends each clip on a finished sentence.
- Renders 1080x1920, word-by-word captions, a hook on top, -14 LUFS sound.
- Two styles: **Neon** (big captions, neon frame) and **Clean**. Two frames: full video on a blurred background, or zoom-fill.
- Caption + hashtags + creator credit per clip, `POSTING.md` ranked best first, and a zip of everything.

## Run it on your computer

Needs ffmpeg and python3.

```bash
./start.sh          # http://localhost:8765, only you
./start.sh --lan    # also phones/laptops on the same Wi-Fi
```

## Put it online for everyone: Oracle Cloud free VM (free forever)

1. Sign up at oracle.com/cloud/free (needs a card for verification, not charged on Always Free).
2. Compute → Instances → Create: image **Ubuntu 24.04**, shape **Ampere VM.Standard.A1.Flex** (4 OCPU, 24 GB), download the SSH private key.
3. Networking → your VCN → Default Security List → add ingress rules for TCP **80** and **443** from `0.0.0.0/0`.
4. SSH in and run:
   ```bash
   git clone https://github.com/thisisnotforyou/clipforge && cd clipforge && sudo bash deploy/server-setup.sh
   ```
5. Open `https://<ip-with-dashes>.sslip.io` (the script prints it). Run the same script again to update.

Extra settings (proxy, cookies, workers) go in `deploy/extra.env`, e.g. `YTDLP_PROXY=http://user:pass@host:port`.

## Or: Hugging Face Spaces (needs PRO, $9/month)

One Space runs the website and the clip engine together at one link.

1. huggingface.co → New Space → SDK **Docker** → Blank → CPU basic (free) → Create.
2. Push this repo to it: `git remote add space https://huggingface.co/spaces/<you>/<space>` then `git push space main`.
3. Wait for the build (about 5 min), then open `https://<you>-<space>.hf.space`.

Free Spaces sleep after 48 h with no visitors and wake on the next visit (about 1 min). Clips are kept until the Space restarts.

## Or: website on Vercel + engine on another host

The site has two parts:

| part | what it is | where it runs |
|---|---|---|
| website | `static/` (one HTML page) | **Vercel** (free) |
| clip engine | `app.py` + yt-dlp + Whisper + ffmpeg (`Dockerfile`) | a container host: **Railway**, **Render**, **Fly.io**, a VPS, or your own PC |

The engine can't run on Vercel: it needs ffmpeg, minutes of CPU per job and disk space for videos.

### 1. Engine (Railway example)
1. railway.app → New Project → Deploy from GitHub repo → pick this repo. It builds the `Dockerfile`.
2. Variables: `ALLOWED_ORIGINS=https://<your-site>.vercel.app`, and optionally `MAX_WORKERS=2`.
3. Add a volume mounted at `/data` (so clips survive restarts), and give it at least 2 GB RAM.
4. Settings → Networking → Generate domain. Check `https://<engine-domain>/health` returns `{"ok":true}`.

Render / Fly.io work the same way: deploy the Dockerfile, set the same variables.

### 2. Website (Vercel)
1. vercel.com → Add New → Project → import this repo. Leave framework as "Other"; `vercel.json` sets everything.
2. Environment variable: `API_BASE=https://<engine-domain>` (no trailing slash).
3. Deploy. Put the Vercel URL back into the engine's `ALLOWED_ORIGINS`.

### YouTube on cloud servers
YouTube often blocks datacenter IPs ("Sign in to confirm you're not a bot"). Fixes, easiest first:
- Run the engine on a home PC and expose it with Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8765`), then use that URL as `API_BASE`.
- Set `YTDLP_PROXY` to a residential proxy.
- Set `YTDLP_COOKIES=/data/cookies.txt` with cookies exported from a **throwaway** YouTube account (never your main one).
Uploading a video file always works.

## Settings (env vars)

| var | default | meaning |
|---|---|---|
| `MAX_WORKERS` | 1 | jobs rendered at the same time |
| `MAX_SOURCE_SECONDS` | 28800 | longest video accepted |
| `MAX_UPLOAD_MB` | 1024 | biggest upload |
| `JOB_TTL_HOURS` | 6 | clips deleted after this |
| `WHISPER_MODEL` | base | `small` / `medium` = better captions, slower |
| `ALLOWED_ORIGINS` | (none) | sites allowed to call the engine, e.g. your Vercel URL |
| `YTDLP_COOKIES` | (none) | path to a cookies.txt for YouTube |
| `YTDLP_PROXY` | (none) | proxy URL for downloads |
| `JOBS_DIR` | ./jobs | where clips are stored |

## Limits right now

- Live streams are refused (paste the link again once the stream ends).
- No facecam detection / split layout yet.
- On a cloud server (AWS, Render, etc.) YouTube often blocks downloads with a bot check; at home it works. Uploading the file always works.
- Only clip videos you have the right to use and credit the creator.
