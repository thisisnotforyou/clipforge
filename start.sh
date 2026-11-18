#!/usr/bin/env bash
# Start ClipForge on this computer.  Usage: ./start.sh          (only you:  http://localhost:8765)
#                                           ./start.sh --lan    (phones/PCs on your Wi-Fi too)
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg is missing. Install it first (brew install ffmpeg / apt install ffmpeg)."; exit 1; }
if [ ! -x .venv312/bin/uvicorn ]; then
  python3 -m pip install --user -q uv
  python3 -m uv venv --python 3.12 .venv312
  VIRTUAL_ENV=.venv312 python3 -m uv pip install -r requirements.txt
fi
HOST=127.0.0.1
if [ "$1" = "--lan" ]; then
  HOST=0.0.0.0
  IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
  echo "Open on other devices: http://$IP:8765"
fi
echo "Open: http://localhost:8765"
exec .venv312/bin/uvicorn app:app --host "$HOST" --port 8765
