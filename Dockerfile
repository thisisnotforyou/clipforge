# ClipForge: website + clip engine (yt-dlp + Whisper + ffmpeg) in one container.
# Works on Hugging Face Spaces (port 7860, runs as user 1000), Railway, Render, Fly.io.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fonts-liberation fonts-noto-core \
 && rm -rf /var/lib/apt/lists/*

# Non-root user (Hugging Face Spaces runs containers as uid 1000).
RUN useradd -m -u 1000 user
ENV HOME=/home/user HF_HOME=/opt/hf PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the Whisper model into the image so the first job doesn't wait for a download.
ARG WHISPER_MODEL=base
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN mkdir -p /opt/hf \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')" \
 && chmod -R a+rwX /opt/hf

COPY app.py .
COPY static ./static
RUN mkdir -p /data/jobs && chown -R user:user /data /app

USER user
ENV JOBS_DIR=/data/jobs PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
