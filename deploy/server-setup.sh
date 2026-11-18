#!/usr/bin/env bash
# One-shot install of ClipForge on a fresh Ubuntu server (made for Oracle Cloud's free Ampere VM).
#   git clone https://github.com/thisisnotforyou/clipforge && cd clipforge && sudo bash deploy/server-setup.sh
# Optional: DOMAIN=clips.example.com sudo -E bash deploy/server-setup.sh   (default: <ip>.sslip.io)
# Running it again updates the app to the latest code.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing Docker"
if ! command -v docker >/dev/null; then
  apt-get update -y
  apt-get install -y docker.io docker-compose-v2 git curl
  systemctl enable --now docker
fi

echo "== Opening ports 80 and 443 in the server firewall"
# Oracle's Ubuntu images block everything except SSH with iptables, on top of the cloud security list.
for p in 80 443; do
  iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || iptables -I INPUT 5 -p tcp --dport $p -j ACCEPT
done
if command -v netfilter-persistent >/dev/null; then netfilter-persistent save; fi

if [ -z "${DOMAIN:-}" ]; then
  IP=$(curl -fsS https://api.ipify.org)
  DOMAIN="${IP//./-}.sslip.io"   # free hostname that points at this IP, so Caddy can get an HTTPS certificate
fi
echo "DOMAIN=$DOMAIN" > deploy/.env
[ -f deploy/extra.env ] && cat deploy/extra.env >> deploy/.env

echo "== Building and starting (first build takes 5-10 minutes)"
git pull --ff-only 2>/dev/null || true
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build

echo
echo "ClipForge is starting at: https://$DOMAIN"
echo "Logs: sudo docker compose -f deploy/docker-compose.yml logs -f"
