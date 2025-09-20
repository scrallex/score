#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.demo.yml"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "[deploy] This script must run as root (needs /etc/letsencrypt + docker access)." >&2
  exit 1
fi

command -v docker >/dev/null 2>&1 || { echo "[deploy] docker is required" >&2; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "[deploy] docker compose plugin is required" >&2; exit 1; }

cert_dir="/etc/letsencrypt"
required=("options-ssl-nginx.conf" "ssl-dhparams.pem" "live/mxbikes.xyz/fullchain.pem" "live/mxbikes.xyz/privkey.pem")
missing=()
for f in "${required[@]}"; do
  if [[ ! -e "$cert_dir/$f" ]]; then
    missing+=("$cert_dir/$f")
  fi
  done

if [[ ${#missing[@]} -ne 0 ]]; then
  echo "[deploy] Missing TLS assets:" >&2
  printf ' - %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "[deploy] Generating latest demo payload"
PYTHONPATH="$ROOT_DIR/src" python "$ROOT_DIR/demo/standalone.py" --no-pretty

if docker compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
  echo "[deploy] Stopping existing stack"
  docker compose -f "$COMPOSE_FILE" down --remove-orphans
fi

echo "[deploy] Building and starting stack"
docker compose -f "$COMPOSE_FILE" up --build -d

echo "[deploy] Current container status"
docker compose -f "$COMPOSE_FILE" ps

for svc in stm-demo-frontend stm-demo-api; do
  echo "[deploy] Last logs for $svc"
  docker logs "$svc" --tail 20 || true
  done

echo "[deploy] Deploy finished"
