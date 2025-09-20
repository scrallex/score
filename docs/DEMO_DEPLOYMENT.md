# STM Demo Deployment Guide

This note documents how to run the Structural Intelligence Engine demo stack in
the `score/` directory without touching the legacy /sep autotrader assets.

## Components

- **Demo payload generator** – `demo/standalone.py` produces
  `demo/demo_payload.json` and syncs key MMS figures into
  `webapp/assets/`.
- **FastAPI service** – `stm_demo.api` exposes `/api/demo` endpoints and keeps
  the payload fresh on startup.
- **Static frontend** – `webapp/` renders the Pattern Prophet, Twin Finder, and
  Context Refinery summaries using vanilla JS.
- **Docker stack** – `docker-compose.demo.yml` launches the API and Nginx
  frontend. Certificates are reused by bind mounting `/etc/letsencrypt`.

## Local workflow

```bash
# 1) Generate (or refresh) the demo payload & assets
PYTHONPATH=src python demo/standalone.py --no-pretty

# 2) Run the API directly (development)
python -m stm_demo --reload --port 8000

# 3) Open the static site
npx serve webapp  # or use any static file server
```

Environment variables:

- `STM_DEMO_PAYLOAD` – optional override for the payload path
  (defaults to `demo/demo_payload.json`).
- `API_BASE` – set on the frontend via `window.__STM_CONFIG__` if the API is
  reverse-proxied somewhere other than `/api`.

## Docker deployment

```bash
# Build & start in the background
docker compose -f docker-compose.demo.yml up --build -d

# Tail combined logs
docker compose -f docker-compose.demo.yml logs -f

# Stop the stack
docker compose -f docker-compose.demo.yml down
```

Notes:

1. The compose file mounts `./analysis` and `./docs` read-only. Any new MMS
   artefacts you generate locally will be visible to the container on restart.
2. The frontend container expects `/etc/letsencrypt` to be mounted so it can
   serve `mxbikes.xyz` with the pre-existing certificates.
3. Both services use the default bridge network; adjust the published ports if
   the droplet already has 80/443 claimed by another stack.

## Next steps

- Collapse the compose pair into a single container once the UI stabilises.
- Add health/liveness endpoints to make upstream load balancers happier.
- Wire the websocket feed once the structural runtime is exposed from the core
  engine.
