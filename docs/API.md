# STM API Notes (Draft)

## Endpoints (current prototype)

- `POST /stm/seen` — recent windows matching a signature (stream runtime).
- `GET /stm/health` — streaming runtime status.
- Future: `POST /v1/propose`, `/v1/lead`, `/v1/onsets/autolabel` once the API layer wraps the batch utilities.

Implementation pending streaming/runtime work.
