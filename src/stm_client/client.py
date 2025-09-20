"""Typed HTTP client for the STM coprocessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - requests is required at runtime
    raise RuntimeError("stm-client requires the 'requests' package") from exc


class STMClientError(RuntimeError):
    """Raised when the STM service returns an error response."""


@dataclass
class STMClient:
    """Convenience wrapper around the STM HTTP API."""

    base_url: str
    timeout: float = 10.0
    session: Optional["requests.Session"] = None

    def _post(self, path: str, payload: Mapping[str, object]) -> MutableMapping[str, object]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        sess = self.session or requests
        response = sess.post(url, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise STMClientError(f"STM request failed ({response.status_code}): {response.text}")
        data = response.json()
        if not isinstance(data, MutableMapping):  # pragma: no cover - defensive
            raise STMClientError(f"STM returned unexpected payload: {data!r}")
        return data

    # Public endpoints -----------------------------------------------------

    def enrich(self, text: str, *, config: Optional[Mapping[str, object]] = None) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {"text": text}
        if config:
            payload["config"] = dict(config)
        return self._post("stm/enrich", payload)

    def dilution(self, tokens: Iterable[str]) -> MutableMapping[str, object]:
        return self._post("stm/dilution", {"tokens": list(tokens)})

    def seen(self, window: Iterable[str]) -> MutableMapping[str, object]:
        return self._post("stm/seen", {"window": list(window)})

    def propose(self, seeds: Iterable[str], *, limit: int | None = None) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {"seeds": list(seeds)}
        if limit is not None:
            payload["limit"] = int(limit)
        return self._post("stm/propose", payload)

    def lead(self, tokens: Iterable[str]) -> MutableMapping[str, object]:
        return self._post("stm/lead", {"tokens": list(tokens)})


__all__ = ["STMClient", "STMClientError"]
