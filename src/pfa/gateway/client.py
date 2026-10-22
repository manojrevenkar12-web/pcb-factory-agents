"""The client an agent uses inside its sandbox. It holds a session token and nothing else.

It retries network failures and 5xx answers with the same idempotency key, which is safe because
the gateway executes each key at most once. Deliberately light (httpx only): it is what a
sandbox imports.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Protocol

import httpx

from pfa.judges import Image


class Client(Protocol):
    cost_usd: float

    def read(self, tool: str, **args: Any) -> Any: ...
    def act(self, tool: str, key: str, **args: Any) -> dict: ...
    def judge(
        self, skill: str, text: str, data: dict | None = None, images: list[Image] | None = None
    ) -> dict: ...


class ToolFailed(Exception):
    def __init__(self, status: int, error: str, detail: str):
        super().__init__(f"{status} {error}: {detail}")
        self.status, self.error, self.detail = status, error, detail


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        retries: int = 5,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
        )
        self.retries = retries
        self.cost_usd = 0.0

    def _post(self, path: str, body: dict) -> dict:
        delay = 0.2
        for attempt in range(self.retries + 1):
            try:
                response = self.http.post(path, json=body)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                if response.status_code < 400:
                    return response.json()
                payload = (
                    response.json()
                    if response.headers.get("content-type", "").startswith("application/json")
                    else {"error": "http", "detail": response.text}
                )
                retryable = response.status_code >= 500 or payload.get("retryable")
                if not retryable or attempt == self.retries:
                    raise ToolFailed(
                        response.status_code, payload.get("error", "?"), payload.get("detail", "")
                    )
            time.sleep(delay)
            delay = min(delay * 2, 3.0)
        raise RuntimeError("unreachable")

    def read(self, tool: str, **args: Any) -> Any:
        return self._post(f"/v1/tools/{tool}", {"args": args})["result"]

    def act(self, tool: str, key: str, **args: Any) -> dict:
        """Returns {"result": ..., "replayed": bool}; replayed means it had already been done."""
        return self._post(f"/v1/tools/{tool}", {"args": args, "idempotency_key": key})

    def judge(
        self, skill: str, text: str, data: dict | None = None, images: list[Image] | None = None
    ) -> dict:
        body = {
            "skill": skill,
            "text": text,
            "data": data or {},
            "images": [
                {"data": base64.b64encode(i.data).decode(), "media_type": i.media_type, "label": i.label}
                for i in images or []
            ],
        }
        out = self._post("/v1/judge", body)
        self.cost_usd += out.get("cost_usd", 0.0)
        return out
