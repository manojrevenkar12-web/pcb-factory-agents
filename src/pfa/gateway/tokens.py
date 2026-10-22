"""Session tokens: what a sandboxed agent holds instead of credentials.

A token names the session, the task and fence it works under, the tools it may call and when it
expires, signed with a key only the gateway knows. Stealing one from a sandbox gives access to a
few tools for one task for a few minutes, and nothing once the task is re-claimed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field


class TokenError(Exception):
    pass


@dataclass
class Claims:
    session_id: str
    scopes: list[str]
    expires_at: float
    task_id: int | None = None
    fence: int | None = None
    shadow: bool = False
    extra: dict = field(default_factory=dict)

    def allows(self, tool: str) -> bool:
        return any(tool == s or (s.endswith(".*") and tool.startswith(s[:-1])) for s in self.scopes)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mint(key: bytes, claims: Claims) -> str:
    body = _b64(json.dumps(asdict(claims), sort_keys=True).encode())
    sig = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify(key: bytes, token: str, now: float | None = None) -> Claims:
    try:
        body, sig = token.split(".")
    except ValueError as exc:
        raise TokenError("malformed token") from exc
    expected = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise TokenError("bad signature")
    claims = Claims(**json.loads(_unb64(body)))
    if (now or time.time()) > claims.expires_at:
        raise TokenError("token expired")
    return claims
