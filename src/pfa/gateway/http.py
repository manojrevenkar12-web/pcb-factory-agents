"""The gateway over HTTP, and an in-process client with the same interface."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pfa.gateway.client import ToolFailed
from pfa.gateway.core import Gateway, GatewayError
from pfa.judges import Image


class ToolCall(BaseModel):
    args: dict[str, Any] = {}
    idempotency_key: str | None = None


class JudgeCall(BaseModel):
    skill: str
    text: str
    data: dict[str, Any] = {}
    images: list[dict[str, str]] = []  # {"data": base64, "media_type": ..., "label": ...}


def _token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    return authorization.removeprefix("Bearer ")


def app(gateway: Gateway) -> FastAPI:
    api = FastAPI(title="pfa tool gateway", version="1.0.0")

    @api.exception_handler(GatewayError)
    def _error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            {"error": type(exc).__name__, "detail": str(exc), "retryable": exc.retryable},
            status_code=exc.status,
        )

    @api.get("/healthz")
    def health() -> dict:
        return {"ok": True}

    @api.get("/v1/tools")
    def tools(authorization: str | None = Header(None)) -> list[dict]:
        return gateway.list_tools(_token(authorization))

    @api.post("/v1/tools/{name}")
    def call(name: str, body: ToolCall, authorization: str | None = Header(None)) -> dict:
        return gateway.call(_token(authorization), name, body.args, body.idempotency_key)

    @api.post("/v1/judge")
    def judge(body: JudgeCall, authorization: str | None = Header(None)) -> dict:
        images = [
            Image(base64.b64decode(i["data"]), i.get("media_type", "image/png"), i.get("label", ""))
            for i in body.images
        ]
        return gateway.judge(_token(authorization), body.skill, body.text, body.data, images)

    return api


class LocalClient:
    """The same interface in-process, for evaluations and tests that do not need a sandbox."""

    def __init__(self, gateway: Gateway, token: str):
        self.gateway, self.token = gateway, token
        self.cost_usd = 0.0

    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except GatewayError as exc:  # same failure type as over HTTP
            raise ToolFailed(exc.status, type(exc).__name__, str(exc)) from exc

    def read(self, tool: str, **args: Any) -> Any:
        return self._guard(self.gateway.call, self.token, tool, args)["result"]

    def act(self, tool: str, key: str, **args: Any) -> dict:
        return self._guard(self.gateway.call, self.token, tool, args, key)

    def judge(
        self, skill: str, text: str, data: dict | None = None, images: list[Image] | None = None
    ) -> dict:
        out = self._guard(self.gateway.judge, self.token, skill, text, data, images)
        self.cost_usd += out.get("cost_usd", 0.0)
        return out
