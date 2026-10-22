"""The tool gateway: the only way an agent touches the business.

Agents run in sandboxes holding a session token, never a credential. Every tool call comes here,
where the gateway
- checks the token's signature, expiry and scopes;
- checks the task fence, so an agent whose task was re-claimed cannot act;
- runs reads directly, and runs actions through durable receipts, exactly once per key;
- in shadow mode records what the agent would have done, and does nothing;
- runs judge calls with the model credentials it holds, caches them durably, and records cost;
- traces every call with OpenTelemetry.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ValidationError

from pfa import receipts, workqueue
from pfa.gateway import tokens
from pfa.judges import Image, Judge, Request
from pfa.telemetry import tracer


class GatewayError(Exception):
    status = 400
    retryable = False


class PermissionDenied(GatewayError):
    status = 403


class StaleSession(GatewayError):
    """The task was re-claimed (fence moved on) or its lease ran out."""

    status = 409


class Conflict(GatewayError):
    status = 409


class NotFound(GatewayError):
    status = 404


class InvalidArguments(GatewayError):
    status = 422


class RemoteError(GatewayError):
    """A downstream system failed; the action may be retried with the same key."""

    status = 502
    retryable = True


@dataclass
class Context:
    conn: psycopg.Connection
    claims: tokens.Claims
    secrets: dict[str, str]
    key: str | None
    services: dict[str, Any]


@dataclass
class Tool:
    name: str
    effect: Literal["read", "action"]
    args: type[BaseModel]
    handler: Callable[[Context, BaseModel], Any]
    description: str = ""


@dataclass
class Skill:
    """A judgement the gateway can run: its prompt (skills/<name>.md), answer schema and judge."""

    name: str
    schema: type[BaseModel]
    judge: Judge


@dataclass
class Gateway:
    database_url: str
    signing_key: bytes
    secrets: dict[str, str] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)
    pool_size: int = 8

    def __post_init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.skills: dict[str, Skill] = {}
        self.pool = ConnectionPool(
            self.database_url,
            min_size=1,
            max_size=self.pool_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        self.tracer = tracer("pfa.gateway")

    def close(self) -> None:
        self.pool.close()

    # ---- registration ----------------------------------------------------------------------------

    def tool(
        self, name: str, effect: Literal["read", "action"], args: type[BaseModel], description: str = ""
    ):
        def register(fn: Callable[[Context, BaseModel], Any]):
            self.tools[name] = Tool(name, effect, args, fn, description or (fn.__doc__ or "").strip())
            return fn

        return register

    def add_skill(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def mint(
        self,
        scopes: list[str],
        ttl_s: float = 900,
        task_id: int | None = None,
        fence: int | None = None,
        shadow: bool = False,
        session_id: str | None = None,
    ) -> str:
        claims = tokens.Claims(
            session_id or uuid.uuid4().hex[:12], scopes, time.time() + ttl_s, task_id, fence, shadow
        )
        return tokens.mint(self.signing_key, claims)

    def list_tools(self, token: str) -> list[dict]:
        claims = self._claims(token)
        return [
            {
                "name": t.name,
                "effect": t.effect,
                "description": t.description,
                "input_schema": t.args.model_json_schema(),
            }
            for t in self.tools.values()
            if claims.allows(t.name)
        ]

    # ---- calls -------------------------------------------------------------------------------------

    def _claims(self, token: str) -> tokens.Claims:
        try:
            return tokens.verify(self.signing_key, token)
        except tokens.TokenError as exc:
            raise PermissionDenied(str(exc)) from exc

    def _check_fence(self, conn: psycopg.Connection, claims: tokens.Claims) -> None:
        if claims.task_id is None:
            return
        fence = workqueue.current_fence(conn, claims.task_id)
        if fence != claims.fence:
            raise StaleSession(f"task {claims.task_id}: fence {claims.fence} is no longer current")

    def call(self, token: str, name: str, args: dict, idempotency_key: str | None = None) -> dict:
        claims = self._claims(token)
        tool = self.tools.get(name)
        if tool is None:
            raise NotFound(f"no tool {name}")
        if not claims.allows(name):
            raise PermissionDenied(f"session may not call {name}")
        try:
            parsed = tool.args.model_validate(args)
        except ValidationError as exc:
            raise InvalidArguments(str(exc.errors()[:3])) from exc
        if tool.effect == "action" and not idempotency_key:
            raise InvalidArguments(f"{name} is an action and needs an idempotency key")

        with self.tracer.start_as_current_span(f"tool {name}") as span:
            span.set_attribute("pfa.session", claims.session_id)
            span.set_attribute("pfa.tool.effect", tool.effect)
            if claims.task_id is not None:
                span.set_attribute("pfa.task", claims.task_id)
            with self.pool.connection() as conn, conn.transaction():
                self._check_fence(conn, claims)
                ctx = Context(conn, claims, self.secrets, None, self.services)
                if tool.effect == "read":
                    return {"result": tool.handler(ctx, parsed), "replayed": False}
                key = f"{name}:{idempotency_key}"
                canonical = parsed.model_dump(mode="json")
                if claims.shadow:
                    conn.execute(
                        """INSERT INTO agents.shadow_actions (session_id, tool, key, args)
                           VALUES (%s, %s, %s, %s) ON CONFLICT (tool, key) DO NOTHING""",
                        (claims.session_id, name, key, Jsonb(canonical)),
                    )
                    span.set_attribute("pfa.shadow", True)
                    return {"result": {"shadow": True}, "replayed": False}
                try:
                    receipt = receipts.begin(conn, key, name, canonical, claims.task_id, claims.session_id)
                except receipts.IdempotencyConflict as exc:
                    raise Conflict(str(exc)) from exc
                span.set_attribute("pfa.receipt.attempts", receipt.attempts)
                if receipt.done:
                    span.set_attribute("pfa.replayed", True)
                    return {"result": receipt.result, "replayed": True}
                ctx.key = key
                result = tool.handler(ctx, parsed)
                receipts.finish(conn, key, result)
                span.set_attribute("pfa.replayed", False)
                return {"result": result, "replayed": False}

    def judge(
        self,
        token: str,
        skill_name: str,
        text: str,
        data: dict | None = None,
        images: list[Image] | None = None,
    ) -> dict:
        claims = self._claims(token)
        if not claims.allows(f"judge.{skill_name}"):
            raise PermissionDenied(f"session may not use skill {skill_name}")
        skill = self.skills.get(skill_name)
        if skill is None:
            raise NotFound(f"no skill {skill_name}")
        request = Request(skill_name, text, skill.schema, images or [], data or {})
        config, digest = skill.judge.config_hash(request), request.input_hash()
        with self.tracer.start_as_current_span(f"judge {skill_name}") as span:
            span.set_attribute("pfa.judge", skill.judge.name)
            span.set_attribute("pfa.config_hash", config)
            with self.pool.connection() as conn:
                cached = conn.execute(
                    "SELECT answer FROM agents.judge_cache WHERE config_hash = %s AND input_hash = %s",
                    (config, digest),
                ).fetchone()
                if cached:
                    self._record(conn, claims, skill, config, digest, None, cached=True)
                    span.set_attribute("pfa.cached", True)
                    return {"answer": cached["answer"], "cached": True, "cost_usd": 0.0, "error": None}
                verdict = skill.judge.judge(request)
                if verdict.ok:
                    conn.execute(
                        """INSERT INTO agents.judge_cache (config_hash, input_hash, answer) VALUES (%s, %s, %s)
                           ON CONFLICT DO NOTHING""",
                        (config, digest, Jsonb(verdict.answer.model_dump(mode="json"))),
                    )
                self._record(conn, claims, skill, config, digest, verdict)
                conn.commit()
            span.set_attribute("pfa.cost_usd", verdict.cost_usd)
            span.set_attribute("pfa.tokens.input", verdict.input_tokens)
            span.set_attribute("pfa.tokens.output", verdict.output_tokens)
            return {
                "answer": verdict.answer.model_dump(mode="json") if verdict.ok else None,
                "cached": False,
                "cost_usd": verdict.cost_usd,
                "error": verdict.error,
            }

    def _record(self, conn, claims, skill, config, digest, verdict, cached=False) -> None:
        conn.execute(
            """INSERT INTO agents.judge_calls (session_id, task_id, skill, config_hash, input_hash, judge, model,
                   input_tokens, output_tokens, cost_usd, latency_ms, cached, error)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                claims.session_id,
                claims.task_id,
                skill.name,
                config,
                digest,
                skill.judge.name,
                skill.judge.model,
                verdict.input_tokens if verdict else 0,
                verdict.output_tokens if verdict else 0,
                verdict.cost_usd if verdict else 0,
                int((verdict.latency_s if verdict else 0) * 1000),
                cached,
                verdict.error if verdict else None,
            ),
        )
        conn.commit()
