"""Judges: whatever returns a structured answer to a question that needs judgement.

A judge is a model (Claude), plain code (a rule or classical-vision baseline), or a wrapper
(recording, replay, fault injection). They share one interface so the same evaluation, the same
checks and the same platform run against any of them, and a change of model, prompt (skill) or
effort level is visible as a change of `config_hash`.

Agents judge, code writes: a judge only ever returns an answer validated against a schema.
What happens next is decided and done by plain code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from pfa.config import SKILLS

# Anthropic list prices, USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing
# (checked October 2026). Haiku 5.5 is quoted for prompts up to 100k tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Vision input is billed as tokens: one per 28x28 patch, after downscaling to the model's limit
# (platform.claude.com/docs/en/build-with-claude/vision; high-resolution tier).
MAX_LONG_EDGE = 2576
MAX_IMAGE_TOKENS = 4784


def image_tokens(width: int, height: int) -> int:
    scale = min(1.0, MAX_LONG_EDGE / max(width, height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    tokens = math.ceil(w / 28) * math.ceil(h / 28)
    while tokens > MAX_IMAGE_TOKENS:  # downscale further until under the token cap
        w, h = round(w * 0.95), round(h * 0.95)
        tokens = math.ceil(w / 28) * math.ceil(h / 28)
    return tokens


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        return 0.0
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def load_skill(name: str) -> str:
    return (SKILLS / f"{name}.md").read_text()


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


@dataclass
class Image:
    data: bytes
    media_type: str = "image/png"
    label: str = ""
    width: int = 0
    height: int = 0


@dataclass
class Request:
    skill: str
    text: str
    schema: type[BaseModel]
    images: list[Image] = field(default_factory=list)
    #: Structured inputs for code judges; models see `text` and `images`.
    data: dict[str, Any] = field(default_factory=dict)

    def input_hash(self) -> str:
        h = hashlib.sha256(self.text.encode())
        for image in self.images:
            h.update(hashlib.sha256(image.data).digest())
        h.update(json.dumps(self.data, sort_keys=True, default=str).encode())
        return h.hexdigest()


@dataclass
class Verdict:
    answer: BaseModel | None
    judge: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.answer is not None and self.error is None


class Judge(Protocol):
    name: str
    model: str

    def config_hash(self, request: Request) -> str: ...

    def judge(self, request: Request) -> Verdict: ...


def _config_hash(*parts: object) -> str:
    return sha(json.dumps([str(p) for p in parts]))[:16]


def strict_schema(model: type[BaseModel]) -> dict:
    """A pydantic model's JSON schema in the subset structured outputs accept."""
    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "title",
    }

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: clean(v) for k, v in node.items() if k not in unsupported}
            if out.get("type") == "object" or "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}))
            if out.get("minItems", 0) > 1:
                out["minItems"] = 1
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(model.model_json_schema())


# ---- code judges ---------------------------------------------------------------------------------


class FunctionJudge:
    """Plain code as a judge: a rule set or a classical-vision baseline. Free and deterministic."""

    def __init__(self, name: str, fn: Callable[[Request], BaseModel], version: str = "1"):
        self.name, self.fn, self.version = name, fn, version
        self.model = f"code:{name}"

    def config_hash(self, request: Request) -> str:
        return _config_hash(self.model, self.version, request.schema.__name__)

    def judge(self, request: Request) -> Verdict:
        started = time.perf_counter()
        try:
            answer = self.fn(request)
            error = None
        except Exception as exc:  # a baseline that cannot answer refuses, like a model would
            answer, error = None, f"{type(exc).__name__}: {exc}"
        return Verdict(answer, self.name, self.model, latency_s=time.perf_counter() - started, error=error)


# ---- Claude ------------------------------------------------------------------------------------------


class ClaudeJudge:
    """Claude with the skill (plain Markdown) as system prompt and a JSON-schema answer.

    Uses structured outputs (`output_config.format`), checks the stop reason, validates the answer
    with pydantic and retries once with the validation error. Credentials stay with whoever builds
    the client: in production that is the tool gateway, never the sandboxed agent.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        effort: str = "low",
        max_tokens: int = 1024,
        client: Any = None,
        retries: int = 1,
    ):
        self.model, self.effort, self.max_tokens, self.retries = model, effort, max_tokens, retries
        self.name = f"claude:{model}:{effort}"
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # optional dependency: pip install .[llm]

            self._client = anthropic.Anthropic()
        return self._client

    def config_hash(self, request: Request) -> str:
        return _config_hash(
            self.model,
            self.effort,
            sha(load_skill(request.skill)),
            json.dumps(strict_schema(request.schema), sort_keys=True),
        )

    def _content(self, request: Request, feedback: str | None) -> list[dict]:
        content: list[dict] = []
        for image in request.images:  # images first, as the vision docs recommend
            if image.label:
                content.append({"type": "text", "text": image.label})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.b64encode(image.data).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": request.text})
        if feedback:
            content.append({"type": "text", "text": feedback})
        return content

    def judge(self, request: Request) -> Verdict:
        started = time.perf_counter()
        tokens_in = tokens_out = 0
        feedback = None
        error = "no attempt"
        for _ in range(self.retries + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=load_skill(request.skill),
                messages=[{"role": "user", "content": self._content(request, feedback)}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": strict_schema(request.schema)},
                },
            )
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            if response.stop_reason in ("refusal", "max_tokens"):
                error = f"stop_reason={response.stop_reason}"
                feedback = None
                continue
            text = next((b.text for b in response.content if b.type == "text"), "")
            try:
                answer = request.schema.model_validate_json(text)
                return Verdict(
                    answer,
                    self.name,
                    self.model,
                    tokens_in,
                    tokens_out,
                    cost_usd(self.model, tokens_in, tokens_out),
                    time.perf_counter() - started,
                )
            except ValidationError as exc:
                error = f"invalid answer: {exc.errors()[:3]}"
                feedback = f"Your previous answer was invalid: {error}. Answer again, following the schema."
        return Verdict(
            None,
            self.name,
            self.model,
            tokens_in,
            tokens_out,
            cost_usd(self.model, tokens_in, tokens_out),
            time.perf_counter() - started,
            error=error,
        )


# ---- wrappers ---------------------------------------------------------------------------------------


class NoisyJudge:
    """Fault injection: corrupts a share of the inner judge's answers.

    Used to prove the checks, not the judge: with a judge that is wrong 20% of the time, the code
    that verifies answers against a second source must still never let a wrong action through.
    """

    def __init__(
        self,
        inner: Judge,
        rate: float,
        corrupt: Callable[[BaseModel, random.Random], BaseModel],
        seed: int = 0,
    ):
        self.inner, self.rate, self.corrupt = inner, rate, corrupt
        self.rng = random.Random(seed)
        self.name = f"noisy({inner.name}, {rate:.0%})"
        self.model = inner.model
        self.corrupted = 0

    def config_hash(self, request: Request) -> str:
        return _config_hash("noisy", self.rate, self.inner.config_hash(request))

    def judge(self, request: Request) -> Verdict:
        verdict = self.inner.judge(request)
        if verdict.ok and self.rng.random() < self.rate:
            verdict.answer = self.corrupt(verdict.answer.model_copy(deep=True), self.rng)
            self.corrupted += 1
        return verdict


class RecordingJudge:
    """Records every verdict to a JSONL cassette so a run can be replayed exactly and for free."""

    def __init__(self, inner: Judge, path: Path):
        self.inner, self.path = inner, path
        self.name, self.model = inner.name, inner.model
        path.parent.mkdir(parents=True, exist_ok=True)

    def config_hash(self, request: Request) -> str:
        return self.inner.config_hash(request)

    def judge(self, request: Request) -> Verdict:
        verdict = self.inner.judge(request)
        record = {
            "key": f"{self.config_hash(request)}:{request.input_hash()}",
            "answer": verdict.answer.model_dump() if verdict.answer else None,
            "judge": verdict.judge,
            "model": verdict.model,
            "input_tokens": verdict.input_tokens,
            "output_tokens": verdict.output_tokens,
            "cost_usd": verdict.cost_usd,
            "latency_s": verdict.latency_s,
            "error": verdict.error,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return verdict


class ReplayJudge:
    """Answers from a cassette. A request that was never recorded is an error, not a guess."""

    def __init__(self, path: Path, config_hash: str):
        self.path, self._config = path, config_hash
        self.records = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            self.records[r["key"]] = r
        first = next(iter(self.records.values()), {})
        self.name = f"replay:{first.get('judge', '?')}"
        self.model = first.get("model", "?")

    def config_hash(self, request: Request) -> str:
        return self._config

    def judge(self, request: Request) -> Verdict:
        r = self.records.get(f"{self._config}:{request.input_hash()}")
        if r is None:
            raise KeyError(f"not in cassette {self.path.name}: {request.input_hash()[:12]}")
        answer = request.schema.model_validate(r["answer"]) if r["answer"] is not None else None
        return Verdict(
            answer,
            r["judge"],
            r["model"],
            r["input_tokens"],
            r["output_tokens"],
            r["cost_usd"],
            r["latency_s"],
            r["error"],
        )
