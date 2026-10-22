from __future__ import annotations

import random
from types import SimpleNamespace

from pfa.backoffice.schemas import InvoiceAnswer
from pfa.judges import (
    ClaudeJudge,
    FunctionJudge,
    NoisyJudge,
    RecordingJudge,
    ReplayJudge,
    Request,
    cost_usd,
    image_tokens,
    strict_schema,
)

ANSWER = {
    "document_type": "invoice",
    "invoice_number": "BW-26-00440",
    "po_number": "4500107",
    "currency": "EUR",
    "lines": [{"mpn": "LM358DR", "quantity": 100, "unit_price": 0.112}],
    "shipping": 0,
    "tax": 2.13,
    "total": 13.33,
}


def test_strict_schema_is_what_structured_outputs_accept():
    schema = strict_schema(InvoiceAnswer)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for bad in ("minimum", "maximum", "minLength", "maxLength", "title"):
                assert bad not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


def test_image_tokens_follow_the_published_formula():
    assert image_tokens(28, 28) == 1
    assert image_tokens(56, 29) == 4
    assert image_tokens(10000, 10000) <= 4784  # downscaled to the high-resolution limit


def test_cost_uses_list_prices():
    assert abs(cost_usd("claude-sonnet-5-5", 1_000_000, 0) - 2.0) < 1e-9
    assert cost_usd("unknown-model", 1000, 1000) == 0.0


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(kw)
        stop, text = self.replies.pop(0)
        return SimpleNamespace(
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
            content=[SimpleNamespace(type="text", text=text)],
        )


def test_claude_judge_sends_skill_schema_and_effort_and_retries_invalid_answers():
    import json

    fake = FakeMessages([("end_turn", '{"document_type": "invoice"}'), ("end_turn", json.dumps(ANSWER))])
    judge = ClaudeJudge("claude-sonnet-5-5", effort="low", client=SimpleNamespace(messages=fake))
    verdict = judge.judge(Request("book-invoice", "an email", InvoiceAnswer))
    assert verdict.ok and verdict.answer.invoice_number == "BW-26-00440"
    first = fake.calls[0]
    assert first["output_config"]["effort"] == "low"
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert "accounts payable" in first["system"]
    assert "invalid" in fake.calls[1]["messages"][0]["content"][-1]["text"]  # the retry says what was wrong
    assert (
        verdict.input_tokens == 2000
        and abs(verdict.cost_usd - cost_usd("claude-sonnet-5-5", 2000, 200)) < 1e-12
    )


def test_claude_judge_reports_refusal_instead_of_guessing():
    fake = FakeMessages([("refusal", ""), ("refusal", "")])
    verdict = ClaudeJudge(client=SimpleNamespace(messages=fake)).judge(
        Request("book-invoice", "x", InvoiceAnswer)
    )
    assert not verdict.ok and "refusal" in verdict.error


def test_config_hash_changes_with_model_effort_and_skill():
    req = Request("book-invoice", "x", InvoiceAnswer)
    a = ClaudeJudge("claude-sonnet-5-5", "low").config_hash(req)
    assert a != ClaudeJudge("claude-haiku-5-5", "low").config_hash(req)
    assert a != ClaudeJudge("claude-sonnet-5-5", "high").config_hash(req)
    assert a != ClaudeJudge("claude-sonnet-5-5", "low").config_hash(
        Request("inspect-part", "x", InvoiceAnswer)
    )


def test_record_then_replay_gives_identical_verdicts(tmp_path):
    inner = FunctionJudge("fixed", lambda r: InvoiceAnswer.model_validate(ANSWER))
    rec = RecordingJudge(inner, tmp_path / "c.jsonl")
    req = Request("book-invoice", "an email", InvoiceAnswer)
    first = rec.judge(req)
    replay = ReplayJudge(tmp_path / "c.jsonl", rec.config_hash(req))
    assert replay.judge(req).answer == first.answer
    try:
        replay.judge(Request("book-invoice", "another email", InvoiceAnswer))
        raise AssertionError("an unrecorded request must not be answered")
    except KeyError:
        pass


def test_noisy_judge_corrupts_about_the_requested_share():
    inner = FunctionJudge("fixed", lambda r: InvoiceAnswer.model_validate(ANSWER))

    def corrupt(a, rng: random.Random):
        a.total = -1
        return a

    noisy = NoisyJudge(inner, 0.2, corrupt, seed=1)
    n = 2000
    bad = sum(
        noisy.judge(Request("book-invoice", str(i), InvoiceAnswer)).answer.total == -1 for i in range(n)
    )
    assert 0.17 < bad / n < 0.23
