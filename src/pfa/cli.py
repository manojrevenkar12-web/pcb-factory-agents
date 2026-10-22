"""pfa: run the evaluations that decide whether each agent may work alone.

pfa fetch         download the public datasets (pinned commits) into data/external
pfa backoffice    replay, fault injection, shadow run, crash test, race and frozen-worker tests
pfa inspect       board inspection on unseen boards, and the classical stage on DeepPCB
pfa parts         part programming on 9,800 real KiCad parts, with injected design errors
pfa schedule      CP-SAT replanning against the dispatch rule on simulated days
pfa gate          compare reports/*.json with evals/baselines; non-zero exit if anything regressed
pfa claude        the same replays with Claude as the judge (needs ANTHROPIC_API_KEY)
pfa all           everything above except claude, then the gate

Needs PostgreSQL for the back office: PFA_DATABASE_URL (default postgresql://postgres@127.0.0.1:5432/pfa).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pfa.config import settings

BASELINES = Path(__file__).resolve().parents[2] / "evals" / "baselines"


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"wrote {path}")


def backoffice(cfg, quick: bool = False) -> dict:
    from pfa import db
    from pfa.backoffice import rules
    from pfa.backoffice.evaluate import corrupt, crash_test, race, replay, shadow, zombie
    from pfa.judges import NoisyJudge

    db.migrate(cfg.database_url)
    summary, _ = replay(cfg.database_url, rules.judge())
    noisy = []
    for seed in (1, 2, 3):
        judge = NoisyJudge(rules.judge(), 0.2, corrupt, seed=seed)
        s, _ = replay(cfg.database_url, judge)
        noisy.append(
            {
                "seed": seed,
                "answers_corrupted": judge.corrupted,
                "wrong_actions": s.wrong_actions,
                "automation": s.automation,
            }
        )
    out = {
        "replay": summary.as_dict(),
        "fault_injection": noisy,
        "shadow": shadow(cfg.database_url, rules.judge()),
        "race": race(cfg.database_url),
        "frozen_worker": zombie(cfg.database_url),
        "crash_test": asdict(crash_test(cfg.database_url, emails=30 if quick else 60)),
    }
    _write(cfg.reports / "backoffice.json", out)
    return out


def inspect(cfg) -> dict:
    from pfa import figures, sources
    from pfa.inspect import deeppcb
    from pfa.inspect import evaluate as ev

    out = ev.run(cfg.reports)
    root = sources.fetch(cfg.data / "external", ["DeepPCB"])["DeepPCB"]
    out["deeppcb"] = deeppcb.run(root)
    _write(cfg.reports / "inspection.json", out)
    figures.capture_curve(out["capture_curve"], cfg.figures / "capture_curve.png")
    figures.board_example(
        json.loads((cfg.reports / "inspection_model.json").read_text()), cfg.figures / "board_example.jpg"
    )
    return out


def parts(cfg) -> dict:
    from pfa import sources
    from pfa.parts.evaluate import run, run_noisy

    paths = sources.fetch(cfg.data / "external", ["kicad-symbols", "kicad-footprints"])
    out = run(paths["kicad-symbols"], paths["kicad-footprints"])
    noisy = run_noisy(paths["kicad-symbols"], paths["kicad-footprints"])
    out["fault_injection"] = {
        k: noisy[k]
        for k in (
            "judge",
            "injected_caught",
            "injected_caught_rate",
            "wrong_designs_programmed",
            "automation",
        )
    }
    _write(cfg.reports / "parts.json", out)
    return out


def schedule(cfg, quick: bool = False) -> dict:
    from pfa.schedule.replan import run

    out = run(days=1 if quick else 3, orders=70, time_limit_s=1.0)
    _write(cfg.reports / "schedule.json", out)
    return out


# ---- the gate -------------------------------------------------------------------------------------------------


def gate(cfg) -> list[str]:
    """Every agent keeps its right to run alone only while these hold."""
    problems: list[str] = []

    def load(name: str) -> dict | None:
        p = cfg.reports / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def base(name: str) -> dict:
        p = BASELINES / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    bo, b = load("backoffice"), base("backoffice")
    if bo:
        if bo["replay"]["wrong_actions"]:
            problems.append(f"back office: {bo['replay']['wrong_actions']} wrong action(s) in the replay")
        if any(n["wrong_actions"] for n in bo["fault_injection"]):
            problems.append("back office: a corrupted judge answer became a wrong action")
        if bo["replay"]["automation"] < b.get("automation", 0) - 0.02:
            problems.append(f"back office: automation fell to {bo['replay']['automation']:.1%}")
        if not bo["crash_test"]["passed"]:
            problems.append("back office: crash test failed (an action ran twice or not at all)")
        if bo["race"]["booked"] != 1:
            problems.append("back office: the race booked both invoices against one delivery")
        if bo["frozen_worker"]["frozen_session_exit"] != 3 or not bo["frozen_worker"]["booked_once"]:
            problems.append("back office: a frozen worker acted after losing its lease")
        if bo["shadow"]["agent_effects_on_erp"]:
            problems.append("back office: a shadow session changed the ERP")
    ins, b = load("inspection"), base("inspection")
    if ins:
        r = ins["judge_top_20pct"]
        if r["recall"] < b.get("recall", 0) - 0.02:
            problems.append(f"inspection: recall fell to {r['recall']:.1%}")
        if r["false_calls_per_board"] > b.get("false_calls_per_board", 1e9) + 0.1:
            problems.append(f"inspection: {r['false_calls_per_board']:.2f} false calls per board")
        if ins["capture_curve"]["20%"] < b.get("capture_20", 0) - 0.02:
            problems.append("inspection: the ranking lets more defects past the worst 20%")
    pa, b = load("parts"), base("parts")
    if pa:
        if pa["wrong_designs_programmed"] > b.get("wrong_designs_programmed", 0):
            problems.append(f"parts: {pa['wrong_designs_programmed']} wrong design(s) programmed")
        if pa["automation"] < b.get("automation", 0) - 0.02:
            problems.append(f"parts: automation fell to {pa['automation']:.1%}")
    return problems


def claude(cfg) -> dict:
    """Replays with Claude as the judge. Records every answer to a cassette for free re-runs."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    from pfa.backoffice.evaluate import replay
    from pfa.judges import ClaudeJudge, RecordingJudge

    model = os.environ.get("PFA_CLAUDE_MODEL", "claude-sonnet-5-5")
    judge = RecordingJudge(
        ClaudeJudge(model, effort=os.environ.get("PFA_CLAUDE_EFFORT", "low")),
        cfg.data / "cassettes" / f"book-invoice-{model}.jsonl",
    )
    summary, _ = replay(cfg.database_url, judge)
    out = {"book-invoice": summary.as_dict()}
    _write(cfg.reports / f"claude-{model}.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pfa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["fetch", "backoffice", "inspect", "parts", "schedule", "gate", "claude", "all"]
    )
    parser.add_argument("--quick", action="store_true", help="smaller runs (CI on every push)")
    args = parser.parse_args(argv)
    cfg = settings()
    steps = (
        ["backoffice", "inspect", "parts", "schedule", "gate"] if args.command == "all" else [args.command]
    )
    for step in steps:
        started = time.time()
        if step == "fetch":
            from pfa import sources

            print(sources.fetch(cfg.data / "external"))
        elif step == "backoffice":
            backoffice(cfg, args.quick)
        elif step == "inspect":
            inspect(cfg)
        elif step == "parts":
            parts(cfg)
        elif step == "schedule":
            schedule(cfg, args.quick)
        elif step == "claude":
            claude(cfg)
        elif step == "gate":
            problems = gate(cfg)
            for p in problems:
                print(f"GATE: {p}")
            if problems:
                sys.exit(1)
            print("gate: every agent keeps its right to run alone")
        print(f"[{step}] {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
