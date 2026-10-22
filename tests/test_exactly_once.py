"""End to end on PostgreSQL: sandboxed sessions, crashes, races and frozen workers."""

from __future__ import annotations

import json

from pfa.backoffice import rules
from pfa.backoffice.env import Environment
from pfa.backoffice.evaluate import crash_test, race, replay, zombie
from pfa.sandbox import collect, start_session


def test_replay_matches_the_clerk_with_no_wrong_actions(database_url):
    summary, _ = replay(database_url, rules.judge(), emails=80)
    assert summary.wrong_actions == 0 and summary.automation > 0.8


def test_sandbox_holds_no_credentials(database_url):
    with Environment(database_url, rules.judge()) as env:
        proc, home = start_session("probe", env.gateway_server.url, "token", {})
        result = collect(proc, 30)
    env_names = set(result.output["env"])
    assert not env_names & {"PFA_DATABASE_URL", "ANTHROPIC_API_KEY", "PGPASSWORD", "AWS_SECRET_ACCESS_KEY"}
    assert env_names <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "HOME",
        "TMPDIR",
        "PFA_GATEWAY_URL",
        "PFA_SESSION_TOKEN",
        "PFA_TASK",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "LC_CTYPE",
    }
    assert result.output["cwd"] == home


def test_two_invoices_racing_for_one_delivery_book_once(database_url):
    out = race(database_url)
    assert out["both_passed_the_agent_check"] and out["booked"] == 1 and out["sent_to_review"] == 1


def test_a_frozen_worker_cannot_act_after_losing_its_lease(database_url):
    out = zombie(database_url)
    assert out["frozen_session_exit"] == 3 and out["booked_once"] and out["rescuer_completed"]


def test_crashes_everywhere_still_book_and_email_exactly_once(database_url):
    report = crash_test(database_url, emails=24, workers=3)
    assert report.passed, json.dumps(report.__dict__, default=str)
    assert report.sessions_killed + report.crashpoint_exits > 0  # the test actually crashed things
