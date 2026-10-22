"""Entry point of a sandboxed agent session: `python -m pfa.agent_entry <agent>`.

Reads its gateway URL, session token and task from the environment, runs the agent, and prints
the result as one JSON line. Exit code 3 means the gateway refused the session as stale.
"""

from __future__ import annotations

import json
import os
import sys

from pfa.gateway.client import GatewayClient, ToolFailed


def main() -> int:
    agent = sys.argv[1]
    if agent in ("invoice", "chase"):
        from pfa.backoffice.agent import AGENTS
    elif agent == "probe":
        AGENTS = {"probe": lambda client, payload: {"env": sorted(os.environ), "cwd": os.getcwd()}}
    else:
        print(f"unknown agent {agent}", file=sys.stderr)
        return 2
    client = GatewayClient(os.environ["PFA_GATEWAY_URL"], os.environ["PFA_SESSION_TOKEN"])
    try:
        result = AGENTS[agent](client, json.loads(os.environ["PFA_TASK"]))
    except ToolFailed as exc:
        print(f"{exc}", file=sys.stderr)
        return 3 if exc.error == "StaleSession" else 1
    print(json.dumps({**result, "session_cost_usd": round(client.cost_usd, 6)}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
