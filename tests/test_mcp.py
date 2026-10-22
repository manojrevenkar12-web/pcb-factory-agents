"""The gateway's tools over MCP: an agent sees only its scoped tools, and actions stay exactly-once."""

from __future__ import annotations

import json
import sys

import anyio
import pytest
from pydantic import BaseModel

from pfa.gateway.core import Gateway
from pfa.gateway.http import app
from pfa.serving import Server, free_port

mcp = pytest.importorskip("mcp")


class Note(BaseModel):
    text: str


def test_tools_over_mcp(database_url):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    gw = Gateway(database_url, b"k" * 32)
    effects: list[str] = []

    @gw.tool("notes.add", "action", Note, "Adds a note")
    def add(ctx, a: Note) -> dict:
        effects.append(a.text)
        return {"n": len(effects)}

    @gw.tool("notes.secret", "read", Note, "Not for this session")
    def secret(ctx, a: Note) -> str:
        return "no"

    token = gw.mint(["notes.add"])
    with Server(app(gw), free_port()) as server:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pfa.gateway.mcp_server"],
            env={"PFA_GATEWAY_URL": server.url, "PFA_SESSION_TOKEN": token, "PATH": "/usr/bin:/bin"},
        )

        async def run() -> tuple[list[str], list[dict]]:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = [t.name for t in listed.tools]
                schema = next(t for t in listed.tools if t.name == "notes.add").input_schema
                assert "idempotency_key" in schema["required"]
                outs = []
                for _ in range(2):
                    r = await session.call_tool("notes.add", {"text": "hi", "idempotency_key": "note-1"})
                    outs.append(json.loads(r.content[0].text))
                denied = await session.call_tool("notes.secret", {"text": "x"})
                assert denied.is_error
                return names, outs

        names, outs = anyio.run(run)
    gw.close()
    assert names == ["notes.add"]
    assert effects == ["hi"] and [o["replayed"] for o in outs] == [False, True]
