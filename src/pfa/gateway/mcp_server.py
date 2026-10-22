"""The gateway's tools over MCP, for agents that speak it (Claude Code, the Agent SDK).

Runs inside the sandbox as a stdio server. It holds only the session token: every tool call is
forwarded to the gateway, which checks scopes and fences, runs actions exactly once and keeps
the credentials. `python -m pfa.gateway.mcp_server` with PFA_GATEWAY_URL and PFA_SESSION_TOKEN.
"""

from __future__ import annotations

import json
import os

import anyio
import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from pfa.gateway.client import GatewayClient, ToolFailed

KEY_FIELD = "idempotency_key"


def build(gateway_url: str, token: str, transport: httpx.BaseTransport | None = None) -> Server:
    client = GatewayClient(gateway_url, token, transport=transport)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        listed = client.http.get("/v1/tools").json()
        tools = []
        for t in listed:
            schema = dict(t["input_schema"])
            if t["effect"] == "action":  # actions take a key the agent derives from the business object
                schema.setdefault("properties", {})[KEY_FIELD] = {
                    "type": "string",
                    "description": "Stable key for this business action, e.g. supplier:invoice_number. "
                    "Re-using it returns the earlier result instead of acting twice.",
                }
                schema["required"] = [*schema.get("required", []), KEY_FIELD]
            tools.append(types.Tool(name=t["name"], description=t["description"], inputSchema=schema))
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params) -> types.CallToolResult:
        args = dict(params.arguments or {})
        key = args.pop(KEY_FIELD, None)
        try:
            if key is not None:
                out = client.act(params.name, key, **args)
            else:
                out = {"result": client.read(params.name, **args), "replayed": False}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(out, default=str))]
            )
        except ToolFailed as exc:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(exc))], isError=True)

    return Server("pfa-gateway", version="1.0.0", on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    server = build(os.environ["PFA_GATEWAY_URL"], os.environ["PFA_SESSION_TOKEN"])

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
