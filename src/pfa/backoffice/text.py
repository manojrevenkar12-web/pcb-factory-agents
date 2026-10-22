"""Email text handling shared by the rule reader and the checks."""

from __future__ import annotations

import html
import re

_QUOTE_START = re.compile(r"^On .+ wrote:\s*$", re.M)
_TAG = re.compile(r"<[^>]+>")


def clean(body: str) -> str:
    """Drops quoted history and turns HTML tables into one row per line."""
    m = _QUOTE_START.search(body)
    if m:
        body = body[: m.start()]
    body = "\n".join(ln for ln in body.splitlines() if not ln.startswith(">"))
    if "<html" in body.lower():
        body = re.sub(r"</t[dh]>", " | ", body, flags=re.I)
        body = re.sub(r"</tr>|<br\s*/?>|</p>", "\n", body, flags=re.I)
        body = html.unescape(_TAG.sub("", body))
    return body
