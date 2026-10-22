"""Public datasets, fetched at pinned commits (never redistributed with the code)."""

from __future__ import annotations

import subprocess
from pathlib import Path

SOURCES = {
    # Real PCB template/test image pairs with boxed defects. MIT licence; research use.
    "DeepPCB": ("https://github.com/tangsanli5201/DeepPCB", "08e98c4db5922613fb97176eb3d6497d48260cb1"),
    # KiCad 5 libraries, CC-BY-SA 4.0 with a design exception.
    "kicad-symbols": ("https://github.com/KiCad/kicad-symbols", "4e17d7595f9691d45d97aea8ad64572652fb5bed"),
    "kicad-footprints": (
        "https://github.com/KiCad/kicad-footprints",
        "7ebfa6b23cc292a56f751b7b5f4a0e12eeef69dd",
    ),
}


def fetch(root: Path, names: list[str] | None = None) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names or list(SOURCES):
        url, commit = SOURCES[name]
        target = root / name
        head = target / ".git"
        if head.exists():
            current = subprocess.run(
                ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            if current == commit:
                out[name] = target
                continue
        target.mkdir(parents=True, exist_ok=True)
        if not head.exists():
            _git(target, "init", "-q")
            _git(target, "remote", "add", "origin", url)
        _git(target, "fetch", "-q", "--depth", "1", "origin", commit)
        _git(target, "checkout", "-q", "FETCH_HEAD")
        out[name] = target
    return out


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
