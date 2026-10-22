"""Where things live. Everything is configured by environment variables with local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE = Path(__file__).parent
SKILLS = PACKAGE / "skills"


@dataclass(frozen=True)
class Settings:
    home: Path
    database_url: str

    @property
    def data(self) -> Path:
        return self.home / "data"

    @property
    def reports(self) -> Path:
        return self.home / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> Settings:
        for p in (self.data, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def settings() -> Settings:
    return Settings(
        home=Path(os.environ.get("PFA_HOME", ".")).resolve(),
        database_url=os.environ.get("PFA_DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/pfa"),
    ).ensure()
