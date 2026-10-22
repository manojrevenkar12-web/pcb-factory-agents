from __future__ import annotations

import os

import pytest

DB = os.environ.get("PFA_TEST_DATABASE_URL")


@pytest.fixture
def database_url() -> str:
    if not DB:
        pytest.skip("PFA_TEST_DATABASE_URL not set")
    from pfa import db

    db.reset(DB)
    return DB
