"""Make the test suite hermetic w.r.t. the developer's local environment.

A local .env for running the app (a persistent ./vettalume.db, SERVE_ONLY_APPROVED=false for serving
draft items to mocks) must never leak into the tests. So BEFORE the app builds its engine we force a
throwaway in-memory database and the default toggles. pydantic-settings gives real environment
variables precedence over the .env file, and this module is imported before any test imports the app.
"""
import os

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["SERVE_ONLY_APPROVED"] = "true"

import pytest  # noqa: E402

from app.config import settings  # noqa: E402

_DEFAULT_TOGGLES = {"serve_only_approved": True}


@pytest.fixture(autouse=True)
def _pin_default_toggles(monkeypatch):
    for name, value in _DEFAULT_TOGGLES.items():
        if hasattr(settings, name):
            monkeypatch.setattr(settings, name, value, raising=False)
    yield
