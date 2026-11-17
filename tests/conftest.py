"""Shared test setup."""
from __future__ import annotations

import pytest

from phasis import stage_timer


@pytest.fixture(autouse=True)
def _runtime_files_in_a_temporary_folder(tmp_path, monkeypatch):
    """Keep the runtime_* files of pipeline tests out of the source checkout.

    A pipeline run writes its runtime file in the current directory, which is
    the repository root when the tests run.  Each test starts without a run
    log, and a run log opened without a folder goes to the test's temporary
    folder.
    """
    original = stage_timer.open_run_log

    def open_in_tmp(directory=None):
        return original(directory if directory else str(tmp_path))

    monkeypatch.setattr(stage_timer, "_LOG_PATH", None)
    monkeypatch.setattr(stage_timer, "open_run_log", open_in_tmp)
