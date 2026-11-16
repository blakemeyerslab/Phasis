"""
Wall-clock time of every pipeline stage.

Each run writes a ``runtime_<month>_<day>_<hour>_<minute>`` file in the run
directory.  It has always held the index time; every stage now adds one
tab-separated line when it ends::

    Stage<TAB>name<TAB>start (ISO 8601)<TAB>end (ISO 8601)<TAB>seconds

so that the time of each stage can be read after the run, for example to
compare it with the mapping time.  Timing never changes what a stage does.
"""
from __future__ import annotations

import datetime
import os
import time
from contextlib import contextmanager

_LOG_PATH = None


def open_run_log(directory: str | None = None) -> str:
    """Create this run's runtime file and return its path."""
    global _LOG_PATH
    name = "runtime_%s" % datetime.datetime.now().strftime("%m_%d_%H_%M")
    path = os.path.join(directory, name) if directory else name
    with open(path, "a", encoding="utf-8"):
        pass
    _LOG_PATH = os.path.abspath(path)
    return path


def ensure_run_log(directory: str | None = None) -> str:
    """Return the current runtime file, creating one when the run has none yet."""
    if _LOG_PATH is None:
        open_run_log(directory)
    return _LOG_PATH


def current_run_log() -> str | None:
    return _LOG_PATH


def format_duration(seconds: float) -> str:
    seconds = int(max(0, round(float(seconds))))
    return "%d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def record(name: str, start_iso: str, end_iso: str, seconds: float) -> None:
    """Append one stage line to the runtime file; never raises."""
    path = _LOG_PATH
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("Stage\t%s\t%s\t%s\t%.2f\n" % (name, start_iso, end_iso, seconds))
    except OSError:
        pass


@contextmanager
def stage(name: str):
    """Time the enclosed stage, print its duration and record it in the runtime file."""
    start_iso = _now_iso()
    started = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - started
        record(name, start_iso, _now_iso(), seconds)
        print("[stage-time] %s: %s" % (name, format_duration(seconds)), flush=True)


def read_stage_times(path: str) -> list[dict]:
    """Parse the stage lines of a runtime file."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5 and parts[0] == "Stage":
                try:
                    rows.append({"stage": parts[1], "start": parts[2], "end": parts[3], "seconds": float(parts[4])})
                except ValueError:
                    continue
    return rows
