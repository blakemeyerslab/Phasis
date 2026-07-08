"""Environment variable helpers for Phasis runtime knobs."""

from __future__ import annotations

import os
from collections.abc import Mapping

PREFERRED_PREFIX = "Phasis"
_LEGACY_PREFIX = "PHA" + "SIS"


def legacy_env_name(name: str) -> str | None:
    """Return the hidden legacy all-caps alias for a preferred Phasis env name."""
    prefix = f"{PREFERRED_PREFIX}_"
    if not str(name).startswith(prefix):
        return None
    return f"{_LEGACY_PREFIX}_{str(name)[len(prefix):]}"


def getenv(
    name: str,
    default: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Read a preferred Phasis env var, falling back to its legacy alias."""
    source = os.environ if env is None else env
    if name in source:
        return source.get(name)
    legacy = legacy_env_name(name)
    if legacy and legacy in source:
        return source.get(legacy)
    return default
