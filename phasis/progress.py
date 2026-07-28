"""Progress-bar helpers for Phasis command-line tools.

``tqdm`` writes to stderr by default.  Batch schedulers commonly capture stderr
as an error log, so ordinary progress updates then look like errors.  Keep the
normal Phasis progress stream on stdout instead.
"""

from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm as _tqdm


def tqdm(*args: Any, **kwargs: Any):
    """Create a tqdm bar that writes to stdout unless a caller overrides it."""
    if kwargs.get("file") is None:
        kwargs["file"] = sys.stdout
    return _tqdm(*args, **kwargs)
