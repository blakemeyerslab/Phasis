# phasis/state.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from phasis.cache import resolve_artifact_path

# Process-local cache used by workers (read-only in worker hot paths).
# Keys are cluster IDs (cID), values are (phasis_score, combined_fishers).
WIN_SCORE_LOOKUP: dict[str, tuple[float, float]] = {}


def clear_win_score_lookup() -> None:
    """Clear the process-local WIN_SCORE_LOOKUP cache in-place."""
    WIN_SCORE_LOOKUP.clear()


def set_win_score_lookup(win_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Build and set a compact lookup: cID -> (phasis_score, combined_fishers).
    Stores it in the module-global WIN_SCORE_LOOKUP and returns it.

    Notes:
    - Mutates WIN_SCORE_LOOKUP in-place so references remain valid.
    - Skips rows that cannot be coerced to floats.
    """
    mapping: dict[str, tuple[float, float]] = {}

    if win_df is None or getattr(win_df, "empty", True):
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    df = win_df.copy()

    # Allow some mild schema flexibility (matches what you already do in legacy.py)
    if "cID" not in df.columns:
        for alt in ("clusterID", "cid", "cluster_id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "cID"})
                break

    if not {"cID", "phasis_score", "combined_fishers"}.issubset(set(df.columns)):
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    for row in df.itertuples(index=False):
        try:
            cid = str(getattr(row, "cID")).strip()
            ps = float(getattr(row, "phasis_score"))
            cf = float(getattr(row, "combined_fishers"))
            if cid:
                mapping[cid] = (ps, cf)
        except Exception:
            continue

    WIN_SCORE_LOOKUP.clear()
    WIN_SCORE_LOOKUP.update(mapping)
    return WIN_SCORE_LOOKUP


def load_win_score_lookup_from_tsv(path: str | Path) -> dict[str, tuple[float, float]]:
    """
    Load window-derived PHAS scores into a process-local lookup dict.

    Required on macOS (spawn) where worker processes do NOT inherit
    module globals from the parent process.

    Returns: dict[cID] -> (phasis_score, combined_fishers)

    Notes:
    - Mutates WIN_SCORE_LOOKUP in-place so references remain valid.
    - Returns empty dict if path missing/unreadable.
    """
    p = Path(path) if path else None
    if not p:
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    try:
        p = p.expanduser().resolve()
    except Exception:
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    physical = resolve_artifact_path(str(p))
    if not physical:
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    mapping: dict[str, tuple[float, float]] = {}
    try:
        df = pd.read_csv(physical, sep="\t", usecols=["cID", "phasis_score", "combined_fishers"])
    except Exception:
        WIN_SCORE_LOOKUP.clear()
        return WIN_SCORE_LOOKUP

    for row in df.itertuples(index=False):
        try:
            cid = str(row.cID).strip()
            ps = float(row.phasis_score)
            cf = float(row.combined_fishers)
            if cid:
                mapping[cid] = (ps, cf)
        except Exception:
            continue

    WIN_SCORE_LOOKUP.clear()
    WIN_SCORE_LOOKUP.update(mapping)
    return WIN_SCORE_LOOKUP
