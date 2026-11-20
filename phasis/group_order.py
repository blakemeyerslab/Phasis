"""
Order of ``DataFrame.groupby(keys, sort=False).indices`` for several keys,
made independent of the pandas version.

Up to pandas 3.0, the ``indices`` dict of a multi-key ``groupby(...,
sort=False)`` listed its groups by the first-appearance code of each key in
turn: every group of the first key value seen came first, ordered by the first
appearance of its second key value, and so on.  pandas 3.1 lists them by the
first appearance of the whole key combination instead (the order in which the
groupby is iterated, which did not change).  The in-memory feature assembly
planned its batches, and so its output rows, in the pre-3.1 ``indices`` order;
this module spells that order out so it no longer depends on pandas.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def legacy_group_indices(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    dropna: bool = True,
) -> dict[tuple, np.ndarray]:
    """Return ``{key tuple: row positions}`` in the pandas <= 3.0 group order.

    Equivalent to ``frame.groupby(list(keys), sort=False, observed=True,
    dropna=dropna).indices`` under pandas 3.0, with the dict in that version's
    iteration order.  Row positions are ascending, as in pandas.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("legacy_group_indices needs at least one key column")
    if frame.empty:
        return {}

    codes = []
    for key in keys:
        column_codes, _ = pd.factorize(frame[key], use_na_sentinel=dropna)
        codes.append(np.asarray(column_codes, dtype=np.int64))

    positions = np.arange(len(frame), dtype=np.int64)
    if dropna:
        keep = np.ones(len(frame), dtype=bool)
        for column_codes in codes:
            keep &= column_codes >= 0
        if not keep.all():
            positions = positions[keep]
            codes = [column_codes[keep] for column_codes in codes]
        if positions.size == 0:
            return {}

    # np.lexsort is stable and sorts by its last key first, so this orders the
    # rows by the first key's code, then the second, ..., keeping row order.
    order = np.lexsort(tuple(reversed(codes)))
    sorted_codes = np.stack([column_codes[order] for column_codes in codes])
    sorted_positions = positions[order]
    changes = np.flatnonzero((np.diff(sorted_codes, axis=1) != 0).any(axis=0)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [sorted_positions.size]))

    columns = [frame[key] for key in keys]
    groups: dict[tuple, np.ndarray] = {}
    for start, end in zip(starts.tolist(), ends.tolist()):
        group_positions = sorted_positions[start:end]
        first = int(group_positions[0])
        key = tuple(_group_key_value(column.iat[first]) for column in columns)
        groups[key] = group_positions
    return groups


def _group_key_value(value):
    """Missing key values become ``nan``, as in pandas group keys."""
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    return value
