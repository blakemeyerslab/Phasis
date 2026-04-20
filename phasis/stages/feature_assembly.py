from __future__ import annotations


from .. import state as st
from .. import ids
import os
import re
import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, MemCache, phase2_basename, stage_signature
from phasis.parallel import run_parallel_with_progress

DCL_OVERHANG = 3          # 2-nt 3' overhang in duplex -> 3-nt genomic offset
WINDOW_MULTIPLIER = 10    # 10 cycles per window
FEATURE_SCHEMA_VERSION = 3
HOWELL_AMBIGUITY_FRACTION = 0.90


# ---- legacy schema (KNN-compatible) ---------------------------------------
FEATURE_COLS = ['identifier',
 'cID',
 'alib',
 'complexity',
 'strand_bias',
 'log_clust_len_norm_counts',
 'ratio_abund_len_phase',
 'phasis_score',
 'combined_fishers',
 'total_abund',
 'w_Howell_score',
 'w_window_start',
 'w_window_end',
 'c_Howell_score',
 'c_window_start',
 'c_window_end',
 'Peak_Howell_score',
 'Howell_exact_support_score',
 'Howell_ambiguity_count',
 'Howell_alt_register_count',
 'Howell_overlap_margin',
 'w_Howell_score_strict',
 'w_window_start_strict',
 'w_window_end_strict',
 'c_Howell_score_strict',
 'c_window_start_strict',
 'c_window_end_strict',
 'Peak_Howell_score_strict']

# Numeric columns (all except id-like fields)
NUMERIC_COLS = set(FEATURE_COLS) - {"identifier", "cID", "alib"}


def _phase_value(default: int = 21) -> int:
    """Return rt.phase as an int (module-level, spawn-safe)."""
    try:
        v = getattr(rt, "phase", None)
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _get_memfile() -> str:
    """Return runtime memFile; create a reasonable default if missing."""
    mem = getattr(rt, "memFile", None)
    if mem:
        return str(mem)

    outdir = getattr(rt, "outdir", None)
    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(str(outdir)))
        os.makedirs(outdir_abs, exist_ok=True)
        mem = os.path.join(outdir_abs, MEM_FILE_DEFAULT)
    else:
        mem = MEM_FILE_DEFAULT

    rt.memFile = mem
    return mem


def ensure_win_score_lookup_ready() -> None:
    """
    Spawn-safe: ensure st.WIN_SCORE_LOOKUP is populated in *this* process.
    If empty and rt.clusters_scored_tsv exists, load it.
    """
    try:
        if st.WIN_SCORE_LOOKUP:
            return
        p = getattr(rt, "clusters_scored_tsv", None)
        if p and os.path.isfile(p):
            st.load_win_score_lookup_from_tsv(p)
    except Exception:
        # keep feature assembly robust; caller will fall back to defaults
        return

def features_to_detection(clusters_data: pd.DataFrame,*,phase: str | int | None = None,outdir: str | None = None,concat_libs: bool | None = None,memFile: str | None = None,outfname: str | None = None,) -> pd.DataFrame:
    """
    Assemble per-cluster feature set (parallel), write TSV, and memoize via md5.
    Uses legacy column names compatible with downstream KNN.
    Expects process_chromosome_features() to return rows in FEATURE_COLS order.
    """
    print("### Step: assemble per-cluster features ###")

    # Resolve defaults from runtime unless explicitly provided
    if phase is None:
        phase = getattr(rt, "phase", None)
    if outdir is None:
        outdir = getattr(rt, "outdir", None)
    if concat_libs is None:
        try:
            concat_libs = bool(getattr(rt, "concat_libs", False))
        except Exception:
            concat_libs = False

    if memFile is None:
        memFile = _get_memfile()
    else:
        # Keep runtime consistent so other cache helpers still behave
        rt.memFile = memFile
        if outdir is not None:
            rt.outdir = outdir

    if outfname is None:
        prefix = "concat_" if concat_libs else ""
        outfname = f"{prefix}{phase}_cluster_set_features.tsv"

    # Input signature: only reuse cached features when upstream inputs match.
    phas_path = phase2_basename("PHAS_to_detect.tab")
    scored_path = phase2_basename("clusters_scored.tsv")

    input_sig = stage_signature(
        files=[phas_path, scored_path],
        params={
            "phase": phase,
            "concat_libs": bool(concat_libs),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    )

    cache = MemCache.load(memFile)
    section = "CLUSTER_FEATURES"

    # ---------- Early cache check ----------
    if cache.hit(section, outfname, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping assembly: {outfname}")
        df = pd.read_csv(outfname, sep="	")

        # Coerce numerics by legacy names only
        for col in df.columns:
            if col in NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure exact column order if all present
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if not missing:
            df = df[FEATURE_COLS]
        else:
            print(f"[WARN] Existing file lacks expected columns: {missing}")
        return df

    # ---------- Validate input ----------
    required_cols = ['clusterID', 'chromosome', 'strand', 'pos', 'len', 'abun', 'identifier', 'tag_seq', 'alib']
    missing_in_input = [c for c in required_cols if c not in clusters_data.columns]
    if missing_in_input:
        raise ValueError(f"clusters_data missing required columns: {missing_in_input}")

    clusters_data = clusters_data[required_cols].copy()

    # Split per chromosome for parallel processing
    chromosome_groups = [df for _, df in clusters_data.groupby('chromosome', sort=False)]
    print(f"  - Found {len(chromosome_groups)} chromosome groups")

    # ---------- Parallel processing ----------
    results = run_parallel_with_progress(
        process_chromosome_features,
        chromosome_groups,
        desc="Assemble features",
        min_chunk=1,
        adaptive_recovery=True,
        unit="lib-chr"
    )

    # ---------- Flatten safely ----------
    flat = []
    bad_chunks = 0
    for sub in results or []:
        if isinstance(sub, list):
            for row in sub:
                if isinstance(row, (list, tuple)) and len(row) == len(FEATURE_COLS):
                    flat.append(list(row))
                else:
                    bad_chunks += 1
        else:
            bad_chunks += 1

    if bad_chunks:
        print(f"[WARN] Skipped {bad_chunks} malformed/failed rows or chunks during feature assembly.")

    if not flat:
        raise RuntimeError("No features assembled; all chunks failed or returned empty results.")

    # ---------- DataFrame materialization ----------
    collected_features = pd.DataFrame(flat, columns=FEATURE_COLS)

    # Numeric coercion on the newly built DF (legacy names)
    for col in collected_features.columns:
        if col in NUMERIC_COLS:
            collected_features[col] = pd.to_numeric(collected_features[col], errors="coerce")
    # ---------- Write + cache record ----------
    collected_features.to_csv(outfname, sep="	", index=False)
    fp = cache.record(section, outfname, input_sig)
    if fp:
        print(f"  - Wrote {outfname} (md5: {fp})")

    return collected_features


def _strand_masks(df: pd.DataFrame):
    """Return boolean masks for W and C strands accepting several encodings."""
    s = df['strand'].astype(str).str.lower()
    w_mask = s.isin(['w', '+', 'watson', '1', 'true'])
    c_mask = s.isin(['c', '-', 'crick', '0', 'false'])
    return w_mask, c_mask

def _build_pos_abun_exact_phase(df: pd.DataFrame, seq_start: int, seq_end: int, phase: int):
    """
    Build {position -> abundance} using ONLY reads with length == phase
    and positions within [seq_start, seq_end].
    """
    ph = int(phase)
    d: dict[int, float] = {}
    # small speed-up: filter by length first
    df_ph = df.loc[pd.to_numeric(df['len'], errors='coerce') == ph]
    if df_ph.empty:
        return d
    for _, row in df_ph.iterrows():
        pos = int(row['pos'])
        if seq_start <= pos <= seq_end:
            d[pos] = d.get(pos, 0.0) + float(row['abun'])
    return d


def _compute_howell_score_from_register(
    in_phase_sum: float,
    effective_total: float,
    n_filled: int,
    num_cycles: int,
):
    out_of_phase = max(0.0, float(effective_total) - float(in_phase_sum))
    numerator = float(in_phase_sum)
    denominator = 1.0 + out_of_phase
    if numerator <= 0.0 or not (int(n_filled) > 3):
        return 0.0, out_of_phase

    log_arg = 1.0 + 10.0 * (numerator / denominator)
    if log_arg <= 0.0 or log_arg != log_arg:
        return 0.0, out_of_phase

    scale = max(min(int(n_filled), int(num_cycles)) - 2, 0)
    return float(scale * (0.0 if log_arg <= 0 else np.log(log_arg))), float(out_of_phase)


def _score_window_registers(window_positions, pos_abun, win_start, win_end, phase, *, forward=True, exact_only: bool = False):
    ph = int(phase)
    if not window_positions:
        return {
            "score": 0.0,
            "best_register": None,
            "best_in_phase_sum": 0.0,
            "best_out_of_phase": 0.0,
            "best_filled": 0,
            "register_scores": [0.0] * ph,
        }

    num_cycles = max(0, (win_end - win_start + 1) // ph)
    if num_cycles < 4:
        return {
            "score": 0.0,
            "best_register": None,
            "best_in_phase_sum": 0.0,
            "best_out_of_phase": 0.0,
            "best_filled": 0,
            "register_scores": [0.0] * ph,
        }

    best_score = -float("inf")
    best_reg_sum = 0.0
    best_reg_out = 0.0
    best_reg_filled = 0
    best_reg = None
    register_scores: list[float] = []
    evaluator = _evaluate_register_strict_exact if exact_only else _evaluate_register

    for reg in range(ph):
        in_sum, eff_total, n_filled = evaluator(
            window_positions, pos_abun, win_start, win_end, ph, reg, forward=forward
        )
        reg_score, out_of_phase = _compute_howell_score_from_register(
            in_sum,
            eff_total,
            n_filled,
            num_cycles,
        )
        register_scores.append(float(reg_score))
        if reg_score > best_score:
            best_score = reg_score
            best_reg_sum = float(in_sum)
            best_reg_out = float(out_of_phase)
            best_reg_filled = int(n_filled)
            best_reg = reg

    return {
        "score": float(0.0 if best_score == -float("inf") else best_score),
        "best_register": best_reg,
        "best_in_phase_sum": float(best_reg_sum),
        "best_out_of_phase": float(best_reg_out),
        "best_filled": int(best_reg_filled),
        "register_scores": register_scores,
    }


def _windows_overlap(start_a, end_a, start_b, end_b) -> bool:
    return int(max(start_a, start_b)) <= int(min(end_a, end_b))


def _summarize_peak_howell_ambiguity(
    best_window_detail: dict | None,
    candidate_windows,
    *,
    threshold_fraction: float = HOWELL_AMBIGUITY_FRACTION,
):
    if not best_window_detail:
        return None

    exact_support_score = float(best_window_detail.get("exact_score", 0.0) or 0.0)
    winner_register = best_window_detail.get("exact_best_register")
    register_scores = best_window_detail.get("exact_register_scores") or []
    result = {
        "winner_strand": best_window_detail.get("strand"),
        "winner_window_start": best_window_detail.get("window_start"),
        "winner_window_end": best_window_detail.get("window_end"),
        "winner_register": best_window_detail.get("best_register"),
        "exact_winner_register": winner_register,
        "Howell_exact_support_score": float(exact_support_score),
        "best_overlapping_competitor_score": np.nan,
        "Howell_ambiguity_count": np.nan,
        "Howell_alt_register_count": np.nan,
        "Howell_overlap_margin": np.nan,
    }

    if exact_support_score <= 0.0:
        return result

    threshold_score = float(threshold_fraction) * exact_support_score
    overlap_best_score = None
    overlap_count = 0
    for candidate in candidate_windows or []:
        if str(candidate.get("strand")) != str(best_window_detail.get("strand")):
            continue
        if (
            int(candidate.get("window_start")) == int(best_window_detail.get("window_start"))
            and int(candidate.get("window_end")) == int(best_window_detail.get("window_end"))
        ):
            continue
        if not _windows_overlap(
            candidate.get("window_start"),
            candidate.get("window_end"),
            best_window_detail.get("window_start"),
            best_window_detail.get("window_end"),
        ):
            continue
        candidate_score = float(candidate.get("exact_score", 0.0) or 0.0)
        if candidate_score < threshold_score:
            continue
        overlap_count += 1
        if overlap_best_score is None or candidate_score > overlap_best_score:
            overlap_best_score = candidate_score

    alt_register_count = 0
    if winner_register is not None:
        for reg_idx, reg_score in enumerate(register_scores):
            if int(reg_idx) == int(winner_register):
                continue
            if float(reg_score) >= threshold_score:
                alt_register_count += 1

    result["best_overlapping_competitor_score"] = (
        np.nan if overlap_best_score is None else float(overlap_best_score)
    )
    result["Howell_ambiguity_count"] = int(overlap_count)
    result["Howell_alt_register_count"] = int(alt_register_count)
    result["Howell_overlap_margin"] = (
        np.nan if overlap_best_score is None else float(exact_support_score - float(overlap_best_score))
    )
    return result

# ---------- RELAXED Howell (positional ±1 wobble allowed) ----------
def _best_sliding_window_score_generic(
    pos_abun,
    phase,
    win_size,
    seq_start=None,
    seq_end=None,
    forward=True,
    *,
    return_detail: bool = False,
):
    """
    Generic window scan (relaxed Howell with ±1 wobble).
    Expects pos_abun already filtered to len == phase.
    """
    positions = sorted(pos_abun.keys())
    if not positions:
        if return_detail:
            return 0.0, None, None, None
        return 0.0, None, None

    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound

    best_score = -float("inf")
    best_window = (None, None)
    best_detail = None
    candidate_windows = [] if return_detail else None

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        relaxed_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            int(phase),
            forward=forward,
        )
        score = float(relaxed_summary["score"])
        exact_summary = None
        if return_detail:
            exact_summary = _score_window_registers(
                window_positions,
                pos_abun,
                win_start,
                win_end,
                int(phase),
                forward=forward,
                exact_only=True,
            )

        if return_detail:
            candidate_windows.append(
                {
                    "strand": "w" if forward else "c",
                    "window_start": int(win_start),
                    "window_end": int(win_end),
                    "score": float(score),
                    "best_register": relaxed_summary.get("best_register"),
                    "exact_score": float(exact_summary["score"]),
                    "exact_best_register": exact_summary.get("best_register"),
                }
            )

        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)
            if return_detail:
                best_detail = {
                    "strand": "w" if forward else "c",
                    "window_start": int(win_start),
                    "window_end": int(win_end),
                    "score": float(score),
                    "best_register": relaxed_summary.get("best_register"),
                    "register_scores": list(relaxed_summary.get("register_scores") or []),
                    "exact_score": float(exact_summary["score"]),
                    "exact_best_register": exact_summary.get("best_register"),
                    "exact_register_scores": list(exact_summary.get("register_scores") or []),
                }

    resolved_score = best_score if best_score != -float("inf") else 0.0
    if not return_detail:
        return resolved_score, best_window[0], best_window[1]

    detail = _summarize_peak_howell_ambiguity(best_detail, candidate_windows)
    return resolved_score, best_window[0], best_window[1], detail

def best_sliding_window_score_forward(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )

def best_sliding_window_score_reverse(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

def _pick_peak_howell_detail(w_detail: dict | None, c_detail: dict | None):
    if w_detail is None:
        return c_detail
    if c_detail is None:
        return w_detail

    w_score = float(w_detail.get("score", 0.0) or 0.0)
    c_score = float(c_detail.get("score", 0.0) or 0.0)
    if c_score > w_score:
        return c_detail
    return w_detail


def compute_phasing_score_Howell(aclust: pd.DataFrame, *, return_detail: bool = False):
    """
    Howell-like phasing WITH positional wobble (±1), but ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    When return_detail=True, appends the peak-window ambiguity summary.
    """
    ph = _phase_value()
    win_size  = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        if w_pos_abun:
            if return_detail:
                w_score, w_s, w_e, w_detail = _best_sliding_window_score_generic(
                    w_pos_abun,
                    int(ph),
                    win_size,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    forward=True,
                    return_detail=True,
                )
            else:
                w_score, w_s, w_e = best_sliding_window_score_forward(
                    w_pos_abun, int(ph), win_size, seq_start, seq_end
                )
                w_detail = None
        else:
            w_score, w_s, w_e, w_detail = 0.0, None, None, None
    else:
        w_score, w_s, w_e, w_detail = None, None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        if c_pos_abun:
            if return_detail:
                c_score, c_s, c_e, c_detail = _best_sliding_window_score_generic(
                    c_pos_abun,
                    int(ph),
                    win_size,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    forward=False,
                    return_detail=True,
                )
            else:
                c_score, c_s, c_e = best_sliding_window_score_reverse(
                    c_pos_abun, int(ph), win_size, seq_start, seq_end
                )
                c_detail = None
        else:
            c_score, c_s, c_e, c_detail = 0.0, None, None, None
    else:
        c_score, c_s, c_e, c_detail = None, None, None, None

    if not return_detail:
        return (w_score, (w_s, w_e), c_score, (c_s, c_e))

    peak_detail = _pick_peak_howell_detail(w_detail, c_detail)
    return (w_score, (w_s, w_e), c_score, (c_s, c_e), peak_detail)

def _evaluate_register_strict_exact(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """Count ONLY exact register hits (no ±1). Returns: (in_phase_sum, total_in_window, n_filled_cycles)"""
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // int(phase))

    in_phase_sum = 0.0
    n_filled = 0
    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * int(phase)) if forward else (win_end - reg - c * int(phase))
        if expected_pos in positions_set:
            in_phase_sum += pos_abun[expected_pos]
            n_filled += 1

    total_in_window = sum(pos_abun[p] for p in window_positions)
    return in_phase_sum, total_in_window, n_filled

def _evaluate_register(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """
    Wobble-tolerant register evaluation (±1 positional wobble).
    Returns: (in_phase_sum, effective_total, n_filled_cycles)
    Semantics:
      - If the exact expected position exists, count it for in-phase and quarantine its ±1 neighbors.
      - Else, pick the better of the ±1 neighbors (if any) and quarantine the sibling neighbor.
      - Each genomic position is counted at most once in-phase.
      - Effective total excludes quarantined neighbors so they don't inflate U.
    """
    ph = int(phase)
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // ph)

    used_positions = set()     # positions used as in-phase
    ignored_positions = set()  # neighbors to exclude from effective_total
    in_phase_sum = 0.0
    n_filled = 0

    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * ph) if forward else (win_end - reg - c * ph)

        # Case 1: exact exists -> use and quarantine neighbors
        if expected_pos in positions_set and expected_pos not in used_positions:
            in_phase_sum += pos_abun[expected_pos]
            used_positions.add(expected_pos)
            n_filled += 1
            for off in (-1, 1):
                npos = expected_pos + off
                if npos in positions_set and npos not in used_positions:
                    ignored_positions.add(npos)
        else:
            # Case 2: consider ±1; choose best if present; quarantine sibling neighbor
            left = expected_pos - 1
            right = expected_pos + 1
            candidates = []
            if left in positions_set and left not in used_positions:
                candidates.append(left)
            if right in positions_set and right not in used_positions:
                candidates.append(right)
            if candidates:
                best = max(candidates, key=lambda p: pos_abun[p])
                in_phase_sum += pos_abun[best]
                used_positions.add(best)
                n_filled += 1
                sibling = right if best == left else left
                if sibling in positions_set and sibling not in used_positions:
                    ignored_positions.add(sibling)

    # Effective total excludes quarantined neighbors of exact/selected hits
    effective_positions = [p for p in window_positions if p not in ignored_positions]
    effective_total = sum(pos_abun[p] for p in effective_positions)
    return in_phase_sum, effective_total, n_filled


def _score_relaxed_window(window_positions, pos_abun, win_start, win_end, phase, forward=True):
    summary = _score_window_registers(
        window_positions,
        pos_abun,
        win_start,
        win_end,
        phase,
        forward=forward,
    )
    return (
        float(summary["score"]),
        float(summary["best_in_phase_sum"]),
        float(summary["best_out_of_phase"]),
        int(summary["best_filled"]),
        summary["best_register"],
    )


def _enumerate_relaxed_trace_for_strand(pos_abun, phase, win_size, *, forward=True):
    positions = sorted(pos_abun.keys())
    if not positions:
        return []

    if forward:
        anchors = positions
    else:
        anchors = list(reversed(positions))

    trace = []
    for anchor in anchors:
        if forward:
            win_start = int(anchor)
            win_end = int(anchor) + int(win_size) - 1
            anchor_position = int(anchor)
        else:
            win_end = int(anchor)
            win_start = int(anchor) - int(win_size) + 1
            anchor_position = int(anchor)

        window_positions = [p for p in positions if win_start <= p <= win_end]
        score, in_phase_sum, out_of_phase, n_filled, best_reg = _score_relaxed_window(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            int(phase),
            forward=forward,
        )
        trace.append(
            {
                "anchor_position": anchor_position,
                "window_start": int(win_start),
                "window_end": int(win_end),
                "score": float(score),
                "in_phase_abund": float(in_phase_sum),
                "out_phase_abund": float(out_of_phase),
                "occupied_cycles": int(n_filled),
                "best_register": best_reg,
            }
        )
    return trace


def enumerate_relaxed_howell_trace(aclust: pd.DataFrame, phase: int | None = None):
    ph = int(phase) if phase is not None else _phase_value()
    win_size = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust["pos"].min())
    seq_end = int(aclust["pos"].max())
    w_mask, c_mask = _strand_masks(aclust)

    w_trace = []
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        if w_pos_abun:
            w_trace = _enumerate_relaxed_trace_for_strand(
                w_pos_abun,
                int(ph),
                int(win_size),
                forward=True,
            )

    c_trace = []
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        if c_pos_abun:
            c_trace = _enumerate_relaxed_trace_for_strand(
                c_pos_abun,
                int(ph),
                int(win_size),
                forward=False,
            )

    return {"w": w_trace, "c": c_trace}


def _best_sliding_window_score_generic_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    positions = sorted(pos_abun.keys())
    if not positions:
        return 0.0, None, None

    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound

    best_score = -float("inf")
    best_window = (None, None)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        if not window_positions:
            score = 0.0
        else:
            num_cycles = max(0, (win_end - win_start + 1) // int(phase))
            if num_cycles < 4:
                score = 0.0
            else:
                best_reg_sum = 0.0
                best_reg_total = 0.0
                best_reg_filled = 0

                for reg in range(int(phase)):
                    in_sum, total, n_filled = _evaluate_register_strict_exact(
                        window_positions, pos_abun, win_start, win_end, int(phase), reg, forward=forward
                    )
                    if in_sum > best_reg_sum:
                        best_reg_sum = in_sum
                        best_reg_total = total
                        best_reg_filled = n_filled

                out_of_phase = max(0.0, best_reg_total - best_reg_sum)
                numerator = best_reg_sum
                denominator = 1.0 + out_of_phase
                if numerator <= 0.0 or not (best_reg_filled > 3):
                    score = 0.0
                else:
                    log_arg = 1.0 + 10.0 * (numerator / denominator)
                    if log_arg <= 0.0 or log_arg != log_arg:
                        score = 0.0
                    else:
                        scale = max(min(best_reg_filled, num_cycles) - 2, 0)
                        score = scale * (0.0 if log_arg <= 0 else np.log(log_arg))

        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    return best_score if best_score != -float("inf") else 0.0, best_window[0], best_window[1]

def best_sliding_window_score_forward_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )

def best_sliding_window_score_reverse_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

def compute_phasing_score_Howell_strict(aclust: pd.DataFrame):
    """
    Classic Howell phasing WITHOUT positional wobble.
    Uses ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    """
    ph = _phase_value()
    win_size  = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        w_score, w_s, w_e = (
            best_sliding_window_score_forward_strict(w_pos_abun, int(ph), win_size, seq_start, seq_end)
            if w_pos_abun else (0.0, None, None)
        )
    else:
        w_score, w_s, w_e = None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        c_score, c_s, c_e = (
            best_sliding_window_score_reverse_strict(c_pos_abun, int(ph), win_size, seq_start, seq_end)
            if c_pos_abun else (0.0, None, None)
        )
    else:
        c_score, c_s, c_e = None, None, None

    return (w_score, (w_s, w_e), c_score, (c_s, c_e))

def process_chromosome_features(chromosome_df: pd.DataFrame):
    """
    Build per-cluster feature rows (wobble + strict Howell).
    Returns rows matching FEATURE_COLS order.
    Expected columns:
      ['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']
    """
    ph = _phase_value()
    ensure_win_score_lookup_ready()

    df = chromosome_df[['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']].copy()
    df['pos']  = pd.to_numeric(df['pos'], errors='coerce')
    df['len']  = pd.to_numeric(df['len'], errors='coerce')
    df['abun'] = pd.to_numeric(df['abun'], errors='coerce').fillna(0)
    df = df.dropna(subset=['pos', 'len'])

    # Normalize score lookup keys once per chromosome chunk
    raw_lookup = st.WIN_SCORE_LOOKUP or {}
    score_lookup = {}
    for k, v in raw_lookup.items():
        try:
            nk = ids.normalize_cluster_id_for_lookup(str(k), phase=ph)
        except Exception:
            nk = str(k).strip()
        score_lookup[nk] = v
        # keep raw key as a fallback too
        score_lookup[str(k).strip()] = v

    rows = []
    for cID, aclust in df.groupby('clusterID', sort=False):
        if aclust.empty:
            continue

        cid_raw = str(cID).strip()
        try:
            cid_norm = ids.normalize_cluster_id_for_lookup(cid_raw, phase=ph)
        except Exception:
            cid_norm = cid_raw

        # derive genomic span from this cluster's rows (works even if lookup misses)
        achr  = str(aclust['chromosome'].iloc[0])
        start = int(aclust['pos'].min()); end = int(aclust['pos'].max())

        # Universal identifier (prefer mergedClusterDict mapping; fallback to coords)
        uid = None
        try:
            uid = ids.getUniversalID(cid_raw)
        except Exception:
            uid = None
        if not uid:
            try:
                uid = ids.getUniversalID(cid_norm)
            except Exception:
                uid = None

        if not uid or ":" not in uid or ".." not in uid:
            uid = f"{achr}:{start}..{end}"  # hard fallback: always coordinate-style

        identifier = uid
        alib = str(aclust['alib'].iloc[0])

        # Normalize strand labels
        s_norm = aclust['strand'].astype(str).str.lower()
        w_mask = s_norm.isin(['w', '+', 'watson', '1', 'true'])
        c_mask = s_norm.isin(['c', '-', 'crick', '0', 'false'])

        # Strand bias
        total_w = int(w_mask.sum())
        total_c = int(c_mask.sum())
        denom = total_w + total_c
        strand_bias = (total_w / denom) if denom > 0 else 1.0

        # Abundance ratios
        sum_abun_len_phase = aclust.loc[aclust['len'] == int(ph), 'abun'].sum()
        sum_abun_other_len = aclust.loc[aclust['len'] != int(ph), 'abun'].sum()
        ratio_abund_len_phase = (sum_abun_len_phase / sum_abun_other_len) if sum_abun_other_len > 0 else 1.0

        # Totals and cluster length
        total_abund = float(aclust['abun'].sum())
        aclust_len = max(end - start, 0)

        # CLNC for phase-length reads (legacy):
        #CLNC = (phase-length abundance) / (cluster_length - phase)
        w_sum_abun_len_phase = aclust.loc[(aclust['len'] == int(ph)) & w_mask, 'abun'].sum()
        c_sum_abun_len_phase = aclust.loc[(aclust['len'] == int(ph)) & c_mask, 'abun'].sum()
        denom_len = max(aclust_len - int(ph), 0)
        clnc = ((w_sum_abun_len_phase + c_sum_abun_len_phase) / denom_len) if denom_len > 0 else 0.0
        log_CLNC = float(np.log10(clnc + 1.0))
        # Complexity
        #   distinct tag sequences / total abundance (all reads)
        distinct_tags = int(aclust['tag_seq'].nunique(dropna=True))
        complexity = (distinct_tags / total_abund) if total_abund > 0 else 0.0

        # Default scores
        aclust_phasis_score = 0.0
        aclust_fishers_combined = 1.0

        tup = score_lookup.get(cid_norm)
        if tup is None:
            # Sometimes scores might be keyed by the universal ID already; try that too.
            tup = score_lookup.get(uid)
        if tup is None:
            tup = score_lookup.get(cid_raw)

        if tup is not None:
            ps, cf = tup
            if ps is not None:
                try:
                    aclust_phasis_score = float(ps)
                except Exception:
                    pass
            if cf is not None:
                try:
                    aclust_fishers_combined = float(cf)
                except Exception:
                    pass

        # Howell (wobble-tolerant)
        (w_Howell, (w_s, w_e),
         c_Howell, (c_s, c_e),
         peak_howell_detail) = compute_phasing_score_Howell(aclust, return_detail=True)
        Peak_Howell = None if (w_Howell is None and c_Howell is None) else max([x for x in (w_Howell, c_Howell) if x is not None])
        if peak_howell_detail:
            exact_support_score = peak_howell_detail.get("Howell_exact_support_score", np.nan)
            ambiguity_count = peak_howell_detail.get("Howell_ambiguity_count", np.nan)
            alt_register_count = peak_howell_detail.get("Howell_alt_register_count", np.nan)
            overlap_margin = peak_howell_detail.get("Howell_overlap_margin", np.nan)
        else:
            exact_support_score = np.nan
            ambiguity_count = np.nan
            alt_register_count = np.nan
            overlap_margin = np.nan

        # Howell (classic strict)
        (w_Howell_strict, (w_s_strict, w_e_strict),
         c_Howell_strict, (c_s_strict, c_e_strict)) = compute_phasing_score_Howell_strict(aclust)
        Peak_Howell_strict = None if (w_Howell_strict is None and c_Howell_strict is None)                              else max([x for x in (w_Howell_strict, c_Howell_strict) if x is not None])

        rows.append([
            identifier, cid_raw, alib,
            float(complexity), float(strand_bias), float(log_CLNC),
            float(ratio_abund_len_phase), aclust_phasis_score, aclust_fishers_combined,
            float(total_abund),
            # wobble-tolerant
            w_Howell, w_s, w_e, c_Howell, c_s, c_e, Peak_Howell,
            exact_support_score, ambiguity_count, alt_register_count, overlap_margin,
            # classic (strict)
            w_Howell_strict, w_s_strict, w_e_strict, c_Howell_strict, c_s_strict, c_e_strict, Peak_Howell_strict
        ])

    return rows
