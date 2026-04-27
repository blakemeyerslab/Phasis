from __future__ import annotations


from .. import state as st
from .. import ids
import os
import re
import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.cache import MemCache, default_memfile_path, phase2_basename, stage_signature
from phasis.parallel import run_parallel_with_progress

DCL_OVERHANG = 3          # 2-nt 3' overhang in duplex -> 3-nt genomic offset
WINDOW_MULTIPLIER = 10    # 10 cycles per window
FEATURE_SCHEMA_VERSION = 7
HOWELL_AMBIGUITY_FRACTION = 0.90
HOWELL_CROWDING_SCORE_GAP = 4.0


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
 'Howell_extension_window_count',
 'Howell_extension_span_nt',
 'Howell_origin_window_count',
 'Howell_origin_frame_count',
 'Howell_origin_margin',
 'Howell_origin_class',
 'Howell_additional_peak_count',
 'Howell_additional_peak_best_score',
 'Howell_overlapping_alt_count',
 'Howell_overlapping_alt_best_score',
 'Howell_overlapping_alt_best_shift_nt',
 'Howell_crowding_window_count',
 'Howell_crowding_best_score',
 'Howell_crowding_score_gap',
 'w_Howell_score_strict',
 'w_window_start_strict',
 'w_window_end_strict',
 'c_Howell_score_strict',
 'c_window_start_strict',
 'c_window_end_strict',
 'Peak_Howell_score_strict']

# Numeric columns (all except id-like/text fields)
NUMERIC_COLS = set(FEATURE_COLS) - {"identifier", "cID", "alib", "Howell_origin_class"}


def _phase_value(default: int = 21) -> int:
    """Return rt.phase as an int (module-level, spawn-safe)."""
    try:
        v = getattr(rt, "phase", None)
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _min_howell_score_value(default: float = 12.5) -> float:
    """Return rt.min_Howell_score as a float, falling back safely."""
    try:
        value = getattr(rt, "min_Howell_score", None)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _get_memfile() -> str:
    """Return runtime memFile; default to the run directory if missing."""
    mem = getattr(rt, "memFile", None)
    if mem:
        return str(mem)
    mem = default_memfile_path()
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


def _window_scan_bounds(positions, win_size, seq_start=None, seq_end=None):
    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound
    return int(lower_bound), int(upper_bound)


def _canonical_exact_frame(
    *,
    window_start: int | None,
    window_end: int | None,
    exact_best_register,
    strand_code: str | None,
    phase: int,
) -> int | None:
    try:
        reg_value = int(exact_best_register)
    except Exception:
        return None

    try:
        phase_local = int(phase)
    except Exception:
        return None
    if phase_local <= 0:
        return None

    strand_text = str(strand_code or "").strip().lower()
    try:
        if strand_text == "c":
            return int((int(window_end) - reg_value) % phase_local)
        return int((int(window_start) + reg_value) % phase_local)
    except Exception:
        return None


def _exact_register_origin(
    *,
    window_start: int | None,
    window_end: int | None,
    exact_best_register,
    strand_code: str | None,
):
    try:
        reg_value = int(exact_best_register)
    except Exception:
        return None

    try:
        strand_text = str(strand_code or "").strip().lower()
        if strand_text == "c":
            return int(window_end) - reg_value
        return int(window_start) + reg_value
    except Exception:
        return None


def _enumerate_relaxed_candidate_windows(
    pos_abun,
    phase,
    win_size,
    seq_start=None,
    seq_end=None,
    *,
    forward=True,
):
    positions = sorted(pos_abun.keys())
    if not positions:
        return [], None

    lower_bound, upper_bound = _window_scan_bounds(
        positions,
        win_size,
        seq_start=seq_start,
        seq_end=seq_end,
    )

    best_score = -float("inf")
    best_detail = None
    candidate_windows = []
    phase_local = int(phase)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        relaxed_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            phase_local,
            forward=forward,
        )
        exact_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            phase_local,
            forward=forward,
            exact_only=True,
        )
        exact_best_register = exact_summary.get("best_register")
        strand_code = "w" if forward else "c"
        candidate = {
            "strand": strand_code,
            "anchor_position": int(win_start if forward else win_end),
            "window_start": int(win_start),
            "window_end": int(win_end),
            "score": float(relaxed_summary["score"]),
            "best_register": relaxed_summary.get("best_register"),
            "exact_score": float(exact_summary["score"]),
            "exact_best_register": exact_best_register,
            "exact_frame": _canonical_exact_frame(
                window_start=win_start,
                window_end=win_end,
                exact_best_register=exact_best_register,
                strand_code=strand_code,
                phase=phase_local,
            ),
            "exact_register_position": _exact_register_origin(
                window_start=win_start,
                window_end=win_end,
                exact_best_register=exact_best_register,
                strand_code=strand_code,
            ),
        }
        candidate_windows.append(candidate)

        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best_detail = {
                "strand": strand_code,
                "anchor_position": candidate["anchor_position"],
                "window_start": candidate["window_start"],
                "window_end": candidate["window_end"],
                "score": candidate["score"],
                "best_register": candidate["best_register"],
                "register_scores": list(relaxed_summary.get("register_scores") or []),
                "exact_score": candidate["exact_score"],
                "exact_best_register": candidate["exact_best_register"],
                "exact_register_scores": list(exact_summary.get("register_scores") or []),
                "exact_frame": candidate["exact_frame"],
                "exact_register_position": candidate["exact_register_position"],
            }

    return candidate_windows, best_detail


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
        "winner_exact_frame": best_window_detail.get("exact_frame"),
        "Howell_exact_support_score": float(exact_support_score),
        "best_overlapping_competitor_score": np.nan,
        "Howell_ambiguity_count": np.nan,
        "Howell_alt_register_count": np.nan,
        "Howell_overlap_margin": np.nan,
        "Howell_extension_window_count": np.nan,
        "Howell_extension_span_nt": np.nan,
        "Howell_origin_window_count": np.nan,
        "Howell_origin_frame_count": np.nan,
        "Howell_origin_margin": np.nan,
        "Howell_origin_class": "insufficient_exact_support",
    }

    winner_exact_frame = result.get("winner_exact_frame")
    if exact_support_score <= 0.0 or winner_exact_frame is None:
        return result

    threshold_score = float(threshold_fraction) * exact_support_score
    overlap_best_score = None
    overlap_count = 0
    same_frame_count = 0
    competing_frame_count = 0
    competing_frames = set()
    best_competing_frame_score = None
    extension_min_start = int(best_window_detail.get("window_start"))
    extension_max_end = int(best_window_detail.get("window_end"))
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
        candidate_exact_frame = candidate.get("exact_frame")
        if candidate_exact_frame == winner_exact_frame:
            same_frame_count += 1
            extension_min_start = min(extension_min_start, int(candidate.get("window_start")))
            extension_max_end = max(extension_max_end, int(candidate.get("window_end")))
        else:
            competing_frame_count += 1
            if candidate_exact_frame is not None:
                competing_frames.add(int(candidate_exact_frame))
            if best_competing_frame_score is None or candidate_score > best_competing_frame_score:
                best_competing_frame_score = candidate_score

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
    result["Howell_extension_window_count"] = int(same_frame_count)
    result["Howell_extension_span_nt"] = int(extension_max_end - extension_min_start + 1)
    result["Howell_origin_window_count"] = int(competing_frame_count)
    result["Howell_origin_frame_count"] = int(len(competing_frames))
    result["Howell_origin_margin"] = (
        np.nan
        if best_competing_frame_score is None
        else float(exact_support_score - float(best_competing_frame_score))
    )
    if overlap_count == 0:
        result["Howell_origin_class"] = "unique_origin"
    elif same_frame_count > 0 and competing_frame_count == 0:
        result["Howell_origin_class"] = "coherent_extension"
    elif same_frame_count == 0 and competing_frame_count > 0:
        result["Howell_origin_class"] = "ambiguous_origin"
    else:
        result["Howell_origin_class"] = "mixed_extension_and_ambiguity"
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

    lower_bound, upper_bound = _window_scan_bounds(
        positions,
        win_size,
        seq_start=seq_start,
        seq_end=seq_end,
    )

    if return_detail:
        candidate_windows, best_detail = _enumerate_relaxed_candidate_windows(
            pos_abun,
            int(phase),
            win_size,
            seq_start=lower_bound,
            seq_end=upper_bound + win_size - 1,
            forward=forward,
        )
        if best_detail is None:
            return 0.0, None, None, None
        detail = _summarize_peak_howell_ambiguity(best_detail, candidate_windows)
        return (
            float(best_detail["score"]),
            int(best_detail["window_start"]),
            int(best_detail["window_end"]),
            detail,
        )

    best_score = -float("inf")
    best_window = (None, None)

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
        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    resolved_score = best_score if best_score != -float("inf") else 0.0
    return resolved_score, best_window[0], best_window[1]


def collect_exact_only_peak_competitors(
    aclust: pd.DataFrame,
    *,
    phase: int | None = None,
    threshold_fraction: float = HOWELL_AMBIGUITY_FRACTION,
) -> dict:
    ph = int(phase) if phase is not None else _phase_value()
    win_size = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust["pos"].min())
    seq_end = int(aclust["pos"].max())
    w_mask, c_mask = _strand_masks(aclust)

    all_candidates = []
    winner_detail = None
    winner_score = -float("inf")

    for strand_code, mask, forward in (("w", w_mask, True), ("c", c_mask, False)):
        if not mask.any():
            continue
        pos_abun = _build_pos_abun_exact_phase(aclust.loc[mask], seq_start, seq_end, int(ph))
        if not pos_abun:
            continue
        candidates, best_detail = _enumerate_relaxed_candidate_windows(
            pos_abun,
            int(ph),
            win_size,
            seq_start=seq_start,
            seq_end=seq_end,
            forward=forward,
        )
        all_candidates.extend(candidates)
        if best_detail is None:
            continue
        score = float(best_detail.get("score", 0.0) or 0.0)
        if score >= winner_score:
            winner_score = score
            winner_detail = best_detail

    summary = _summarize_peak_howell_ambiguity(
        winner_detail,
        all_candidates,
        threshold_fraction=threshold_fraction,
    )
    if not winner_detail or not summary:
        return {
            "winner_detail": winner_detail,
            "summary": summary,
            "competing_windows": [],
        }

    exact_support_score = float(summary.get("Howell_exact_support_score", 0.0) or 0.0)
    winner_exact_frame = summary.get("winner_exact_frame")
    threshold_score = float(threshold_fraction) * exact_support_score
    competing_windows = []
    for candidate in all_candidates:
        if str(candidate.get("strand")) != str(winner_detail.get("strand")):
            continue
        if (
            int(candidate.get("window_start")) == int(winner_detail.get("window_start"))
            and int(candidate.get("window_end")) == int(winner_detail.get("window_end"))
        ):
            continue
        if not _windows_overlap(
            candidate.get("window_start"),
            candidate.get("window_end"),
            winner_detail.get("window_start"),
            winner_detail.get("window_end"),
        ):
            continue
        candidate_score = float(candidate.get("exact_score", 0.0) or 0.0)
        if candidate_score < threshold_score:
            continue
        if candidate.get("exact_frame") == winner_exact_frame:
            continue
        competing_windows.append(dict(candidate))

    return {
        "winner_detail": winner_detail,
        "summary": summary,
        "competing_windows": competing_windows,
    }

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


def _find_relaxed_trace_peak_row(trace_rows) -> dict | None:
    best_row = None
    best_score = float("-inf")
    for row in trace_rows or []:
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score >= best_score:
            best_score = score
            best_row = row
    return best_row


def _same_trace_window(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    try:
        return (
            int(a.get("anchor_position")) == int(b.get("anchor_position"))
            and int(a.get("window_start")) == int(b.get("window_start"))
            and int(a.get("window_end")) == int(b.get("window_end"))
        )
    except Exception:
        return False


def _normalize_trace_strand_code(strand_code) -> str:
    text = str(strand_code or "").strip().lower()
    if text in {"c", "-", "crick", "0", "false"}:
        return "c"
    return "w"


def _is_forward_trace_strand(strand_code) -> bool:
    return _normalize_trace_strand_code(strand_code) == "w"


def _build_relaxed_trace_register_origin(peak_row: dict | None, phase: int, strand_code) -> int | None:
    if not peak_row:
        return None
    best_register = peak_row.get("best_register")
    if best_register is None:
        return None

    if _is_forward_trace_strand(strand_code):
        return int(peak_row["window_start"]) + int(best_register)
    return int(peak_row["window_end"]) - int(best_register)


def _classify_relaxed_trace_relation(anchor_position: int, register_origin: int | None, phase: int):
    if register_origin is None:
        return "other", None

    delta = int(anchor_position) - int(register_origin)
    remainder = delta % int(phase)
    if remainder == 0:
        return "exact", int(anchor_position)
    if remainder == 1:
        return "offset", int(anchor_position) - 1
    if remainder == int(phase) - 1:
        return "offset", int(anchor_position) + 1
    return "other", None


def _compute_phase_shift_nt(
    reference_origin: int | None,
    candidate_origin: int | None,
    phase: int,
) -> int | None:
    if reference_origin is None or candidate_origin is None:
        return None

    try:
        phase_local = int(phase)
        if phase_local <= 0:
            return None
        shift_value = (int(candidate_origin) - int(reference_origin)) % phase_local
    except Exception:
        return None

    return None if shift_value == 0 else int(shift_value)


def classify_browser_style_relaxed_trace(
    trace: dict,
    *,
    phase: int | None = None,
    crowded_gap: float = HOWELL_CROWDING_SCORE_GAP,
) -> dict:
    """
    Classify relaxed trace windows using browser-style semantics.

    Returns a dict with per-strand classified rows plus crowding metrics derived
    from non-in-phase windows on the winning strand that overlap the relaxed
    HPSP window and stay within the requested Howell score gap.
    """
    ph = int(phase) if phase is not None else _phase_value()
    result = {
        "w": [],
        "c": [],
        "strand_hpsp_rows": {"w": None, "c": None},
        "strand_register_origins": {"w": None, "c": None},
        "winner_strand": None,
        "winner_row": None,
        "Howell_crowding_window_count": 0,
        "Howell_crowding_best_score": np.nan,
        "Howell_crowding_score_gap": np.nan,
        "crowding_rows": [],
    }

    winner_strand = None
    winner_row = None
    winner_score = float("-inf")
    for strand_code in ("w", "c"):
        peak_row = _find_relaxed_trace_peak_row(trace.get(strand_code, []) or [])
        result["strand_hpsp_rows"][strand_code] = peak_row
        register_origin = _build_relaxed_trace_register_origin(peak_row, ph, strand_code)
        result["strand_register_origins"][strand_code] = register_origin
        if peak_row is None:
            continue
        try:
            peak_score = float(peak_row.get("score", 0.0) or 0.0)
        except Exception:
            peak_score = 0.0
        if peak_score >= winner_score:
            winner_score = peak_score
            winner_strand = strand_code
            winner_row = peak_row

    result["winner_strand"] = winner_strand
    result["winner_row"] = winner_row

    for strand_code in ("w", "c"):
        peak_row = result["strand_hpsp_rows"][strand_code]
        register_origin = result["strand_register_origins"][strand_code]
        strand_rows = []
        for row in trace.get(strand_code, []) or []:
            try:
                anchor_position = int(row.get("anchor_position"))
            except Exception:
                continue
            relation, expected_position = _classify_relaxed_trace_relation(
                anchor_position,
                register_origin,
                ph,
            )
            classified = dict(row)
            classified["strand"] = strand_code
            classified["phase_relation"] = relation
            classified["expected_register_position"] = expected_position
            classified["register_origin"] = register_origin
            classified["is_hpsp"] = _same_trace_window(row, peak_row)
            strand_rows.append(classified)
        result[strand_code] = strand_rows

    crowding_rows = []
    if winner_row is not None and winner_strand is not None:
        for row in result.get(winner_strand, []) or []:
            if row.get("is_hpsp"):
                continue
            if str(row.get("phase_relation")) != "other":
                continue
            if not _windows_overlap(
                row.get("window_start"),
                row.get("window_end"),
                winner_row.get("window_start"),
                winner_row.get("window_end"),
            ):
                continue
            try:
                candidate_score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                continue
            if abs(float(winner_score) - candidate_score) > float(crowded_gap):
                continue
            crowding_rows.append(dict(row))

    result["crowding_rows"] = crowding_rows
    result["Howell_crowding_window_count"] = int(len(crowding_rows))
    if crowding_rows:
        best_crowding_score = max(float(row.get("score", 0.0) or 0.0) for row in crowding_rows)
        result["Howell_crowding_best_score"] = float(best_crowding_score)
        result["Howell_crowding_score_gap"] = float(float(winner_score) - best_crowding_score)
    return result


def _group_trace_rows_by_overlap(trace_rows, *, score_cutoff: float) -> list[dict]:
    qualifying_rows = []
    for row in trace_rows or []:
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            continue
        if score < float(score_cutoff):
            continue
        qualifying_rows.append(row)

    qualifying_rows.sort(
        key=lambda row: (
            int(row.get("window_start")),
            int(row.get("window_end")),
            int(row.get("anchor_position")),
        )
    )

    groups = []
    for row in qualifying_rows:
        window_start = int(row.get("window_start"))
        window_end = int(row.get("window_end"))
        if not groups or window_start > groups[-1]["max_end"]:
            groups.append(
                {
                    "rows": [row],
                    "min_start": window_start,
                    "max_end": window_end,
                }
            )
        else:
            groups[-1]["rows"].append(row)
            groups[-1]["min_start"] = min(groups[-1]["min_start"], window_start)
            groups[-1]["max_end"] = max(groups[-1]["max_end"], window_end)
    return groups


def _build_relaxed_group_summary(
    group: dict,
    *,
    strand_code: str,
    category: str,
    phase: int | None = None,
    winner_register_origin: int | None = None,
) -> dict | None:
    rows = list(group.get("rows") or [])
    if not rows:
        return None

    peak_row = _find_relaxed_trace_peak_row(rows)
    if peak_row is None:
        return None

    ph = int(phase) if phase is not None else None
    register_origin = None
    shift_nt = None
    if ph is not None:
        register_origin = _build_relaxed_trace_register_origin(peak_row, ph, strand_code)
        shift_nt = _compute_phase_shift_nt(winner_register_origin, register_origin, ph)

    try:
        peak_score = float(peak_row.get("score", 0.0) or 0.0)
    except Exception:
        peak_score = 0.0

    return {
        "category": str(category),
        "strand": str(strand_code),
        "rows": rows,
        "peak_row": peak_row,
        "peak_score": float(peak_score),
        "register_origin": register_origin,
        "shift_nt": shift_nt,
        "min_start": int(group.get("min_start")),
        "max_end": int(group.get("max_end")),
    }


def _group_overlapping_alternative_candidates(
    main_group_rows,
    winner_row: dict | None,
    *,
    phase: int,
    strand_code: str,
    score_cutoff: float,
) -> list[dict]:
    if winner_row is None:
        return []

    winner_register_origin = _build_relaxed_trace_register_origin(winner_row, phase, strand_code)
    if winner_register_origin is None:
        return []

    grouped = {}
    for row in main_group_rows or []:
        if _same_trace_window(row, winner_row):
            continue
        if not _windows_overlap(
            row.get("window_start"),
            row.get("window_end"),
            winner_row.get("window_start"),
            winner_row.get("window_end"),
        ):
            continue

        try:
            candidate_score = float(row.get("score", 0.0) or 0.0)
            anchor_position = int(row.get("anchor_position"))
        except Exception:
            continue
        if candidate_score < float(score_cutoff):
            continue

        relation_to_winner, _ = _classify_relaxed_trace_relation(
            anchor_position,
            winner_register_origin,
            int(phase),
        )
        if relation_to_winner != "other":
            continue

        candidate_register_origin = _build_relaxed_trace_register_origin(row, int(phase), strand_code)
        shift_nt = _compute_phase_shift_nt(winner_register_origin, candidate_register_origin, int(phase))
        if shift_nt is None:
            continue

        shift_group = grouped.setdefault(
            int(shift_nt),
            {
                "rows": [],
                "min_start": int(row.get("window_start")),
                "max_end": int(row.get("window_end")),
            },
        )
        shift_group["rows"].append(dict(row))
        shift_group["min_start"] = min(shift_group["min_start"], int(row.get("window_start")))
        shift_group["max_end"] = max(shift_group["max_end"], int(row.get("window_end")))

    summaries = []
    for shift_nt, group in grouped.items():
        summary = _build_relaxed_group_summary(
            group,
            strand_code=strand_code,
            category="overlapping_alternative",
            phase=int(phase),
            winner_register_origin=winner_register_origin,
        )
        if summary is None:
            continue
        summary["shift_nt"] = int(shift_nt)
        summaries.append(summary)

    summaries.sort(key=lambda item: (-float(item.get("peak_score", 0.0) or 0.0), int(item.get("shift_nt") or 0)))
    return summaries


def summarize_relaxed_trace_subregions(
    trace: dict,
    *,
    score_cutoff: float | None = None,
    phase: int | None = None,
) -> dict:
    """
    Summarize additional relaxed Howell peak regions across the whole locus.

    These are intentionally separate from the exact-only HPSP ambiguity metrics:
    we group supra-threshold relaxed trace windows by genomic overlap on each
    strand, identify the region containing the relaxed trace HPSP as the main
    region, and count any other supra-threshold regions as alternative
    phased-like subregions.
    """
    cutoff = _min_howell_score_value() if score_cutoff is None else float(score_cutoff)
    phase_local = int(phase) if phase is not None else _phase_value()
    additional_region_scores = []
    additional_peak_groups = []
    overlapping_alt_groups = []
    global_peak_strand = None
    global_peak_row = None
    global_peak_score = float("-inf")

    for strand_code in ("w", "c"):
        strand_peak = _find_relaxed_trace_peak_row(trace.get(strand_code, []) or [])
        if strand_peak is None:
            continue
        try:
            strand_peak_score = float(strand_peak.get("score", 0.0) or 0.0)
        except Exception:
            strand_peak_score = 0.0
        if strand_peak_score >= global_peak_score:
            global_peak_score = strand_peak_score
            global_peak_strand = strand_code
            global_peak_row = strand_peak

    for strand_code in ("w", "c"):
        strand_rows = list(trace.get(strand_code, []) or [])
        if not strand_rows:
            continue

        groups = _group_trace_rows_by_overlap(strand_rows, score_cutoff=cutoff)
        if not groups:
            continue

        main_group_index = None
        for idx, group in enumerate(groups):
            if strand_code == global_peak_strand and any(_same_trace_window(row, global_peak_row) for row in group["rows"]):
                main_group_index = idx
                break

        if strand_code == global_peak_strand and main_group_index is not None:
            overlapping_alt_groups.extend(
                _group_overlapping_alternative_candidates(
                    groups[main_group_index]["rows"],
                    global_peak_row,
                    phase=phase_local,
                    strand_code=strand_code,
                    score_cutoff=cutoff,
                )
            )

        for idx, group in enumerate(groups):
            if main_group_index is not None and idx == main_group_index:
                continue
            group_summary = _build_relaxed_group_summary(
                group,
                strand_code=strand_code,
                category="other_local_peak",
                phase=phase_local,
            )
            if group_summary is None:
                continue
            additional_peak_groups.append(group_summary)
            additional_region_scores.append(float(group_summary.get("peak_score", 0.0) or 0.0))

    best_overlap_group = None
    if overlapping_alt_groups:
        best_overlap_group = max(
            overlapping_alt_groups,
            key=lambda item: float(item.get("peak_score", 0.0) or 0.0),
        )

    return {
        "Howell_additional_peak_count": int(len(additional_region_scores)),
        "Howell_additional_peak_best_score": (
            np.nan if not additional_region_scores else float(max(additional_region_scores))
        ),
        "Howell_overlapping_alt_count": int(len(overlapping_alt_groups)),
        "Howell_overlapping_alt_best_score": (
            np.nan if best_overlap_group is None else float(best_overlap_group.get("peak_score", 0.0) or 0.0)
        ),
        "Howell_overlapping_alt_best_shift_nt": (
            np.nan if best_overlap_group is None else float(best_overlap_group.get("shift_nt"))
        ),
        "additional_peak_groups": additional_peak_groups,
        "overlapping_alt_groups": overlapping_alt_groups,
    }


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
        relaxed_trace = enumerate_relaxed_howell_trace(aclust, phase=ph)
        browser_trace_summary = classify_browser_style_relaxed_trace(relaxed_trace, phase=ph)
        additional_peak_summary = summarize_relaxed_trace_subregions(
            relaxed_trace,
            score_cutoff=_min_howell_score_value(),
            phase=ph,
        )
        if peak_howell_detail:
            exact_support_score = peak_howell_detail.get("Howell_exact_support_score", np.nan)
            ambiguity_count = peak_howell_detail.get("Howell_ambiguity_count", np.nan)
            alt_register_count = peak_howell_detail.get("Howell_alt_register_count", np.nan)
            overlap_margin = peak_howell_detail.get("Howell_overlap_margin", np.nan)
            extension_window_count = peak_howell_detail.get("Howell_extension_window_count", np.nan)
            extension_span_nt = peak_howell_detail.get("Howell_extension_span_nt", np.nan)
            origin_window_count = peak_howell_detail.get("Howell_origin_window_count", np.nan)
            origin_frame_count = peak_howell_detail.get("Howell_origin_frame_count", np.nan)
            origin_margin = peak_howell_detail.get("Howell_origin_margin", np.nan)
            origin_class = peak_howell_detail.get("Howell_origin_class", np.nan)
        else:
            exact_support_score = np.nan
            ambiguity_count = np.nan
            alt_register_count = np.nan
            overlap_margin = np.nan
            extension_window_count = np.nan
            extension_span_nt = np.nan
            origin_window_count = np.nan
            origin_frame_count = np.nan
            origin_margin = np.nan
            origin_class = np.nan
        additional_peak_count = additional_peak_summary.get("Howell_additional_peak_count", np.nan)
        additional_peak_best_score = additional_peak_summary.get("Howell_additional_peak_best_score", np.nan)
        overlapping_alt_count = additional_peak_summary.get("Howell_overlapping_alt_count", 0)
        overlapping_alt_best_score = additional_peak_summary.get("Howell_overlapping_alt_best_score", np.nan)
        overlapping_alt_best_shift_nt = additional_peak_summary.get("Howell_overlapping_alt_best_shift_nt", np.nan)
        crowding_window_count = browser_trace_summary.get("Howell_crowding_window_count", 0)
        crowding_best_score = browser_trace_summary.get("Howell_crowding_best_score", np.nan)
        crowding_score_gap = browser_trace_summary.get("Howell_crowding_score_gap", np.nan)

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
            extension_window_count, extension_span_nt, origin_window_count, origin_frame_count, origin_margin, origin_class,
            additional_peak_count, additional_peak_best_score,
            overlapping_alt_count, overlapping_alt_best_score, overlapping_alt_best_shift_nt,
            crowding_window_count, crowding_best_score, crowding_score_gap,
            # classic (strict)
            w_Howell_strict, w_s_strict, w_e_strict, c_Howell_strict, c_s_strict, c_e_strict, Peak_Howell_strict
        ])

    return rows
