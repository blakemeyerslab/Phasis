"""Combine per-library and pooled evidence with a data-driven cut-off.

Pooling libraries adds reads to weak loci, but in a dense pooled cluster reads
can also fall into a phase-length register by chance.  The combined evidence
mode therefore treats the two kinds of pooled locus differently:

* **Anchors** are pooled loci that overlap a locus called per library.  They
  are real loci seen in the pooled data and serve as a check.
* **Pooled-only** loci overlap no per-library locus.  They are kept only when
  their strict Howell score (``Peak_Howell_score_strict``) reaches a cut-off;
  the others go to an exploratory tier and are not deleted.

With ``cutoff="auto"`` the cut-off is set by a target-decoy estimate of the
false-discovery rate.  Each pooled-only locus is scored again after its reads
are moved by a random offset of 0 to ``phase - 1`` nt (one offset per distinct
position and strand), which keeps the reads, abundances and footprint of the
cluster but removes any phasing.  The decoy scores show how high noise of the
same density scores by chance, and the cut-off is the lowest strict score at
which the estimated FDR among kept pooled-only loci is at most ``fdr``.  The
share of anchors above the cut-off is reported as a check.

Per-library calls are never filtered.
"""
from __future__ import annotations

import csv
import gzip
import math
import os
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .env import getenv
from . import result_comparison as rc

DEFAULT_FDR = 0.05
DEFAULT_DECOY_REPLICATES = 20
DEFAULT_DECOY_SEED = 20261118
# The auto cut-off never goes below this floor; it is also the cut-off when there
# are too few pooled-only loci for a stable estimate, or no read table for decoys.
# The command line passes -min_Howell_score (default 12.5) here.
DEFAULT_STRICT_FLOOR = 12.5
DEFAULT_AUTO_MIN_LOCI = 20
POOLED_LIBRARY = "ALL_LIBS"

_READ_COLUMNS = ["identifier", "clusterID", "strand", "pos", "len", "abun"]


class EvidenceError(ValueError):
    """Raised when the combined evidence cannot be computed."""


# --------------------------------------------------------------------------
# Cut-off option
# --------------------------------------------------------------------------

def parse_cutoff(value) -> float | str:
    """Return 'auto', 'off' or a non-negative float."""
    text = str(value if value is not None else "auto").strip().lower()
    if text in {"auto", "off"}:
        return text
    try:
        number = float(text)
    except ValueError as exc:
        raise EvidenceError(f"cut-off must be auto, off or a number, not {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"cut-off must be a non-negative number, not {value!r}")
    return number


def auto_min_loci() -> int:
    try:
        value = int(str(getenv("Phasis_POOLED_AUTO_MIN_LOCI") or DEFAULT_AUTO_MIN_LOCI))
    except ValueError:
        value = DEFAULT_AUTO_MIN_LOCI
    return max(1, value)


def decoy_replicates() -> int:
    try:
        value = int(str(getenv("Phasis_POOLED_DECOY_REPLICATES") or DEFAULT_DECOY_REPLICATES))
    except ValueError:
        value = DEFAULT_DECOY_REPLICATES
    return max(1, value)


# --------------------------------------------------------------------------
# Strict Howell score of one cluster, original and decoy
# --------------------------------------------------------------------------

def _pos_abun(strand_mask, pos, length, abun, phase, seq_start, seq_end):
    out: dict[int, float] = {}
    keep = strand_mask & (length == phase) & (pos >= seq_start) & (pos <= seq_end)
    for p, a in zip(pos[keep].tolist(), abun[keep].tolist()):
        p = int(p)
        out[p] = out.get(p, 0.0) + float(a)
    return out


def strict_score_arrays(strand, pos, length, abun, phase: int) -> float:
    """Peak strict Howell score of one cluster, as computed in feature assembly.

    ``strand`` holds 'w'/'c' (or the other encodings feature assembly accepts).
    Returns NaN when the cluster has no reads on either strand.
    """
    from .stages import feature_assembly as fa

    phase = int(phase)
    win_size = fa.WINDOW_MULTIPLIER * phase
    pos = np.asarray(pos, dtype=np.int64)
    length = np.asarray(length, dtype=np.int64)
    abun = np.asarray(abun, dtype=float)
    s = np.char.lower(np.asarray(strand, dtype=str))
    w_mask = np.isin(s, ["w", "+", "watson", "1", "true"])
    c_mask = np.isin(s, ["c", "-", "crick", "0", "false"])
    if pos.size == 0:
        return float("nan")
    seq_start = int(pos.min())
    seq_end = int(pos.max())
    scores = []
    for mask, scan in (
        (w_mask, fa.best_sliding_window_score_forward_strict),
        (c_mask, fa.best_sliding_window_score_reverse_strict),
    ):
        if not mask.any():
            continue
        pa = _pos_abun(mask, pos, length, abun, phase, seq_start, seq_end)
        scores.append(float(scan(pa, phase, win_size, seq_start, seq_end)[0]) if pa else 0.0)
    return max(scores) if scores else float("nan")


def _locus_seed(identifier: str, base_seed: int) -> int:
    return (zlib.crc32(str(identifier).encode("utf-8")) ^ int(base_seed)) & 0xFFFFFFFF


def decoy_scores_arrays(strand, pos, length, abun, phase: int, replicates: int, seed: int) -> list[float]:
    """Strict scores of ``replicates`` decoys: each distinct (strand, position)
    moves by its own random offset of 0 to phase - 1 nt."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, dtype=np.int64)
    strand = np.asarray(strand, dtype=str)
    keys = np.char.add(strand, np.char.mod("%d", pos))
    _, inverse = np.unique(keys, return_inverse=True)
    n_keys = int(inverse.max()) + 1 if inverse.size else 0
    out = []
    for _ in range(int(replicates)):
        offsets = rng.integers(0, int(phase), size=n_keys)
        out.append(strict_score_arrays(strand, pos + offsets[inverse], length, abun, phase))
    return out


def _score_locus(task):
    identifier, clusters, phase, replicates, seed, expected = task
    best = None
    for cluster_id, frame in clusters:
        observed = strict_score_arrays(frame["strand"], frame["pos"], frame["len"], frame["abun"], phase)
        candidate = (abs(observed - expected) if (expected is not None and np.isfinite(expected)) else 0.0, cluster_id, frame, observed)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return identifier, None, float("nan"), []
    _, cluster_id, frame, observed = best
    decoys = decoy_scores_arrays(
        frame["strand"], frame["pos"], frame["len"], frame["abun"], phase, replicates,
        _locus_seed(identifier, seed),
    )
    return identifier, cluster_id, observed, decoys


def load_cluster_reads(path: str | Path, identifiers: Iterable[str], *, chunksize: int = 2_000_000):
    """Read the rows of the given loci from a PHAS_to_detect table, grouped by
    identifier and then by cluster ID."""
    wanted = set(str(i) for i in identifiers)
    found: dict[str, dict[str, list[pd.DataFrame]]] = {}
    if not wanted:
        return {}
    reader = pd.read_csv(path, sep="\t", usecols=_READ_COLUMNS, chunksize=chunksize,
                         dtype={"identifier": str, "clusterID": str, "strand": str})
    for chunk in reader:
        chunk = chunk[chunk["identifier"].isin(wanted)]
        if chunk.empty:
            continue
        for (identifier, cluster_id), frame in chunk.groupby(["identifier", "clusterID"], sort=False):
            found.setdefault(identifier, {}).setdefault(cluster_id, []).append(frame)
    out = {}
    for identifier, clusters in found.items():
        out[identifier] = []
        for cid, parts in clusters.items():
            frame = pd.concat(parts) if len(parts) > 1 else parts[0]
            out[identifier].append((cid, {
                "strand": frame["strand"].to_numpy(dtype=str),
                "pos": frame["pos"].to_numpy(dtype=np.int64),
                "len": frame["len"].to_numpy(dtype=np.int64),
                "abun": frame["abun"].to_numpy(dtype=float),
            }))
    return out


def score_pooled_only_loci(
    reads: Mapping[str, list],
    expected: Mapping[str, float],
    phase: int,
    *,
    replicates: int,
    seed: int = DEFAULT_DECOY_SEED,
    cores: int = 1,
):
    """Observed and decoy strict scores for each locus; run in parallel when cores > 1."""
    tasks = [
        (identifier, reads[identifier], int(phase), int(replicates), int(seed), expected.get(identifier))
        for identifier in sorted(reads)
    ]
    if cores > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from .parallel import _effective_start_method

        ctx = mp.get_context(_effective_start_method())
        with ctx.Pool(processes=min(int(cores), len(tasks))) as pool:
            results = pool.map(_score_locus, tasks, chunksize=max(1, len(tasks) // (int(cores) * 4)))
    else:
        results = [_score_locus(task) for task in tasks]
    return {identifier: (cluster_id, observed, decoys) for identifier, cluster_id, observed, decoys in results}


# --------------------------------------------------------------------------
# Cut-off from target and decoy scores
# --------------------------------------------------------------------------

def qvalues(observed: Sequence[float], decoys: Sequence[float], replicates: int) -> np.ndarray:
    """Target-decoy q-value of each observed score.

    FDR(t) = (decoy scores >= t / replicates) / (observed scores >= t); the
    q-value of a score is the lowest FDR of any cut-off at or below it.
    Missing scores get q = 1.
    """
    obs = np.asarray(observed, dtype=float)
    dec = np.sort(np.asarray([d for d in decoys if np.isfinite(d)], dtype=float))
    finite = np.isfinite(obs)
    q = np.ones(obs.shape, dtype=float)
    if not finite.any():
        return q
    obs_sorted = np.sort(obs[finite])
    thresholds = np.unique(obs_sorted)
    n_obs_ge = obs_sorted.size - np.searchsorted(obs_sorted, thresholds, side="left")
    n_dec_ge = (dec.size - np.searchsorted(dec, thresholds, side="left")) / float(max(1, replicates))
    fdr = np.minimum(1.0, n_dec_ge / n_obs_ge)
    q_at = np.minimum.accumulate(fdr)  # lowest FDR at this cut-off or any lower one
    q[finite] = q_at[np.searchsorted(thresholds, obs[finite])]
    return q


def auto_cutoff(observed: Sequence[float], q: Sequence[float], fdr: float) -> float:
    """Lowest observed score whose q-value is at most ``fdr``; inf when none."""
    obs = np.asarray(observed, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(obs) & (q <= float(fdr))
    return float(obs[ok].min()) if ok.any() else float("inf")


# --------------------------------------------------------------------------
# Bundles
# --------------------------------------------------------------------------

def _filter_bundle(src: Path, dst: Path, phase: int, identifiers: set[str]) -> None:
    """Copy a result bundle keeping only rows of the given loci."""
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in ("calls.tsv", "all_clusters.tsv", "classification_evidence.tsv", "phasiRNAs.tsv"):
        source = src / f"{phase}_{suffix}"
        with source.open("r", encoding="utf-8", newline="") as fin, \
                (dst / f"{phase}_{suffix}").open("w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin, delimiter="\t")
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in reader:
                if str(row.get("identifier", "")) in identifiers:
                    writer.writerow(row)
    with (src / f"{phase}_PHAS.gff").open("r", encoding="utf-8") as fin, \
            (dst / f"{phase}_PHAS.gff").open("w", encoding="utf-8") as fout:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            identifier = rc._gff_identifier(line.rstrip("\r\n").split("\t"), src / f"{phase}_PHAS.gff", line_number)
            if identifier in identifiers:
                fout.write(line if line.endswith("\n") else line + "\n")


def find_phas_to_detect(run_dir: str | Path, phase: int) -> Path | None:
    base = Path(run_dir) / f"concat_{phase}_PHAS_to_detect.tab"
    for candidate in (base.with_name(base.name + ".gz"), base):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class EvidenceResult:
    phase: int
    outdir: Path
    cutoff: float | None
    cutoff_source: str
    summary: dict = field(default_factory=dict)


def _write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "inf" if f > 0 else ""
    return f"{f:.6g}"


def combine_evidence(
    individual_dir: str | Path,
    pooled_dir: str | Path,
    outdir: str | Path,
    *,
    run_dir: str | Path,
    cutoff="auto",
    fdr: float = DEFAULT_FDR,
    replicates: int | None = None,
    seed: int = DEFAULT_DECOY_SEED,
    cores: int = 1,
    phas_to_detect: str | Path | None = None,
    floor: float = DEFAULT_STRICT_FLOOR,
    min_auto_loci: int | None = None,
) -> EvidenceResult:
    """Write the combined result bundle to ``outdir``.

    The bundle holds every per-library locus, every pooled locus that overlaps
    one (as support in the audit tables), and the pooled-only loci that pass the
    cut-off.  Pooled-only loci below it are written to ``<phase>_exploratory/``.
    """
    individual_dir = Path(individual_dir).resolve()
    pooled_dir = Path(pooled_dir).resolve()
    outdir = Path(outdir).resolve()
    cutoff_spec = parse_cutoff(cutoff)
    replicates = int(replicates or decoy_replicates())
    min_auto_loci = int(min_auto_loci or auto_min_loci())
    floor = float(floor)

    individual = rc._load_bundle(individual_dir, "run_a", "individual")
    pooled = rc._load_bundle(pooled_dir, "run_b", "pooled")
    rc._validate_compatible(individual, pooled)
    phase = individual.phase
    overlaps = rc._find_overlaps(individual.loci.values(), pooled.loci.values(), 0)
    anchor_ids = {o.run_b.identifier for o in overlaps}
    pooled_only_ids = set(pooled.loci) - anchor_ids

    strict = {}
    for row in pooled.call_rows:
        try:
            value = float(row.get("Peak_Howell_score_strict", "nan"))
        except ValueError:
            value = float("nan")
        identifier = str(row["identifier"])
        previous = strict.get(identifier, float("nan"))
        if math.isfinite(value) and not (math.isfinite(previous) and previous >= value):
            strict[identifier] = value
        else:
            strict.setdefault(identifier, previous)

    # ---- cut-off ----
    q = {i: float("nan") for i in pooled_only_ids}
    observed_check: dict[str, float] = {}
    decoy_count = 0
    mismatches = 0
    source = ""
    if cutoff_spec == "off":
        cutoff_value = float("-inf")
        source = "off (every pooled-only locus kept)"
    elif cutoff_spec != "auto":
        cutoff_value = float(cutoff_spec)
        source = "set by the user"
    elif not pooled_only_ids:
        cutoff_value = float("nan")
        source = "auto (no pooled-only loci)"
    elif len(pooled_only_ids) < min_auto_loci:
        cutoff_value = floor
        source = f"floor {floor:g} (fewer than {min_auto_loci} pooled-only loci for an auto estimate)"
    else:
        reads_path = Path(phas_to_detect) if phas_to_detect else find_phas_to_detect(run_dir, phase)
        reads = load_cluster_reads(reads_path, pooled_only_ids) if reads_path and reads_path.is_file() else {}
        if not reads:
            cutoff_value = floor
            source = f"floor {floor:g} (no pooled read table found for decoys)"
        else:
            scored = score_pooled_only_loci(reads, strict, phase, replicates=replicates, seed=seed, cores=cores)
            ids = sorted(pooled_only_ids)
            obs = [strict.get(i, float("nan")) for i in ids]
            decoys = [d for i in ids for d in scored.get(i, (None, None, []))[2]]
            decoy_count = len(decoys)
            for i in ids:
                if i in scored:
                    observed_check[i] = scored[i][1]
                    e = strict.get(i)
                    if e is not None and math.isfinite(e) and abs(scored[i][1] - e) > 1e-6 * max(1.0, abs(e)):
                        mismatches += 1
            qs = qvalues(obs, decoys, replicates)
            q = dict(zip(ids, qs.tolist()))
            estimate = auto_cutoff(obs, qs, fdr)
            source = f"auto (target-decoy FDR <= {fdr:g}, {replicates} decoys per locus)"
            if estimate < floor:
                source += f"; estimate {estimate:.6g} raised to the floor {floor:g}"
            cutoff_value = max(estimate, floor)

    def passes(identifier: str) -> bool:
        if not math.isfinite(cutoff_value) and cutoff_value < 0:
            return True
        value = strict.get(identifier, float("nan"))
        return bool(math.isfinite(value) and math.isfinite(cutoff_value) and value >= cutoff_value)

    kept_pooled_only = {i for i in pooled_only_ids if passes(i)}
    exploratory = pooled_only_ids - kept_pooled_only

    # ---- combined bundle ----
    tmp = Path(run_dir).resolve() / f".phasis_evidence_{phase}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        kept_pooled = tmp / f"{phase}_pooled_kept"
        _filter_bundle(pooled_dir, kept_pooled, phase, anchor_ids | kept_pooled_only)
        result = rc.compare_result_directories(
            individual_dir, kept_pooled, outdir,
            label_a="individual", label_b="pooled", prefer="a", overlap_buffer=0, force=True,
        )
        explore_dir = outdir / f"{phase}_exploratory"
        if explore_dir.exists():
            shutil.rmtree(explore_dir)
        _filter_bundle(pooled_dir, explore_dir, phase, exploratory)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- per-locus table and summary ----
    by_pooled: dict[str, list[str]] = {}
    for o in overlaps:
        by_pooled.setdefault(o.run_b.identifier, []).append(o.run_a.identifier)
    rows = []
    for identifier, locus in sorted(pooled.loci.items(), key=lambda kv: rc._locus_key(kv[1])):
        is_anchor = identifier in anchor_ids
        rows.append({
            "identifier": identifier,
            "achr": locus.chromosome,
            "start": locus.start,
            "end": locus.end,
            "category": "anchor" if is_anchor else "pooled_only",
            "Peak_Howell_score_strict": _fmt(strict.get(identifier)),
            "recomputed_strict": _fmt(observed_check.get(identifier)) if not is_anchor else "",
            "q_value": _fmt(q.get(identifier)) if not is_anchor else "",
            "tier": "supports_individual" if is_anchor else ("kept" if identifier in kept_pooled_only else "exploratory"),
            "overlapping_individual_loci": ",".join(sorted(by_pooled.get(identifier, []))),
        })
    _write_rows(outdir / f"{phase}_pooled_evidence.tsv",
                ["identifier", "achr", "start", "end", "category", "Peak_Howell_score_strict",
                 "recomputed_strict", "q_value", "tier", "overlapping_individual_loci"], rows)

    anchor_scores = [strict.get(i, float("nan")) for i in anchor_ids]
    finite_anchor = [s for s in anchor_scores if math.isfinite(s)]
    if math.isfinite(cutoff_value):
        anchors_above = sum(1 for s in finite_anchor if s >= cutoff_value)
    elif cutoff_value < 0:
        anchors_above = len(finite_anchor)
    else:
        anchors_above = 0
    summary = {
        "phase": phase,
        "individual_results": str(individual_dir),
        "pooled_results": str(pooled_dir),
        "individual_loci": len(individual.loci),
        "pooled_loci": len(pooled.loci),
        "anchors": len(anchor_ids),
        "pooled_only_loci": len(pooled_only_ids),
        "strict_cutoff": _fmt(cutoff_value) if not (not math.isfinite(cutoff_value) and cutoff_value < 0) else "off",
        "cutoff_source": source,
        "target_fdr": fdr if cutoff_spec == "auto" else "",
        "floor": floor if cutoff_spec == "auto" else "",
        "decoy_scores": decoy_count,
        "recomputed_score_mismatches": mismatches if cutoff_spec == "auto" else "",
        "pooled_only_kept": len(kept_pooled_only),
        "pooled_only_exploratory": len(exploratory),
        "anchors_at_or_above_cutoff": f"{anchors_above} of {len(finite_anchor)}",
        "combined_loci": result.summary["combined_loci"],
    }
    with (outdir / f"{phase}_evidence_summary.txt").open("w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}\t{value}\n")
    return EvidenceResult(phase, outdir, None if not math.isfinite(cutoff_value) else cutoff_value, source, summary)
