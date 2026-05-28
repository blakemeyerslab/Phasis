from __future__ import annotations

"""
phasis.ids
----------
Lightweight, spawn-safe ID utilities and (Phase II) universal-ID builders.

Goals:
- Provide a single, unambiguous implementation of mergedClusterDict loading.
- Provide a reverse index (raw clusterID -> universalID) cached in:
  - phasis.runtime (preferred single source of truth)
  - this module's globals (legacy compatibility)

Important:
- The reverse map is process-local. On macOS (spawn), each worker may need to
  load/rebuild it; this module keeps it cheap by:
  - loading the persisted {phase}_mergedClusterDict.tab when present
  - caching results in-process
"""

import csv
import functools
import os
import re
import threading
from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import phasis.runtime as rt
from phasis.cache import MemCache, default_memfile_path, phase2_basename, stage_signature
from phasis.parallel import run_parallel_with_progress

# ---------------------------------------------------------------------
# Process-local caches (safe for fork; rebuilt per-process for spawn)
# ---------------------------------------------------------------------

_MERGED_DICT_LOCK = threading.Lock()


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _pick_colname(cols: List[str], candidates: Sequence[str]) -> Optional[str]:
    """Pick the first matching column name, case-insensitive, from candidates."""
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    return None


# ---------------------------------------------------------------------
# Basic load/cache helpers
# ---------------------------------------------------------------------

def _load_simple_tab_dict(path: str) -> Dict[str, List[str]]:
    """Load a key \\t values... tab into {key: [values...]}; tolerate empty lines."""
    out: Dict[str, List[str]] = {}
    with open(path, "r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            key = parts[0].strip()
            vals = [v for v in (p.strip() for p in parts[1:]) if v]
            out[key] = vals if vals else [key]
    return out


def _set_reverse_merged_map(mcd: Dict[str, List[str]]) -> None:
    """Cache clusterID -> universalID reverse map in both module globals and runtime."""
    rev: Dict[str, str] = {}
    for u, members in (mcd or {}).items():
        for cid in members or []:
            s = str(cid).strip()
            if s:
                rev[s] = str(u)

    # module globals (compat for legacy-style access)
    globals()["MERGED_CLUSTER_REVERSE"] = rev
    globals()["mergedClusterReverse"] = rev
    globals()["mergedClusterDict"] = mcd

    # runtime (preferred going forward)
    rt.mergedClusterDict = mcd
    rt.mergedClusterReverse = rev


def ensure_mergedClusterDict(phase: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Light loader:
      1) return cached mergedClusterDict if available (either in runtime or module globals),
      2) else load {phase}_mergedClusterDict.tab (phase2_basename),
      3) else set empty caches and return {}.

    NOTE: We intentionally do NOT build identity maps here (only universal IDs).
    """
    with _MERGED_DICT_LOCK:
        # 1) runtime cache
        mcd = getattr(rt, "mergedClusterDict", None)
        if isinstance(mcd, dict) and mcd:
            rev = getattr(rt, "mergedClusterReverse", None)
            if not isinstance(rev, dict) or not rev:
                _set_reverse_merged_map(mcd)
            return mcd

        # 1b) module global cache
        g = globals()
        if isinstance(g.get("mergedClusterDict"), dict) and g["mergedClusterDict"]:
            mcd = g["mergedClusterDict"]
            _set_reverse_merged_map(mcd)
            return mcd

        # 2) load from disk
        dict_tab = phase2_basename("mergedClusterDict.tab")
        if os.path.isfile(dict_tab):
            try:
                mcd = _load_simple_tab_dict(dict_tab)
                _set_reverse_merged_map(mcd)
                return mcd
            except Exception as e:
                print(f"[WARN] Failed to load {dict_tab}: {e}")

        # 3) empty fallback
        _set_reverse_merged_map({})
        return {}


def _strip_fileprefix_from_id(
    cid: str,
    lib: str | None = None,
    phase: str | int | None = None,
) -> str:
    """
    Remove glued filename/prefix from cluster IDs like:
      ALL_LIBS.21-PHAS.candidate.clustersALL_LIBS-10_120402_10  ->  10_120402_10
      s.S3.1_Pre_1.24-PHAS.candidate.clusters1006_18            ->  1006_18

    Works with/without lib and phase; returns input if no match.
    Leaves universal IDs (chr:start..end) untouched.
    """
    s = str(cid).strip()
    if ":" in s and ".." in s:  # already universal
        return s

    # Resolve phase (prefer explicit, else runtime)
    ph = phase
    if ph is None:
        ph = getattr(rt, "phase", None)

    pats: List[str] = []
    if ph is not None and str(ph).strip():
        p = re.escape(str(ph).strip())
        if lib:
            pats.append(rf"^{re.escape(str(lib))}\.{p}-PHAS\.candidate\.clusters{re.escape(str(lib))}-(.+)$")
        # generic with any lib after 'clusters'
        pats.append(rf"^[^.]+\.{p}-PHAS\.candidate\.clusters[^-]*-(.+)$")
        # common '...clustersXXXX_YY' (no dash) cases
        pats.append(rf".*?\.{p}-PHAS\.candidate\.clusters(.+)$")

    # broad fallback: anything up to '.PHAS.candidate.clusters' then a '-' then the real ID
    pats.append(r"^.+?\.PHAS\.candidate\.clusters[^-]*-(.+)$")
    # broad fallback without dash
    pats.append(r"^.+?\.PHAS\.candidate\.clusters(.+)$")

    for pat in pats:
        m = re.match(pat, s)
        if m:
            return m.group(1).strip()
    return s


def normalize_cluster_id_for_lookup(cid: str, phase: Optional[int] = None) -> str:
    """
    Normalize a clusterID so it can be used as a key in caches/lookups.

    Rules:
    - If it's already coordinate-style ("chr:start..end"), return as-is.
    - Else, strip the PHAS candidate file prefix (when present).
    """
    s = str(cid).strip()
    if (":" in s) and (".." in s):
        return s
    return _strip_fileprefix_from_id(s, phase=phase)


def _ensure_reverse_index() -> Dict[str, str]:
    """Return the cached reverse index (clusterID -> universalID), building if needed."""
    rev = getattr(rt, "mergedClusterReverse", None)
    if isinstance(rev, dict) and rev:
        globals()["MERGED_CLUSTER_REVERSE"] = rev
        globals()["mergedClusterReverse"] = rev
        return rev

    g_rev = globals().get("MERGED_CLUSTER_REVERSE")
    if isinstance(g_rev, dict) and g_rev:
        rt.mergedClusterReverse = g_rev
        globals()["mergedClusterReverse"] = g_rev
        return g_rev

    # If neither exists, try loading dict and building
    ensure_mergedClusterDict()
    rev = getattr(rt, "mergedClusterReverse", None)
    if isinstance(rev, dict):
        return rev
    return {}


def getUniversalID(clusterID: str) -> Optional[str]:
    """
    Map a raw clusterID to its universal ID using mergedClusterDict.

    Strategy:
    1) Ensure mergedClusterDict + reverse map are available (light loader).
    2) Try exact key in mergedClusterDict (already universal).
    3) Try prefix-stripped key in mergedClusterDict (already universal).
    4) Try reverse map lookup on raw and stripped.
    5) Return None if not resolvable.
    """
    ensure_mergedClusterDict()
    rev = _ensure_reverse_index()

    cid_raw = str(clusterID).strip()
    cid_norm = normalize_cluster_id_for_lookup(cid_raw)

    mcd = getattr(rt, "mergedClusterDict", None)
    if not isinstance(mcd, dict) or not mcd:
        mcd = globals().get("mergedClusterDict", {}) or {}

    # Already universal?
    if cid_raw in mcd:
        return cid_raw
    if cid_norm in mcd:
        return cid_norm

    # Reverse map
    if cid_raw in rev:
        return rev[cid_raw]
    if cid_norm in rev:
        return rev[cid_norm]

    return None


# ---------------------------------------------------------------------
# Step 6.4.2 — builder logic: universal-ID dict creation (concat + non-concat)
# ---------------------------------------------------------------------

def group_loci_by_chromosome_for_parallel(loci_by_chr):
    """Convert pandas groupby('chr') object into list-of-lists per chromosome."""
    out: List[List[List[Any]]] = []
    for _, group in loci_by_chr:
        if isinstance(group, pd.DataFrame):
            cols = ["name", "pval", "chr", "start", "end"]
            if all(c in group.columns for c in cols):
                out.append(group[cols].values.tolist())
            else:
                out.append(group.values.tolist())
        else:
            out.append(list(group))
    return out


def merge_loci_pairs_by_chromosome(loci_group, *, clustbuffer: int = 0):
    """
    loci_group: list of [name, pval, chr, start, end] (all same chr).
    Returns list of [A,B] pairs; adds [X,'singleLibOccurrence'] only for IDs
    that never paired with anyone.
    """
    buf = int(clustbuffer)

    # 1) normalize to compact records: [name, start, end]
    L: List[List[Any]] = []
    for name, _pval, _chr, start, end in loci_group:
        L.append([str(name).strip(), int(start), int(end)])

    n = len(L)
    if n <= 1:
        return [[L[0][0], "singleLibOccurrence"]] if n == 1 else []

    # 2) O(n) monotonicity check on (start, end)
    maybe_sorted = True
    ps, pe = L[0][1], L[0][2]
    for i in range(1, n):
        cs, ce = L[i][1], L[i][2]
        if cs < ps or (cs == ps and ce < pe):
            maybe_sorted = False
            break
        ps, pe = cs, ce

    # 3) Only sort if needed (by start, then end) — no lambda
    if not maybe_sorted:
        L.sort(key=itemgetter(1, 2))

    # 4) pair generation with early break and true-singleton tracking
    pairs: List[List[str]] = []
    paired_ids: set[str] = set()
    all_ids: set[str] = {r[0] for r in L}  # set avoids duplicates

    for i in range(n):
        aname, astart, aend = L[i]
        for j in range(i + 1, n):
            bname, bstart, bend = L[j]

            # since L is sorted by start, once bstart is beyond a’s buffered end we can stop
            if bstart > aend + buf:
                break

            # buffered overlap test
            if (bend >= astart - buf) and (bstart <= aend + buf):
                pairs.append([aname, bname])
                paired_ids.add(aname)
                paired_ids.add(bname)

    # 5) emit singletons that never appeared in any pair
    for name in all_ids - paired_ids:
        pairs.append([name, "singleLibOccurrence"])

    return pairs


def _norm_cluster_id_for_merge(x: Any, *, phase: str | int) -> Optional[str]:
    """Normalize cluster id for merging: stringify, strip, drop empty/sentinel, strip prefix."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "singleLibOccurrence":
        return None
    return _strip_fileprefix_from_id(s, phase=phase)


def _dsu_find(parent: Dict[str, str], x: str) -> str:
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = _dsu_find(parent, parent[x])
    return parent[x]


def _dsu_union(parent: Dict[str, str], rank: Dict[str, int], a: str, b: str) -> None:
    ra = _dsu_find(parent, a)
    rb = _dsu_find(parent, b)
    if ra == rb:
        return
    ra_rank = rank.get(ra, 0)
    rb_rank = rank.get(rb, 0)
    if ra_rank < rb_rank:
        parent[ra] = rb
    elif ra_rank > rb_rank:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] = ra_rank + 1


def assemble_candidate_clusters_parametric(
    mergedClusters: Sequence[Sequence[Any]],
    allClusters_df: pd.DataFrame,
    phase: str,
) -> Dict[str, List[str]]:
    """
    Build merged components from overlap pairs and assign universal IDs as chr:start..end
    using the loci-table coordinate map (candidate.loci_table.tab).
    """
    # --- build clusterID -> (chr, start, end) from loci table ---------------
    cid2coord: Dict[str, Tuple[str, int, int]] = {}
    try:
        loci_path = phase2_basename("candidate.loci_table.tab")
        ldf = pd.read_csv(loci_path, sep="\t", engine="python")

        # normalize headers from stage output
        ldf = ldf.rename(
            columns={
                "Cluster": "name",
                "value1": "pval",
                "chromosome": "chr",
                "Start": "start",
                "End": "end",
            }
        )

        if not all(c in ldf.columns for c in ("name", "chr", "start", "end")):
            raise ValueError(f"candidate.loci_table.tab missing required columns: {list(ldf.columns)}")

        ldf["chr"] = ldf["chr"].astype(str)
        ldf["start"] = pd.to_numeric(ldf["start"], errors="coerce").astype("Int64")
        ldf["end"] = pd.to_numeric(ldf["end"], errors="coerce").astype("Int64")
        ldf = ldf.dropna(subset=["start", "end"])

        for name, achr, s, e in ldf[["name", "chr", "start", "end"]].itertuples(index=False, name=None):
            k = _norm_cluster_id_for_merge(name, phase=phase)
            if k is not None:
                cid2coord[k] = (str(achr), int(s), int(e))
    except Exception as e:
        # stay permissive; we still produce groups, but keys may fallback
        print(f"[WARN] assemble_candidate_clusters_parametric(): failed to build coord map: {e}")

    # Union-Find / Disjoint Set ---------------------------------------------
    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    # 1) add edges from mergedClusters
    if mergedClusters:
        for pair in mergedClusters:
            if not pair:
                continue
            a = _norm_cluster_id_for_merge(pair[0], phase=phase) if len(pair) >= 1 else None
            b = _norm_cluster_id_for_merge(pair[1], phase=phase) if len(pair) >= 2 else None
            if a:
                parent.setdefault(a, a)
            if b:
                parent.setdefault(b, b)
            if a and b:
                _dsu_union(parent, rank, a, b)

    # 2) add singletons from allClusters_df
    try:
        if allClusters_df is not None and "clusterID" in allClusters_df.columns:
            for raw in allClusters_df["clusterID"].astype(str):
                cid = _norm_cluster_id_for_merge(raw, phase=phase)
                if cid and cid not in parent:
                    parent[cid] = cid
    except Exception:
        pass

    if not parent:
        return {}

    # 3) connected components
    comps: Dict[str, set[str]] = {}
    for node in list(parent.keys()):
        root = _dsu_find(parent, node)
        comps.setdefault(root, set()).add(node)

    # 4) produce universal IDs as COORDINATE KEYS
    mergedClusterDict: Dict[str, List[str]] = {}
    for members in comps.values():
        if not members:
            continue
        m_sorted = sorted(members)

        coords = [cid2coord.get(m) for m in m_sorted if m in cid2coord]
        coords = [c for c in coords if c is not None]
        if coords:
            achr = coords[0][0]
            smin = min(s for _, s, _ in coords)
            emax = max(e for _, _, e in coords)
            key = f"{achr}:{smin}..{emax}"
        else:
            key = m_sorted[0]  # fallback (should be rare)

        mergedClusterDict[key] = m_sorted

    return mergedClusterDict



# ---------------------------------------------------------------------
# Legacy compatibility: moved from legacy.py (cluster assembly + ID helpers)
# ---------------------------------------------------------------------

copy_mergedClusterDict_global: Dict[str, List[str]] = {}

def preprocess_single_library_clusters(mergedClusters: Sequence[Sequence[Any]]) -> Dict[str, List[str]]:
    """
    Legacy-compatible helper:
    build one-member merged clusters for single-library mode.
    """
    mergedClusterDict: Dict[str, List[str]] = defaultdict(list)
    for acount, apair in enumerate(mergedClusters):
        aPname = apair[0]
        mergedClusterDict[f"cluster_{acount}"].append(aPname)
    return dict(mergedClusterDict)


def assemble_clusters_by_chromosome(mergedClusters_chr: Sequence[Sequence[Any]]) -> Dict[str, List[str]]:
    """
    Build merged sets per chromosome from pair list produced by merge_loci_pairs_by_chromosome.
    Handles single-library case inside.
    """
    mergedClusterDict: Dict[str, List[str]] = defaultdict(list)
    assigned = set()
    acount = 0

    for apair in mergedClusters_chr:
        aPname, bPname = apair[0], apair[1]

        if aPname == bPname:
            print("single-library chromosome: collapsing pairs")
            return preprocess_single_library_clusters(mergedClusters_chr)

        if bPname != "singleLibOccurrence":
            if acount == 0 or aPname not in assigned:
                key = f"cluster_{acount}"
                mergedClusterDict[key].extend([aPname, bPname])
                assigned.update([aPname, bPname])

                for cPname, dPname in mergedClusters_chr:
                    if (aPname, bPname) == (cPname, dPname):
                        continue
                    if aPname == cPname and dPname not in assigned and dPname != "singleLibOccurrence":
                        mergedClusterDict[key].append(dPname)
                        assigned.add(dPname)
                acount += 1

        elif aPname not in assigned:
            mergedClusterDict[f"cluster_{acount}"].append(aPname)
            assigned.add(aPname)
            acount += 1
        else:
            acount += 1

    return dict(mergedClusterDict)


def assign_final_ids_by_chromosome(chromosome_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Rename merged cluster IDs into genomic span keys (chrom:start..stop) per chromosome.
    Uses the module-global mapping prepared by assign_final_cluster_ids().
    """
    reNamesClusterDict: Dict[str, set[str]] = defaultdict(set)

    chromosome_df = chromosome_df.copy()
    chromosome_df["pos"] = pd.to_numeric(chromosome_df["pos"], errors="coerce")
    chromosome_df["clusterID"] = chromosome_df["clusterID"].astype(str).str.strip()

    cluster_info = chromosome_df.groupby("clusterID").agg(
        start=("pos", "min"),
        stop=("pos", "max"),
    ).to_dict("index")

    chromosome = chromosome_df["chromosome"].iloc[0]

    for entry in copy_mergedClusterDict_global:
        cluster_ids = copy_mergedClusterDict_global[entry]
        valid_cluster_ids = [cid for cid in cluster_ids if cid in cluster_info]
        if not valid_cluster_ids:
            continue

        if len(valid_cluster_ids) == 1:
            info = cluster_info[valid_cluster_ids[0]]
            start, stop = info["start"], info["stop"]
        else:
            starts = [cluster_info[cid]["start"] for cid in valid_cluster_ids]
            stops = [cluster_info[cid]["stop"] for cid in valid_cluster_ids]
            start, stop = min(starts), max(stops)

        reNamesClusterDict[f"{chromosome}:{start}..{stop}"].update(valid_cluster_ids)

    return {k: list(v) for k, v in reNamesClusterDict.items()}


def assign_final_cluster_ids(mergedClusterDict: Dict[str, List[str]], allClusters: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parallel per-chromosome remapping of mergedClusterDict -> genomic span IDs,
    then flattened back into one dict with unique clusterIDs per span.
    """
    chromosome_groups = [df for _, df in allClusters.groupby("chromosome")]

    copy_merged = {k.strip(): [cid.strip() for cid in v] for k, v in mergedClusterDict.items()}

    global copy_mergedClusterDict_global
    copy_mergedClusterDict_global = copy_merged

    results = run_parallel_with_progress(
        assign_final_ids_by_chromosome,
        chromosome_groups,
        desc="Assign final cluster IDs",
        min_chunk=1,
        unit="lib-chr",
    )

    flattened: Dict[str, List[str]] = defaultdict(list)
    assigned = set()
    for rd in results:
        for k, vals in rd.items():
            new_vals = [v for v in vals if v not in assigned]
            assigned.update(new_vals)
            flattened[k].extend(new_vals)

    copy_mergedClusterDict_global = {}
    return dict(flattened)


def _identity_dict_from_tsv_firstcol(
    path: str,
    id_col: Sequence[str] = ("Cluster", "cluster", "clusterID", "name", "cID"),
) -> Dict[str, List[str]]:
    """
    Build an identity mapping {cid: [cid]} from the first usable ID column in a TSV.
    """
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None, engine="python")
        if df.shape[1] >= 1:
            df.columns = ["Cluster"] + [f"col{i}" for i in range(2, df.shape[1] + 1)]

    col = next((c for c in id_col if c in df.columns), df.columns[0])
    ids = df[col].astype(str).tolist()
    return {cid: [cid] for cid in ids}

def merge_candidate_clusters_parametric(
    loci_df: pd.DataFrame,
    allClusters_df: pd.DataFrame,
    phase: str,
    memFile: str | None,
    *,
    clustbuffer: int | None = None,
) -> Dict[str, List[str]]:
    """
    Non-concat universal-ID builder:
    - finds overlaps across libraries per chromosome (buffered by clustbuffer)
    - assembles components
    - assigns universal IDs as chr:start..end using loci_table coords
    - writes:
        {phase}_merged_clusters.tab
        {phase}_mergedClusterDict.tab
      with md5 caching in memFile.
    """
    memFile_local = memFile or getattr(rt, "memFile", None) or default_memfile_path()
    buf = int(getattr(rt, "clustbuffer", 0) or 0) if clustbuffer is None else int(clustbuffer)

    cache = MemCache.load(memFile_local)

    merged_pairs_path = phase2_basename("merged_clusters.tab")
    merged_dict_path = phase2_basename("mergedClusterDict.tab")

    # ---- cache check ------------------------------------------------------
    # Cache is valid only when BOTH output hashes and the input signature match.
    # This prevents stale Phase II results when new libraries are added.
    loci_tab_path = phase2_basename("candidate.loci_table.tab")
    proc_tab_path = phase2_basename("processed_clusters.tab")
    input_sig = stage_signature(
        files=[loci_tab_path, proc_tab_path],
        params={
            "phase": phase,
            "clustbuffer": buf,
            "concat_libs": bool(getattr(rt, "concat_libs", False)),
        },
    )

    if cache.hit("MERGED_CLUSTERS", merged_pairs_path, input_sig) and cache.hit(
        "MERGED_DICT", merged_dict_path, input_sig
    ):
        print("Outputs up-to-date (hash+sig match). Skipping merge computation.")
        return _load_simple_tab_dict(merged_dict_path)

    # ---- build overlap pairs per chromosome (parallel if multi-lib) --------
    mergedClusters: List[List[Any]] = []
    try:
        nlibs = int(allClusters_df["alib"].nunique())
    except Exception:
        nlibs = 1

    loci_df_sorted = loci_df.sort_values(["chr", "start", "end"], ascending=True)

    if nlibs >= 2:
        groups_as_lists = group_loci_by_chromosome_for_parallel(loci_df_sorted.groupby("chr"))
        worker_fn = functools.partial(merge_loci_pairs_by_chromosome, clustbuffer=buf)

        results = run_parallel_with_progress(
            worker_fn,
            groups_as_lists,
            desc="Find overlapping loci across libs",
            min_chunk=1,
            unit="lib-chr",
        )
        mergedClusters = [pair for sub in results for pair in sub]
    else:
        for aname, _apval, _achr, _astart, _aend in loci_df_sorted[["name", "pval", "chr", "start", "end"]].itertuples(index=False, name=None):
            mergedClusters.append([aname, aname])

    # write pairs (tab)
    with open(merged_pairs_path, "w") as fh:
        for pair in mergedClusters:
            fh.write("\t".join(map(str, [e for e in pair if str(e).strip()])) + "\n")

    # assemble dictionary and assign final IDs (universal IDs)
    mcd = assemble_candidate_clusters_parametric(mergedClusters, allClusters_df, phase)

    # write dict (key \\t values…)
    with open(merged_dict_path, "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        for key, values in mcd.items():
            wr.writerow([key] + values)

    # update cache
    cache.record("MERGED_CLUSTERS", merged_pairs_path, input_sig)
    cache.record("MERGED_DICT", merged_dict_path, input_sig)

    return mcd

def ensure_mergedClusterDict_always(
    *,
    concat_libs: bool,
    phase: str,
    merged_out_path: str,
    loci_table_df: pd.DataFrame,
    allClusters_df: pd.DataFrame,
    memFile: str | None,
) -> Dict[str, List[str]]:
    """
    Always return a dict universal_id -> [member clusterIDs] and ensure the tab exists.

    - In concat mode we derive universal IDs (chr:start..end) from {phase}_merged_candidates.tab.
    - In non-concat mode we do the real cross-lib merge via merge_candidate_clusters_parametric().
    - Always caches reverse map (clusterID -> universalID) for later lookups.

    Cache policy:
    - If mergedClusterDict.tab exists, reuse it ONLY when BOTH its md5 AND its input signature
      match the current Phase II inputs.
    """
    dict_tab = phase2_basename("mergedClusterDict.tab")

    memFile_local = memFile or getattr(rt, "memFile", None) or default_memfile_path()
    cache = MemCache.load(memFile_local)

    try:
        buf = int(getattr(rt, "clustbuffer", 0) or 0)
    except Exception:
        buf = 0

    # Build the *input* signature that should invalidate mergedClusterDict.tab.
    if concat_libs:
        sig_files = [merged_out_path]
    else:
        sig_files = [phase2_basename("candidate.loci_table.tab"), phase2_basename("processed_clusters.tab")]

    dict_input_sig = stage_signature(
        files=sig_files,
        params={"phase": phase, "clustbuffer": buf, "concat_libs": bool(concat_libs)},
    )

    # Reuse existing outputs only if hash + signature match.
    # In non-concat mode, also require merged_clusters.tab to exist and be fresh,
    # so users don't end up with a valid dict but missing pair table.
    pairs_tab = phase2_basename("merged_clusters.tab")
    reuse_ok = cache.hit("MERGED_DICT", dict_tab, dict_input_sig)
    if not concat_libs:
        reuse_ok = reuse_ok and cache.hit("MERGED_CLUSTERS", pairs_tab, dict_input_sig)

    if reuse_ok:
        try:
            mcd = _load_simple_tab_dict(dict_tab)
            _set_reverse_merged_map(mcd)
            return mcd
        except Exception as e:
            print(f"[WARN] Failed to load {dict_tab}: {e}. Recomputing…")

    # ---- recompute ----
    if concat_libs:
        if not os.path.isfile(merged_out_path):
            raise FileNotFoundError(f"Missing merged candidates TSV: {merged_out_path}")

        mdf = pd.read_csv(merged_out_path, sep="	", engine="python")

        cid_col = _pick_colname(list(mdf.columns), ("Cluster", "clusterID", "name", "cID"))
        chr_col = _pick_colname(list(mdf.columns), ("chromosome", "chr"))
        start_col = _pick_colname(list(mdf.columns), ("Start", "start", "begin"))
        end_col = _pick_colname(list(mdf.columns), ("End", "end", "stop"))

        if not all([cid_col, chr_col, start_col, end_col]):
            raise ValueError(f"{merged_out_path} lacks required columns (have: {list(mdf.columns)})")

        mcd_tmp: Dict[str, List[str]] = defaultdict(list)
        for cid, achr, s, e in mdf[[cid_col, chr_col, start_col, end_col]].itertuples(index=False, name=None):
            try:
                u = f"{str(achr)}:{int(s)}..{int(e)}"
            except Exception:
                u = f"{str(achr)}:{str(s)}..{str(e)}"
            mcd_tmp[u].append(str(cid).strip())

        # dedup + sort for stability
        mcd = {u: sorted(set(vs)) for u, vs in mcd_tmp.items()}

        # persist tab
        with open(dict_tab, "w", newline="") as fh:
            wr = csv.writer(fh, delimiter="	")
            for key, values in mcd.items():
                wr.writerow([key] + values)

        cache.record("MERGED_DICT", dict_tab, dict_input_sig)

        _set_reverse_merged_map(mcd)
        return mcd

    # Non-concat: perform true cross-lib merge (also writes mergedClusterDict.tab)
    mcd = merge_candidate_clusters_parametric(
        loci_table_df,
        allClusters_df,
        phase,
        memFile_local,
    )

    _set_reverse_merged_map(mcd)
    return mcd
