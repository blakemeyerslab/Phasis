from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PANEL_PLOTS = [
    "sTR_dcl5_1_2.0__9_11575724..11579329.png",
    "sTR_dcl5_1_2.0__9_11573639..11574985.png",
    "sTR_dcl5_1_2.0__9_11606220..11607628.png",
    "sTR_dcl5_1_2.0__9_11586194..11587735.png",
    "sTR_dcl5_1_2.0__9_11566730..11569330.png",
    "sTR_dcl5_1_2.0__9_22890460..22893596.png",
    "sTP_dcl5_1_2.0__9_11575724..11579329.png",
    "sTP_dcl5_1_2.0__9_11606220..11607628.png",
    "sTR_dcl5_1_2.0__9_22786269..22787664.png",
    "sTR_dcl5_1_2.0__9_23133599..23134528.png",
    "sTR_dcl5_1_2.0__9_134289543..134290617.png",
    "sTR_dcl5_1_2.0__9_35736665..35738897.png",
    "sTR_dcl5_1_2.0__9_22894495..22895200.png",
    "sTP_dcl5_1_2.0__9_11566730..11569330.png",
    "sTP_dcl5_1_2.0__9_35736665..35738897.png",
    "sTP_dcl5_1_2.0__9_11586194..11587735.png",
    "sTP_dcl5_1_2.0__9_11573639..11574985.png",
    "W23_2.0_2__9_23053495..23055794.png",
    "sTP_dcl5_1_2.0__9_23133599..23134528.png",
    "sTP_dcl5_1_2.0__9_22890460..22893596.png",
    "sTP_dcl5_1_2.0__9_22894495..22895200.png",
    "W23_2.0_2__9_23133599..23134528.png",
]


def _parse_plot_stub(stub: str) -> tuple[str, str]:
    stem = stub[:-4] if stub.endswith(".png") else stub
    lib, ident = stem.split("__", 1)
    ident = ident.replace("_", ":", 1)
    return lib, ident


DEFAULT_PANEL = [_parse_plot_stub(item) for item in DEFAULT_PANEL_PLOTS]


def _load_result_tables(result_dir: Path, phase: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc = pd.read_csv(result_dir / f"{phase}_KNN_classification_qc.tsv", sep="\t")
    all_clusters = pd.read_csv(result_dir / f"{phase}_KNN_all_clusters.tsv", sep="\t")
    phas = pd.read_csv(result_dir / f"{phase}_KNN_phasiRNAs.tsv", sep="\t")
    phas_like = pd.read_csv(result_dir / f"{phase}_KNN_PHAS_like_phasiRNAs.tsv", sep="\t")
    exports = pd.concat([phas, phas_like], ignore_index=True)
    return qc, all_clusters, exports


def _load_trace_table(result_dir: Path, phase: int) -> pd.DataFrame:
    trace_path = result_dir / f"{phase}_main_partner_trace.tsv"
    if not trace_path.exists():
        return pd.DataFrame()
    return pd.read_csv(trace_path, sep="\t")


def _winner_strand(row: pd.Series) -> str:
    w_score = row.get("w_Howell_score")
    c_score = row.get("c_Howell_score")
    try:
        w_value = float(w_score)
    except Exception:
        w_value = float("-inf")
    try:
        c_value = float(c_score)
    except Exception:
        c_value = float("-inf")
    return "w" if w_value > c_value else "c"


def _diagnosis(record: dict) -> str:
    if int(record.get("main_partner_rows", 0) or 0) > 0:
        return "accepted_main_partner"
    if int(record.get("overlap_rows", 0) or 0) > 0 and bool(record.get("has_cross_strand_export")):
        return "opposite_strand_diverted_to_secondary"
    if bool(record.get("has_cross_strand_export")):
        return "cross_strand_export_without_partner_role"
    if bool(record.get("both_strands_have_scores")):
        return "both_strands_scored_but_partner_not_accepted"
    return "one_sided_scaffold"


def _best_trace_candidate(trace_rows: pd.DataFrame) -> pd.Series | None:
    if trace_rows.empty:
        return None
    ranked = trace_rows.copy()
    if "canonical_compatible" not in ranked.columns:
        ranked["canonical_compatible"] = 0
    else:
        ranked["canonical_compatible"] = (
            ranked["canonical_compatible"].fillna(False).astype(bool).astype(int)
        )
    if "candidate_tier" in ranked.columns:
        tier_order = {"canonical": 0, "fallback_noncanonical": 1, "same_strand_extension": 2}
        ranked["_candidate_tier_rank"] = ranked["candidate_tier"].map(tier_order).fillna(9)
    else:
        ranked["_candidate_tier_rank"] = 9
    for col in [
        "candidate_peak_score",
        "bridge_support_ratio",
        "shared_cycles",
        "supported_position_count",
        "exact_support_count",
    ]:
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked = ranked.sort_values(
        [
            "_candidate_tier_rank",
            "canonical_compatible",
            "candidate_peak_score",
            "bridge_support_ratio",
            "shared_cycles",
            "supported_position_count",
            "exact_support_count",
        ],
        ascending=[True, False, False, False, False, False, False],
        na_position="last",
    )
    return ranked.iloc[0] if not ranked.empty else None


def _augment_record_from_trace(record: dict, trace_rows: pd.DataFrame) -> dict:
    if trace_rows.empty:
        record["first_failing_gate"] = pd.NA
        record["best_candidate_source"] = pd.NA
        record["best_candidate_shift_nt"] = pd.NA
        record["best_candidate_support_ratio"] = pd.NA
        record["best_candidate_exact_support"] = pd.NA
        record["best_candidate_shared_cycles"] = pd.NA
        record["control_like_except_one_threshold"] = False
        return record

    accepted = trace_rows[trace_rows["final_route"].astype(str) == "main_partner"]
    diverted = trace_rows[
        trace_rows["first_reject_reason"].astype(str) == "accepted_then_diverted_to_secondary"
    ]
    rejected = trace_rows[trace_rows["final_route"].astype(str) == "rejected"]
    best_candidate = _best_trace_candidate(rejected if not rejected.empty else trace_rows)

    if not accepted.empty:
        first_gate = "accepted_main_partner"
    elif not diverted.empty:
        first_gate = "accepted_then_diverted_to_secondary"
    elif not rejected.empty:
        reason_counts = (
            rejected["first_reject_reason"]
            .dropna()
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        first_gate = reason_counts.index[0] if not reason_counts.empty else pd.NA
    else:
        first_gate = pd.NA

    record["first_failing_gate"] = first_gate
    selected_candidate = _best_trace_candidate(accepted if not accepted.empty else trace_rows)
    record["canonical_candidate_exists"] = bool(
        "canonical_compatible" in trace_rows.columns
        and trace_rows["canonical_compatible"].fillna(False).astype(bool).any()
    )
    record["selected_partner_tier"] = (
        pd.NA if selected_candidate is None else selected_candidate.get("candidate_tier")
    )
    record["canonical_partner_selected"] = bool(
        selected_candidate is not None and bool(selected_candidate.get("canonical_compatible"))
    )
    record["selected_partner_raw_shift_nt"] = (
        pd.NA if selected_candidate is None else selected_candidate.get("observed_shift_nt")
    )
    record["selected_partner_normalized_shift_nt"] = (
        pd.NA if selected_candidate is None else selected_candidate.get("normalized_shift_nt")
    )
    record["best_candidate_source"] = pd.NA if best_candidate is None else best_candidate.get("candidate_source")
    record["best_candidate_shift_nt"] = pd.NA if best_candidate is None else best_candidate.get("observed_shift_nt")
    record["best_candidate_normalized_shift_nt"] = (
        pd.NA if best_candidate is None else best_candidate.get("normalized_shift_nt")
    )
    record["best_candidate_support_ratio"] = (
        pd.NA if best_candidate is None else best_candidate.get("bridge_support_ratio")
    )
    record["best_candidate_exact_support"] = (
        pd.NA if best_candidate is None else best_candidate.get("exact_support_count")
    )
    record["best_candidate_shared_cycles"] = (
        pd.NA if best_candidate is None else best_candidate.get("shared_cycles")
    )
    record["control_like_except_one_threshold"] = bool(
        best_candidate is not None
        and str(best_candidate.get("first_reject_reason")) in {"support_ratio_below_threshold", "exact_support_below_threshold"}
        and bool(best_candidate.get("duplex_orientation_ok"))
    )
    return record


def _build_record(
    lib: str,
    identifier: str,
    *,
    qc: pd.DataFrame,
    all_clusters: pd.DataFrame,
    exports: pd.DataFrame,
    trace: pd.DataFrame,
    panel_group: str,
) -> dict | None:
    qc_row = qc[(qc["identifier"] == identifier) & (qc["alib"] == lib)]
    cluster_row = all_clusters[(all_clusters["identifier"] == identifier) & (all_clusters["alib"] == lib)]
    export_rows = exports[(exports["identifier"] == identifier) & (exports["alib"] == lib)]
    if qc_row.empty or cluster_row.empty:
        return None

    q = qc_row.iloc[0]
    c = cluster_row.iloc[0]
    roles = sorted(export_rows["window_unit_role"].dropna().astype(str).unique().tolist()) if not export_rows.empty else []
    strands = sorted(export_rows["strand"].dropna().astype(str).unique().tolist()) if not export_rows.empty else []
    record = {
        "panel_group": panel_group,
        "alib": lib,
        "identifier": identifier,
        "final_class": q.get("final_class"),
        "qc_reason": q.get("qc_reason"),
        "origin_class": q.get("Howell_origin_class"),
        "winner_strand_guess": _winner_strand(c),
        "w_Howell_score": c.get("w_Howell_score"),
        "c_Howell_score": c.get("c_Howell_score"),
        "Peak_Howell_score": q.get("Peak_Howell_score"),
        "Howell_exact_support_score": q.get("Howell_exact_support_score"),
        "Howell_overlapping_alt_count": q.get("Howell_overlapping_alt_count"),
        "Howell_additional_peak_count": q.get("Howell_additional_peak_count"),
        "Howell_crowding_window_count": q.get("Howell_crowding_window_count"),
        "main_hpsp_rows": int((export_rows["window_unit_role"] == "main_hpsp").sum()) if not export_rows.empty else 0,
        "main_partner_rows": int((export_rows["window_unit_role"] == "main_partner").sum()) if not export_rows.empty else 0,
        "main_extension_rows": int((export_rows["window_unit_role"] == "main_extension").sum()) if not export_rows.empty else 0,
        "overlap_rows": int((export_rows["window_unit_role"] == "overlapping_alternative").sum()) if not export_rows.empty else 0,
        "roles": ",".join(roles),
        "exported_strands": ",".join(strands),
        "has_cross_strand_export": len(set(strands)) > 1,
        "both_strands_have_scores": pd.notna(c.get("w_Howell_score")) and pd.notna(c.get("c_Howell_score")),
    }
    record["diagnosis"] = _diagnosis(record)
    if not trace.empty:
        trace_rows = trace[(trace["identifier"] == identifier) & (trace["alib"] == lib)]
        record = _augment_record_from_trace(record, trace_rows)
    else:
        record = _augment_record_from_trace(record, pd.DataFrame())
    return record


def _select_positive_controls(
    qc: pd.DataFrame,
    all_clusters: pd.DataFrame,
    exports: pd.DataFrame,
    *,
    panel_keys: set[tuple[str, str]],
    count: int,
) -> list[tuple[str, str]]:
    grouped = (
        exports.groupby(["identifier", "alib"])["window_unit_role"]
        .apply(lambda series: set(map(str, series.dropna())))
        .reset_index(name="roles")
    )
    positives = grouped[grouped["roles"].apply(lambda roles: "main_partner" in roles)]
    if positives.empty:
        return []
    merged = positives.merge(
        qc[["identifier", "alib", "Peak_Howell_score", "Howell_exact_support_score"]],
        on=["identifier", "alib"],
        how="left",
    )
    merged = merged.sort_values(
        ["Howell_exact_support_score", "Peak_Howell_score", "identifier", "alib"],
        ascending=[False, False, True, True],
    )
    selected = []
    for row in merged.itertuples(index=False):
        key = (str(row.alib), str(row.identifier))
        if key in panel_keys:
            continue
        selected.append(key)
        if len(selected) >= int(count):
            break
    return selected


def build_debug_panel(result_dir: Path, *, phase: int = 21, positive_controls: int = 6) -> pd.DataFrame:
    qc, all_clusters, exports = _load_result_tables(result_dir, phase)
    trace = _load_trace_table(result_dir, phase)
    panel_keys = set(DEFAULT_PANEL)
    records = []
    for lib, identifier in DEFAULT_PANEL:
        record = _build_record(
            lib,
            identifier,
            qc=qc,
            all_clusters=all_clusters,
            exports=exports,
            trace=trace,
            panel_group="target_panel",
        )
        if record is not None:
            records.append(record)

    for lib, identifier in _select_positive_controls(
        qc,
        all_clusters,
        exports,
        panel_keys=panel_keys,
        count=positive_controls,
    ):
        record = _build_record(
            lib,
            identifier,
            qc=qc,
            all_clusters=all_clusters,
            exports=exports,
            trace=trace,
            panel_group="positive_control",
        )
        if record is not None:
            records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["panel_group", "main_partner_rows", "overlap_rows", "Howell_exact_support_score", "Peak_Howell_score", "alib", "identifier"],
        ascending=[True, False, False, False, False, True, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a fixed 21-PHAS debugging panel plus a few positive main_partner controls."
    )
    parser.add_argument("result_dir", help="Path to a Phasis result directory such as 2.6_full_21_results_wwww")
    parser.add_argument("--phase", type=int, default=21, help="Phase length to inspect (default: 21)")
    parser.add_argument(
        "--positive-controls",
        type=int,
        default=6,
        help="Number of additional loci with accepted main_partner rows to append (default: 6)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional TSV output path. Defaults to <result_dir>/<phase>_main_partner_debug_panel.tsv",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    output_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else result_dir / f"{int(args.phase)}_main_partner_debug_panel.tsv"
    )

    frame = build_debug_panel(
        result_dir,
        phase=int(args.phase),
        positive_controls=int(args.positive_controls),
    )
    frame.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote: {output_path}")
    if frame.empty:
        print("No rows found.")
        return
    print()
    print(frame.to_string(index=False))
    print()
    print("Summary:")
    print(frame.groupby(["panel_group", "diagnosis"]).size().rename("count").to_string())


if __name__ == "__main__":
    main()
