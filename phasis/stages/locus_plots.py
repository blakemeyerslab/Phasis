from __future__ import annotations

import os
import re
import shutil
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.stages import feature_assembly as st_feat

MAX_LOCUS_PLOT_WORKERS = 10
GUIDE_CYCLES = 10
PHASE_PANEL_COLORS = {
    19: "#CC3299",
    20: "#CFB53B",
    21: "#6DC8F8",
    22: "#008000",
    23: "#7F00FF",
    24: "#FF7F00",
    25: "#1EF000",
}
READ_LEN_COLORS = {
    21: "#6DC8F8",
    22: "#008000",
    23: "#7F00FF",
    24: "#FF7F00",
}
LIGHT_GREY = "#B3B3B3"
DARK_GREY = "#5A5A5A"
HPSP_RED = "#D62828"
CENTER_LINE_COLOR = "#8C8C8C"
OTHER_TRACE_ALPHA = 0.35
NON_PHASE_GREY = "#9A9A9A"
EXTENDED_GUIDE_COLOR = "#000000"


def _join_outdir(dirpath: str | None, name: str) -> str:
    if not dirpath:
        return name
    return dirpath + name if dirpath.endswith("/") else dirpath + "/" + name


def _sanitize_plot_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    clean = re.sub(r"_+", "_", clean).strip("._")
    return clean or "PHAS_locus"


def _parse_identifier_interval(identifier_text: str):
    text = str(identifier_text).strip()
    match = re.search(r":\s*(\d+)\.\.(\d+)\s*$", text)
    if not match:
        match = re.search(r"(\d+)\.\.(\d+)\s*$", text)
    if not match:
        return None, None

    start_value = int(match.group(1))
    end_value = int(match.group(2))
    if start_value <= end_value:
        return start_value, end_value
    return end_value, start_value


def _phase_color_hex(phase_value: int) -> str:
    return PHASE_PANEL_COLORS.get(int(phase_value), "#4C78A8")


def _read_length_color_hex(length_value) -> str:
    try:
        ilen = int(length_value)
    except Exception:
        return DARK_GREY
    if ilen <= 20:
        return LIGHT_GREY
    if ilen >= 25:
        return DARK_GREY
    return READ_LEN_COLORS.get(ilen, DARK_GREY)


def _find_hpsp_trace_row(trace_rows) -> dict | None:
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


def _is_forward_strand(strand_code: str) -> bool:
    return str(strand_code).strip().lower().startswith("w")


def _classify_register_relation(anchor_position: int, register_origin: int | None, phase_value: int):
    if register_origin is None:
        return "other", None

    anchor_local = int(anchor_position)
    phase_local = int(phase_value)
    delta = anchor_local - int(register_origin)
    remainder = delta % phase_local

    if remainder == 0:
        return "exact", anchor_local
    if remainder == 1:
        return "offset", anchor_local - 1
    if remainder == phase_local - 1:
        return "offset", anchor_local + 1
    return "other", None


def _build_scored_register_positions(hpsp_row: dict | None, phase_value: int, strand_code: str):
    if not hpsp_row:
        return [], None

    best_register = hpsp_row.get("best_register")
    if best_register is None:
        return [], None

    phase_local = int(phase_value)
    reg_value = int(best_register)
    if _is_forward_strand(strand_code):
        register_origin = int(hpsp_row["window_start"]) + reg_value
        positions = [register_origin + cycle * phase_local for cycle in range(GUIDE_CYCLES)]
    else:
        register_origin = int(hpsp_row["window_end"]) - reg_value
        positions = [register_origin - cycle * phase_local for cycle in range(GUIDE_CYCLES)]

    return positions, register_origin


def _collect_extended_register_positions(
    trace_rows,
    register_origin: int | None,
    base_positions,
    phase_value: int,
):
    if register_origin is None or not base_positions:
        return []

    base_set = {int(pos) for pos in base_positions}
    extended_positions = set()
    for row in trace_rows or []:
        try:
            anchor_position = int(row["anchor_position"])
        except Exception:
            continue
        relation, expected_position = _classify_register_relation(
            anchor_position,
            register_origin,
            phase_value,
        )
        if relation not in {"exact", "offset"} or expected_position is None:
            continue
        if int(expected_position) in base_set:
            continue
        extended_positions.add(int(expected_position))

    return sorted(extended_positions)


def _build_score_exact_guide_specs(hpsp_row: dict | None, phase_value: int, trace_rows, strand_code: str) -> list[dict]:
    base_positions, register_origin = _build_scored_register_positions(hpsp_row, phase_value, strand_code)
    hpsp_position = None if register_origin is None else int(register_origin)
    extended_positions = _collect_extended_register_positions(
        trace_rows,
        register_origin,
        base_positions,
        phase_value,
    )

    specs_by_pos = {}
    for pos in base_positions:
        specs_by_pos[int(pos)] = {"pos": int(pos), "extended": False, "is_hpsp": False}
    for pos in extended_positions:
        specs_by_pos.setdefault(int(pos), {"pos": int(pos), "extended": True, "is_hpsp": False})

    if hpsp_position is not None:
        spec = specs_by_pos.setdefault(
            hpsp_position,
            {"pos": hpsp_position, "extended": False, "is_hpsp": False},
        )
        spec["is_hpsp"] = True

    return [specs_by_pos[pos] for pos in sorted(specs_by_pos)]


def _axis_abs_formatter(value, _pos):
    if abs(value) < 1e-9:
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(abs(round(value))))
    return f"{abs(value):.1f}"


def _draw_centerline(ax) -> None:
    ax.axhline(0.0, color=CENTER_LINE_COLOR, linewidth=0.9, linestyle="-", zorder=1)
    ax.yaxis.set_major_formatter(FuncFormatter(_axis_abs_formatter))


def _draw_strand_guides(ax, guide_specs, color: str, *, upper_half: bool) -> None:
    if not guide_specs:
        return
    if upper_half:
        ymin, ymax = 0.5, 1.0
    else:
        ymin, ymax = 0.0, 0.5
    for spec in guide_specs:
        pos = float(spec["pos"])
        is_hpsp = bool(spec.get("is_hpsp", False))
        is_extended = bool(spec.get("extended", False))
        line_color = HPSP_RED if is_hpsp else (EXTENDED_GUIDE_COLOR if is_extended else color)
        alpha_value = 0.60 if is_hpsp else (0.28 if is_extended else 0.38)
        linestyle = "--" if not is_extended else ":"
        linewidth = 0.95 if is_hpsp else (0.8 if is_extended else 0.9)
        ax.axvline(
            pos,
            ymin=ymin,
            ymax=ymax,
            color=line_color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha_value,
            zorder=0,
        )


def _build_abundance_rows(cluster_df: pd.DataFrame):
    rows = []
    if cluster_df.empty:
        return rows

    local = cluster_df.copy()
    local["pos"] = pd.to_numeric(local["pos"], errors="coerce")
    local["abun"] = pd.to_numeric(local["abun"], errors="coerce").fillna(0.0)
    local["len"] = pd.to_numeric(local["len"], errors="coerce")
    local["hits"] = pd.to_numeric(local.get("hits", pd.Series([1] * len(local))), errors="coerce").fillna(1)
    local = local.dropna(subset=["pos"])

    for row in local.itertuples(index=False):
        strand_text = str(getattr(row, "strand", "")).strip().lower()
        if strand_text in {"c", "-", "crick", "0", "false"}:
            y_value = -float(getattr(row, "abun", 0.0))
            strand_group = "c"
        else:
            y_value = float(getattr(row, "abun", 0.0))
            strand_group = "w"
        rows.append(
            {
                "x": float(getattr(row, "pos")),
                "y": y_value,
                "edgecolor": _read_length_color_hex(getattr(row, "len", None)),
                "facecolor": _read_length_color_hex(getattr(row, "len", None))
                if float(getattr(row, "hits", 1) or 1) <= 1.0
                else "none",
                "marker": "o" if float(getattr(row, "hits", 1) or 1) <= 1.0 else "s",
                "strand": strand_group,
            }
        )
    return rows


def _build_howell_rows(trace_rows, phase_value: int, strand_code: str):
    rows = []
    hpsp_row = _find_hpsp_trace_row(trace_rows)
    _base_positions, register_origin = _build_scored_register_positions(hpsp_row, phase_value, strand_code)
    hpsp_position = None if register_origin is None else int(register_origin)
    phase_color = _phase_color_hex(phase_value)
    direction = 1.0 if _is_forward_strand(strand_code) else -1.0

    for row in trace_rows or []:
        anchor_position = int(row["anchor_position"])
        plot_position = float(anchor_position)
        score_value = float(row.get("score", 0.0) or 0.0)
        relation, _expected_position = _classify_register_relation(
            anchor_position,
            register_origin,
            int(phase_value),
        )
        edgecolor = HPSP_RED if hpsp_row is not None and anchor_position == int(hpsp_row["anchor_position"]) else phase_color
        facecolor = "none"
        alpha_value = 1.0
        linewidth = 0.9
        size = 22

        if hpsp_row is not None and anchor_position == int(hpsp_row["anchor_position"]):
            plot_position = float(hpsp_position)
            facecolor = HPSP_RED
            alpha_value = 1.0
            linewidth = 0.9
            size = 26
        elif relation == "exact":
            facecolor = phase_color
            alpha_value = 1.0
        elif relation == "offset":
            facecolor = "none"
            alpha_value = 1.0
        else:
            edgecolor = NON_PHASE_GREY
            facecolor = "none"
            alpha_value = 0.7
            linewidth = 0.7
            size = 18

        rows.append(
            {
                "x": plot_position,
                "y": direction * score_value,
                "edgecolor": edgecolor,
                "facecolor": facecolor,
                "alpha": alpha_value,
                "linewidth": linewidth,
                "size": size,
                "strand": str(strand_code).lower()[0] if str(strand_code) else "w",
                "phase_relation": relation,
                "is_hpsp": bool(hpsp_row is not None and anchor_position == int(hpsp_row["anchor_position"])),
            }
        )
    return rows, hpsp_position, hpsp_row


def _build_plot_legend_groups(phase_value: int):
    phase_color = _phase_color_hex(phase_value)
    return {
        "read_lengths": [
            Line2D([0], [0], marker="D", color="none", markeredgecolor=LIGHT_GREY, markerfacecolor=LIGHT_GREY, markersize=5, label="<=20 nt"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=READ_LEN_COLORS[21], markerfacecolor=READ_LEN_COLORS[21], markersize=5, label="21 nt"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=READ_LEN_COLORS[22], markerfacecolor=READ_LEN_COLORS[22], markersize=5, label="22 nt"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=READ_LEN_COLORS[23], markerfacecolor=READ_LEN_COLORS[23], markersize=5, label="23 nt"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=READ_LEN_COLORS[24], markerfacecolor=READ_LEN_COLORS[24], markersize=5, label="24 nt"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor=DARK_GREY, markerfacecolor=DARK_GREY, markersize=5, label=">=25 nt"),
        ],
        "abundance": [
            Line2D([0], [0], marker="D", color="none", markeredgecolor="#444444", markerfacecolor="#444444", markersize=6, label="Uni-mapper read"),
            Line2D([0], [0], marker="D", color="none", markeredgecolor="#444444", markerfacecolor="none", markersize=6, label="Multi-mapper read"),
        ],
        "howell": [
            Line2D([0], [0], marker="o", color="none", markeredgecolor=phase_color, markerfacecolor=phase_color, markersize=6, label="Exact in-phase"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor=phase_color, markerfacecolor="none", markersize=6, label="Offset (+/-1)"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor=NON_PHASE_GREY, markerfacecolor="none", markersize=6, label="Out-of-phase"),
        ],
        "hpsp": [
            Line2D([0], [0], marker="o", color="none", markeredgecolor=HPSP_RED, markerfacecolor=HPSP_RED, markersize=6, label="Highest phasing score position / register anchor (HPSP)"),
        ],
    }


def _add_grouped_legends(fig, phase_value: int) -> None:
    legend_groups = _build_plot_legend_groups(phase_value)
    legend_common = {
        "frameon": False,
        "fontsize": 8,
        "handletextpad": 0.5,
        "columnspacing": 1.0,
        "borderaxespad": 0.0,
    }

    legend_read = fig.legend(
        handles=legend_groups["read_lengths"],
        loc="upper left",
        bbox_to_anchor=(0.08, 0.915),
        ncol=2,
        **legend_common,
    )
    legend_abundance = fig.legend(
        handles=legend_groups["abundance"],
        loc="upper left",
        bbox_to_anchor=(0.40, 0.915),
        ncol=1,
        **legend_common,
    )
    legend_howell = fig.legend(
        handles=legend_groups["howell"],
        loc="upper left",
        bbox_to_anchor=(0.55, 0.915),
        ncol=1,
        **legend_common,
    )
    legend_hpsp = fig.legend(
        handles=legend_groups["hpsp"],
        loc="upper left",
        bbox_to_anchor=(0.70, 0.915),
        ncol=1,
        **legend_common,
    )

    fig.add_artist(legend_read)
    fig.add_artist(legend_abundance)
    fig.add_artist(legend_howell)
    fig.add_artist(legend_hpsp)


def _write_placeholder_plot(plot_path: str, title_text: str, message_text: str) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.axis("off")
    ax.text(0.02, 0.75, title_text, fontsize=12, fontweight="bold", ha="left")
    ax.text(0.02, 0.45, message_text, fontsize=10, ha="left")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _write_single_locus_plot(task: dict) -> str:
    plot_path = task["plot_path"]
    title_text = task["title_text"]
    identifier_text = task.get("identifier_text", "")
    phase_value = int(task["phase"])
    cluster_rows = task["cluster_rows"]

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    if not cluster_rows:
        return _write_placeholder_plot(
            plot_path,
            title_text,
            "No cluster rows were available for this PHAS call.",
        )

    cluster_df = pd.DataFrame(cluster_rows)
    if cluster_df.empty:
        return _write_placeholder_plot(
            plot_path,
            title_text,
            "The PHAS locus data were empty after reconstruction.",
        )

    trace = st_feat.enumerate_relaxed_howell_trace(cluster_df, phase=phase_value)
    w_trace = trace.get("w", [])
    c_trace = trace.get("c", [])
    if not w_trace and not c_trace:
        return _write_placeholder_plot(
            plot_path,
            title_text,
            "No phase-length reads were available to compute the relaxed Howell trace.",
        )

    abundance_rows = _build_abundance_rows(cluster_df)
    howell_rows_w, hpsp_w, hpsp_row_w = _build_howell_rows(w_trace, phase_value, "w")
    howell_rows_c, hpsp_c, hpsp_row_c = _build_howell_rows(c_trace, phase_value, "c")
    howell_rows = howell_rows_w + howell_rows_c
    guide_specs_w = _build_score_exact_guide_specs(hpsp_row_w, phase_value, w_trace, "w")
    guide_specs_c = _build_score_exact_guide_specs(hpsp_row_c, phase_value, c_trace, "c")

    cluster_positions = pd.to_numeric(cluster_df["pos"], errors="coerce").dropna()
    if not cluster_positions.empty:
        xmin = float(cluster_positions.min())
        xmax = float(cluster_positions.max())
    else:
        x_values = []
        if abundance_rows:
            x_values.extend(row["x"] for row in abundance_rows)
        if howell_rows:
            x_values.extend(row["x"] for row in howell_rows)
        x_values = [float(x) for x in x_values if x is not None]
        xmin, xmax = _parse_identifier_interval(identifier_text)
        if xmin is None or xmax is None:
            xmin = min(x_values) if x_values else 0.0
            xmax = max(x_values) if x_values else float(phase_value)

    span = max(float(xmax) - float(xmin), float(phase_value))
    xpad = max(float(phase_value) * 4.0, span * 0.06)

    max_abun = max([abs(row["y"]) for row in abundance_rows], default=1.0)
    max_score = max([abs(row["y"]) for row in howell_rows], default=1.0)
    abun_ylim = max(1.0, max_abun * 1.15)
    score_ylim = max(1.0, max_score * 1.20)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    axes[0].set_facecolor("#F1F1F1")
    axes[1].set_facecolor("#F1F1F1")

    fig.suptitle(title_text, fontsize=12, y=0.98)
    _add_grouped_legends(fig, phase_value)

    _draw_centerline(axes[0])
    _draw_centerline(axes[1])
    _draw_strand_guides(axes[0], guide_specs_w, _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[0], guide_specs_c, _phase_color_hex(phase_value), upper_half=False)
    _draw_strand_guides(axes[1], guide_specs_w, _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[1], guide_specs_c, _phase_color_hex(phase_value), upper_half=False)

    for row in abundance_rows:
        axes[0].scatter(
            row["x"],
            row["y"],
            s=18,
            marker="D",
            facecolors=row["facecolor"],
            edgecolors=row["edgecolor"],
            linewidths=0.7,
            alpha=0.95,
            zorder=3,
        )

    for row in howell_rows:
        if row.get("is_hpsp", False):
            continue
        axes[1].scatter(
            row["x"],
            row["y"],
            s=row["size"],
            marker="o",
            facecolors=row["facecolor"],
            edgecolors=row["edgecolor"],
            linewidths=row["linewidth"],
            alpha=row["alpha"],
            zorder=3,
        )

    for row in howell_rows:
        if not row.get("is_hpsp", False):
            continue
        axes[1].scatter(
            row["x"],
            row["y"],
            s=row["size"],
            marker="o",
            facecolors=row["facecolor"],
            edgecolors=row["edgecolor"],
            linewidths=row["linewidth"],
            alpha=row["alpha"],
            zorder=6,
        )

    axes[0].set_ylabel("Abundance", fontsize=10)
    axes[1].set_ylabel("Howell score", fontsize=10)
    axes[1].set_xlabel("Genomic position", fontsize=10)

    axes[0].set_xlim(float(xmin) - xpad, float(xmax) + xpad)
    axes[0].set_ylim(-abun_ylim, abun_ylim)
    axes[1].set_ylim(-score_ylim, score_ylim)

    axes[0].text(0.01, 0.92, "+", transform=axes[0].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[0].text(0.01, 0.08, "-", transform=axes[0].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[1].text(0.01, 0.92, "+", transform=axes[1].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[1].text(0.01, 0.08, "-", transform=axes[1].transAxes, fontsize=9, fontweight="bold", color="#444444")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, linestyle="-", zorder=0)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.82, hspace=0.08)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_individual_phas_locus_plots(
    method_name: str,
    labeled_features: pd.DataFrame,
    clusters_data: pd.DataFrame,
    *,
    job_outdir: str | None = None,
    job_phase: str | int | None = None,
) -> None:
    phase_value = int(job_phase) if job_phase is not None else int(getattr(rt, "phase", 21))
    plot_dir_name = f"{phase_value}_{method_name}_PHAS_locus_plots"
    plot_dir = _join_outdir(job_outdir, plot_dir_name)

    phas_calls = labeled_features.loc[labeled_features["label"] == "PHAS"].copy()
    if phas_calls.empty:
        shutil.rmtree(plot_dir, ignore_errors=True)
        print("[INFO] No PHAS calls were detected; skipping individual locus plots.")
        return

    print("#### Plotting individual PHAS loci ######")

    call_cols = ["identifier", "alib", "cID"]
    missing_cols = [col for col in call_cols if col not in phas_calls.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for locus plots: {missing_cols}")

    cluster_key_cols = ["clusterID", "identifier", "alib"]
    missing_cluster_cols = [col for col in cluster_key_cols if col not in clusters_data.columns]
    if missing_cluster_cols:
        raise ValueError(f"clusters_data missing required columns for locus plots: {missing_cluster_cols}")

    shutil.rmtree(plot_dir, ignore_errors=True)
    os.makedirs(plot_dir, exist_ok=True)

    grouped_by_cid = {}
    grouped_by_identifier = {}
    for (cid_value, alib_value), subdf in clusters_data.groupby(["clusterID", "alib"], sort=False):
        grouped_by_cid[(str(cid_value).strip(), str(alib_value).strip())] = subdf.copy()
    for (identifier_value, alib_value), subdf in clusters_data.groupby(["identifier", "alib"], sort=False):
        grouped_by_identifier[(str(identifier_value).strip(), str(alib_value).strip())] = subdf.copy()

    phas_calls = phas_calls.drop_duplicates(subset=call_cols, keep="first").reset_index(drop=True)
    tasks = []
    for row in phas_calls.itertuples(index=False):
        identifier_value = str(getattr(row, "identifier")).strip()
        alib_value = str(getattr(row, "alib")).strip()
        cid_value = str(getattr(row, "cID")).strip()

        cluster_df = grouped_by_cid.get((cid_value, alib_value))
        if cluster_df is None:
            cluster_df = grouped_by_identifier.get((identifier_value, alib_value))

        filename = f"{_sanitize_plot_name(alib_value)}__{_sanitize_plot_name(identifier_value)}.png"
        title_text = f"{alib_value} | {identifier_value} | {phase_value}-$\\it{{PHAS}}$"

        tasks.append(
            {
                "plot_path": os.path.join(plot_dir, filename),
                "title_text": title_text,
                "identifier_text": identifier_value,
                "phase": int(phase_value),
                "cluster_rows": [] if cluster_df is None else cluster_df.to_dict("records"),
            }
        )

    if not tasks:
        shutil.rmtree(plot_dir, ignore_errors=True)
        print("[INFO] No PHAS locus plot tasks were produced.")
        return

    worker_cap = min(MAX_LOCUS_PLOT_WORKERS, max(1, cpu_count()), len(tasks))
    run_parallel_with_progress(
        _write_single_locus_plot,
        tasks,
        desc="Writing PHAS locus plots",
        min_chunk=1,
        batch_factor=1.0,
        unit="file",
        kind="plot",
        initial_worker_cap=worker_cap,
        max_worker_cap=worker_cap,
        adaptive_recovery=False,
    )
    print(f"[INFO] Wrote {len(tasks)} individual PHAS locus plot(s) to {plot_dir}")
