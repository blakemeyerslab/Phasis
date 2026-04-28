from __future__ import annotations

import colorsys
import os
import re
import shutil
import tempfile
import textwrap
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.stages import feature_assembly as st_feat

MAX_LOCUS_PLOT_WORKERS = 10
REMOTE_DIRECT_LOCUS_PLOT_WORKER_CAP = 4
GUIDE_CYCLES = 10
MAX_COLORED_ALTERNATIVE_CANDIDATES = 6
PLOT_STAGING_CHOICES = frozenset({"auto", "local", "direct"})
LOCUS_PLOT_MODE_CHOICES = frozenset({"clean", "debug"})
REMOTE_FILESYSTEM_TYPES = frozenset(
    {
        "beegfs",
        "ceph",
        "ceph-fuse",
        "fuse.ceph",
        "fuse.glusterfs",
        "fuse.quobyte",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "panfs",
        "quobyte",
    }
)
REMOTE_PATH_PREFIXES = (
    "/beegfs",
    "/ceph",
    "/gpfs",
    "/lustre",
    "/net",
    "/nfs",
    "/panfs",
    "/quobyte",
)
SCHEDULER_ENV_VARS = (
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "JOB_ID",
    "NSLOTS",
)
PHASIRNA_EXPORT_COLUMNS = [
    "identifier",
    "cID",
    "alib",
    "phase",
    "window_unit_id",
    "window_unit_role",
    "window_unit_rank",
    "window_unit_shift_nt",
    "strand",
    "observed_pos",
    "expected_register_pos",
    "register_class",
    "abun",
    "tag_seq",
    "hits",
]
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
NON_PHASE_GREY = "#9A9A9A"
EXTENDED_GUIDE_COLOR = "#000000"
DETACHABLE_STRIP_BG = "#FAFAFA"
DETACHABLE_SEPARATOR_COLOR = "#9A9A9A"
DETACHABLE_SEPARATOR_STYLE = (0, (1.2, 2.2))
ALTERNATIVE_CATEGORY_LABELS = {
    "other_local_peak": "Other local peaks",
    "overlapping_alternative": "Overlapping alternative candidates",
}
ALTERNATIVE_CATEGORY_PRIORITY = {
    "overlapping_alternative": 0,
    "other_local_peak": 1,
}
ORIGIN_CLASS_LABELS = {
    "insufficient_exact_support": "Insufficient exact support",
    "unique_origin": "Unique origin",
    "coherent_extension": "Coherent extension",
    "ambiguous_origin": "Ambiguous origin",
    "mixed_extension_and_ambiguity": "Mixed extension + ambiguity",
}
LOCUS_LAYOUT = {
    "figsize": (13.2, 6.8),
    "main_left": 0.08,
    "main_right": 0.80,
    "strip_left": 0.835,
    "strip_right": 0.98,
    "abun_bottom": 0.50,
    "abun_height": 0.28,
    "howell_bottom": 0.11,
    "howell_height": 0.28,
    "strip_bottom": 0.13,
    "strip_top": 0.80,
    "separator_x": 0.818,
    "separator_bottom": 0.085,
    "separator_top": 0.895,
    "legend_y": 0.915,
    "title_y": 0.975,
}


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


def _adjust_phase_family_color(
    base_hex: str,
    *,
    hue_shift: float = 0.0,
    light_delta: float = 0.0,
    sat_scale: float = 1.0,
) -> str:
    base_local = str(base_hex or "#4C78A8")
    base_local = base_local.lstrip("#")
    if len(base_local) != 6:
        return f"#{base_local}" if base_local.startswith("#") else f"#{base_local}"

    r_value = int(base_local[0:2], 16) / 255.0
    g_value = int(base_local[2:4], 16) / 255.0
    b_value = int(base_local[4:6], 16) / 255.0
    hue, lightness, saturation = colorsys.rgb_to_hls(r_value, g_value, b_value)
    hue = (hue + float(hue_shift)) % 1.0
    lightness = min(max(lightness + float(light_delta), 0.18), 0.82)
    saturation = min(max(saturation * float(sat_scale), 0.25), 1.0)
    r_out, g_out, b_out = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r_out * 255.0)),
        int(round(g_out * 255.0)),
        int(round(b_out * 255.0)),
    )


def _alternative_color_series(phase_value: int, category: str, *, count: int = MAX_COLORED_ALTERNATIVE_CANDIDATES) -> list[str]:
    base_color = _phase_color_hex(phase_value)
    recipes = [
        (0.020, -0.010, 1.04),
        (0.016, 0.040, 0.98),
        (0.012, 0.090, 0.94),
        (0.008, 0.135, 0.90),
        (0.004, 0.185, 0.86),
        (0.000, 0.235, 0.82),
    ]
    colors = []
    for idx in range(max(int(count), 0)):
        hue_shift, light_delta, sat_scale = recipes[idx % len(recipes)]
        colors.append(
            _adjust_phase_family_color(
                base_color,
                hue_shift=hue_shift,
                light_delta=light_delta,
                sat_scale=sat_scale,
            )
        )
    return colors


def _main_unit_color(phase_value: int) -> str:
    return _adjust_phase_family_color(
        _phase_color_hex(phase_value),
        hue_shift=0.028,
        light_delta=-0.075,
        sat_scale=1.10,
    )


def _alternative_category_label(category: str) -> str:
    return ALTERNATIVE_CATEGORY_LABELS.get(str(category or "").strip(), "Alternative candidate")


def _format_shift_nt_text(value) -> str:
    try:
        shift_value = int(round(float(value)))
    except Exception:
        return "NA"
    if shift_value == 0:
        return "0 nt"
    if shift_value > 0:
        return f"+{shift_value} nt"
    return f"{shift_value} nt"


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


def _normalize_strand_code(strand_code: str) -> str:
    text = str(strand_code).strip().lower()
    if text in {"c", "-", "crick", "0", "false"}:
        return "c"
    return "w"


def _format_phas_text(text_value: str) -> str:
    text_local = str(text_value or "").strip()
    if text_local == "PHAS":
        return r"$\it{PHAS}$"
    if text_local == "PHAS-like":
        return r"$\it{PHAS}$-like"
    if text_local == "non-PHAS":
        return r"non-$\it{PHAS}$"
    text_local = text_local.replace("PHAS-like", "__PHAS_LIKE__")
    text_local = text_local.replace("non-PHAS", "__NON_PHAS__")
    text_local = text_local.replace("PHAS", r"$\it{PHAS}$")
    text_local = text_local.replace("__PHAS_LIKE__", r"$\it{PHAS}$-like")
    text_local = text_local.replace("__NON_PHAS__", r"non-$\it{PHAS}$")
    return text_local


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
    return _normalize_strand_code(strand_code) == "w"


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


def _normalize_locus_plot_mode(value) -> str:
    mode = str(value or "clean").strip().lower()
    if mode not in LOCUS_PLOT_MODE_CHOICES:
        return "clean"
    return mode


def _current_locus_plot_mode() -> str:
    return _normalize_locus_plot_mode(getattr(rt, "locus_plot_mode", "clean"))


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


def _build_unit_member_guide_specs(member: dict, phase_value: int, color: str) -> list[dict]:
    peak_row = member.get("peak_row")
    strand_code = member.get("strand", "w")
    trace_rows = member.get("rows", [])
    guide_specs = _build_score_exact_guide_specs(peak_row, phase_value, trace_rows, strand_code)
    for spec in guide_specs:
        spec["color"] = str(color)
        spec["strand"] = _normalize_strand_code(strand_code)
        # The HPSP dot stays red in the lower panel, but the main-unit guide
        # itself should remain within the phase-family color.
        spec["is_hpsp"] = False
    return guide_specs


def _build_main_unit_guide_specs(main_unit: dict | None, phase_value: int) -> dict:
    guide_specs = {"w": [], "c": []}
    if not main_unit:
        return guide_specs
    color = _main_unit_color(phase_value)
    seen = set()
    for member in list(main_unit.get("members") or []):
        for spec in _build_unit_member_guide_specs(member, phase_value, color):
            key = (
                _normalize_strand_code(spec.get("strand")),
                round(float(spec.get("pos", 0.0)), 6),
                bool(spec.get("extended", False)),
                str(color),
            )
            if key in seen:
                continue
            seen.add(key)
            guide_specs[_normalize_strand_code(spec.get("strand"))].append(spec)
    return guide_specs


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
        line_color = HPSP_RED if is_hpsp else spec.get("color", (EXTENDED_GUIDE_COLOR if is_extended else color))
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


def _build_colored_candidate_guide_specs(candidate: dict, phase_value: int) -> list[dict]:
    peak_row = candidate.get("peak_row")
    strand_code = candidate.get("strand", "w")
    trace_rows = candidate.get("rows", [])
    color = str(candidate.get("color", _phase_color_hex(phase_value)))
    guide_specs = _build_score_exact_guide_specs(peak_row, phase_value, trace_rows, strand_code)
    for spec in guide_specs:
        spec["color"] = color
        spec["strand"] = _normalize_strand_code(strand_code)
        spec["is_hpsp"] = False
    return guide_specs


def _candidate_unit_members(candidate: dict) -> list[dict]:
    members = list(candidate.get("members") or [])
    if members:
        return [dict(member) for member in members]
    return [dict(candidate)]


def _build_colored_unit_guide_specs(candidate: dict, phase_value: int) -> list[dict]:
    color = str(candidate.get("color", _phase_color_hex(phase_value)))
    seen = set()
    guide_specs = []
    for member in _candidate_unit_members(candidate):
        member["color"] = color
        for spec in _build_colored_candidate_guide_specs(member, phase_value):
            key = (
                _normalize_strand_code(member.get("strand", "w")),
                round(float(spec.get("pos", 0.0)), 6),
                bool(spec.get("is_hpsp", False)),
                bool(spec.get("extended", False)),
                color,
            )
            if key in seen:
                continue
            seen.add(key)
            guide_specs.append(spec)
    return guide_specs


def _build_candidate_exact_overlay_rows(candidate: dict, phase_value: int) -> list[dict]:
    register_origin = candidate.get("register_origin")
    if register_origin is None:
        return []

    rows = []
    seen = set()
    strand_code = str(candidate.get("strand", "w"))
    color = str(candidate.get("color", _phase_color_hex(phase_value)))
    direction = 1.0 if _is_forward_strand(strand_code) else -1.0
    for row in candidate.get("rows", []) or []:
        try:
            anchor_position = int(row.get("anchor_position"))
            score_value = float(row.get("score", 0.0) or 0.0)
        except Exception:
            continue
        relation, _expected_position = _classify_register_relation(
            anchor_position,
            int(register_origin),
            int(phase_value),
        )
        if relation != "exact":
            continue
        key = (strand_code, anchor_position, round(score_value, 6), color)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "x": float(anchor_position),
                "y": direction * score_value,
                "edgecolor": color,
                "facecolor": color,
                "alpha": 0.98,
                "linewidth": 0.8,
                "size": 24,
                "category": candidate.get("category"),
            }
        )
    return rows


def _build_candidate_unit_exact_overlay_rows(candidate: dict, phase_value: int) -> list[dict]:
    color = str(candidate.get("color", _phase_color_hex(phase_value)))
    seen = set()
    rows = []
    for member in _candidate_unit_members(candidate):
        member["color"] = color
        for row in _build_candidate_exact_overlay_rows(member, phase_value):
            key = (
                round(float(row.get("x", 0.0)), 6),
                round(float(row.get("y", 0.0)), 6),
                str(row.get("edgecolor")),
                str(candidate.get("category", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            row["category"] = candidate.get("category")
            rows.append(row)
    return rows


def _build_abundance_rows(cluster_df: pd.DataFrame):
    rows = []
    if cluster_df.empty:
        return rows

    local = cluster_df.copy()
    local["pos"] = pd.to_numeric(local["pos"], errors="coerce")
    local["abun"] = pd.to_numeric(local["abun"], errors="coerce").fillna(0.0)
    local["len"] = pd.to_numeric(local["len"], errors="coerce")
    local["hits"] = pd.to_numeric(local.get("hits", pd.Series(np.nan, index=local.index)), errors="coerce")
    local = local.dropna(subset=["pos"])

    for row in local.itertuples(index=False):
        strand_group = _normalize_strand_code(getattr(row, "strand", ""))
        y_value = float(getattr(row, "abun", 0.0))
        if strand_group == "c":
            y_value = -y_value
        hits_value = getattr(row, "hits", np.nan)
        is_multi = False if pd.isna(hits_value) else float(hits_value) > 1.0
        rows.append(
            {
                "x": float(getattr(row, "pos")),
                "y": y_value,
                "edgecolor": _read_length_color_hex(getattr(row, "len", None)),
                "facecolor": "none" if is_multi else _read_length_color_hex(getattr(row, "len", None)),
                "marker": "D",
                "strand": strand_group,
            }
        )
    return rows


def _build_howell_rows(
    trace_rows,
    phase_value: int,
    strand_code: str,
    *,
    plot_mode: str = "debug",
    trace_context=None,
    exact_color: str | None = None,
):
    rows = []
    strand_local = _normalize_strand_code(strand_code)
    if trace_context is None:
        trace_context = st_feat.classify_browser_style_relaxed_trace(
            {strand_local: list(trace_rows or [])},
            phase=phase_value,
        )
    strand_rows = list(trace_context.get(strand_local, []) or [])
    hpsp_row = (trace_context.get("strand_hpsp_rows") or {}).get(strand_local)
    register_origin = (trace_context.get("strand_register_origins") or {}).get(strand_local)
    hpsp_position = None if register_origin is None else int(register_origin)
    phase_color = str(exact_color or _phase_color_hex(phase_value))
    direction = 1.0 if _is_forward_strand(strand_code) else -1.0

    for row in strand_rows:
        anchor_position = int(row["anchor_position"])
        plot_position = float(anchor_position)
        score_value = float(row.get("score", 0.0) or 0.0)
        relation = str(row.get("phase_relation", "other") or "other")
        is_hpsp = bool(row.get("is_hpsp", False))
        edgecolor = HPSP_RED if is_hpsp else phase_color
        facecolor = "none"
        alpha_value = 1.0
        linewidth = 0.9
        size = 22

        if is_hpsp:
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
            alpha_value = 0.78 if _normalize_locus_plot_mode(plot_mode) == "debug" else 0.88
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
                "strand": strand_local,
                "phase_relation": relation,
                "is_hpsp": is_hpsp,
            }
        )
    return rows, hpsp_position, hpsp_row


def _rank_plot_alternative_candidates(summary: dict, phase_value: int) -> list[dict]:
    candidate_pool = []
    promoted_additional_groups = summary.get("promoted_additional_peak_groups")
    if promoted_additional_groups is None:
        promoted_additional_groups = summary.get("additional_peak_groups", [])
    for group in promoted_additional_groups or []:
        candidate = dict(group)
        candidate["category"] = "other_local_peak"
        candidate_pool.append(candidate)
    for group in summary.get("overlapping_alt_groups", []) or []:
        candidate = dict(group)
        candidate["category"] = "overlapping_alternative"
        candidate_pool.append(candidate)

    candidate_pool.sort(
        key=lambda item: (
            -float(item.get("peak_score", 0.0) or 0.0),
            ALTERNATIVE_CATEGORY_PRIORITY.get(str(item.get("category", "")).strip(), 99),
            int(item.get("shift_nt") or 0),
        )
    )
    selected = candidate_pool[:MAX_COLORED_ALTERNATIVE_CANDIDATES]

    secondary_colors = _alternative_color_series(
        phase_value,
        "secondary_units",
        count=MAX_COLORED_ALTERNATIVE_CANDIDATES,
    )
    for rank_index, candidate in enumerate(selected, start=1):
        candidate["category_rank"] = sum(
            1
            for earlier in selected[: rank_index - 1]
            if str(earlier.get("category", "")).strip() == str(candidate.get("category", "")).strip()
        ) + 1
        candidate["secondary_rank"] = int(rank_index)
        candidate["color"] = secondary_colors[rank_index - 1]
    return selected


def _build_alternative_plot_layers(summary: dict, phase_value: int) -> dict:
    selected = _rank_plot_alternative_candidates(summary, phase_value)
    guide_specs = {"w": [], "c": []}
    overlay_rows = []
    legend_groups = {}

    grouped_candidates: dict[str, list[dict]] = {}
    for candidate in selected:
        category = str(candidate.get("category", "")).strip()
        grouped_candidates.setdefault(category, []).append(candidate)
        for spec in _build_colored_unit_guide_specs(candidate, phase_value):
            strand_code = _normalize_strand_code(spec.get("strand", candidate.get("strand", "w")))
            guide_specs[strand_code].append(spec)
        overlay_rows.extend(_build_candidate_unit_exact_overlay_rows(candidate, phase_value))

    for category in ("other_local_peak", "overlapping_alternative"):
        candidates = grouped_candidates.get(category)
        if not candidates:
            continue
        legend_groups[category] = {
            "label": _alternative_category_label(category),
            "colors": [str(candidate.get("color", _phase_color_hex(phase_value))) for candidate in candidates],
        }

    return {
        "selected_candidates": selected,
        "guide_specs_w": guide_specs["w"],
        "guide_specs_c": guide_specs["c"],
        "overlay_rows": overlay_rows,
        "legend_groups": legend_groups,
    }


def _build_plot_legend_groups(phase_value: int, *, plot_mode: str = "clean", alternative_legend_groups=None):
    phase_color = _main_unit_color(phase_value)
    howell_third_label = "Non-in-phase phased window"
    legend_groups = {
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
            Line2D([0], [0], marker="o", color="none", markeredgecolor=NON_PHASE_GREY, markerfacecolor="none", markersize=6, label=howell_third_label),
        ],
        "hpsp": [
            Line2D([0], [0], marker="o", color="none", markeredgecolor=HPSP_RED, markerfacecolor=HPSP_RED, markersize=6, label="Highest phasing score position / register anchor (HPSP)"),
        ],
    }
    if alternative_legend_groups:
        legend_groups["alternatives"] = list(alternative_legend_groups)
    return legend_groups


def _add_grouped_legends(
    fig,
    phase_value: int,
    *,
    main_left: float,
    main_right: float,
    legend_y: float,
    plot_mode: str = "clean",
    alternative_legend_groups=None,
) -> None:
    legend_groups = _build_plot_legend_groups(
        phase_value,
        plot_mode=plot_mode,
        alternative_legend_groups=alternative_legend_groups,
    )
    legend_common = {
        "frameon": False,
        "fontsize": 8,
        "handletextpad": 0.5,
        "columnspacing": 1.0,
        "borderaxespad": 0.0,
    }
    main_width = float(main_right) - float(main_left)
    anchor_positions = (
        float(main_left),
        float(main_left) + main_width * 0.23,
        float(main_left) + main_width * 0.43,
        float(main_left) + main_width * 0.60,
    )

    legend_read = fig.legend(
        handles=legend_groups["read_lengths"],
        loc="upper left",
        bbox_to_anchor=(anchor_positions[0], legend_y),
        ncol=2,
        **legend_common,
    )
    legend_abundance = fig.legend(
        handles=legend_groups["abundance"],
        loc="upper left",
        bbox_to_anchor=(anchor_positions[1], legend_y),
        ncol=1,
        **legend_common,
    )
    legend_howell = fig.legend(
        handles=legend_groups["howell"],
        loc="upper left",
        bbox_to_anchor=(anchor_positions[2], legend_y),
        ncol=1,
        **legend_common,
    )
    legend_hpsp = fig.legend(
        handles=legend_groups["hpsp"],
        loc="upper left",
        bbox_to_anchor=(anchor_positions[3], legend_y),
        ncol=1,
        **legend_common,
    )

    fig.add_artist(legend_read)
    fig.add_artist(legend_abundance)
    fig.add_artist(legend_howell)
    fig.add_artist(legend_hpsp)

    alternative_groups = legend_groups.get("alternatives", [])
    if alternative_groups:
        alt_anchor_positions = (
            float(main_left) + main_width * 0.43,
            float(main_left) + main_width * 0.62,
        )
        alt_y = max(float(legend_y) - 0.082, LOCUS_LAYOUT["abun_bottom"] + LOCUS_LAYOUT["abun_height"] + 0.03)
        for idx, group in enumerate(alternative_groups[:2]):
            swatches = tuple(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="--",
                    color=str(color),
                    markerfacecolor=str(color),
                    markeredgecolor=str(color),
                    markersize=5,
                    linewidth=0.9,
                )
                for color in group.get("colors", [])
            )
            alt_legend = fig.legend(
                handles=[swatches],
                labels=[group["label"]],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                loc="upper left",
                bbox_to_anchor=(alt_anchor_positions[idx], alt_y),
                ncol=1,
                **legend_common,
            )
            fig.add_artist(alt_legend)


def _normalize_plot_staging_mode(value) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in PLOT_STAGING_CHOICES:
        return "auto"
    return mode


def _scheduler_environment_detected(env=None) -> bool:
    env = os.environ if env is None else env
    return any(env.get(key) for key in SCHEDULER_ENV_VARS)


def _decode_mountinfo_value(value: str) -> str:
    text = str(value)
    return (
        text.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _parse_mountinfo_entries(mountinfo_text: str) -> list[tuple[str, str, str]]:
    entries = []
    for raw_line in str(mountinfo_text or "").splitlines():
        line = raw_line.strip()
        if not line or " - " not in line:
            continue
        pre, post = line.split(" - ", 1)
        pre_fields = pre.split()
        post_fields = post.split()
        if len(pre_fields) < 5 or not post_fields:
            continue
        mount_point = _decode_mountinfo_value(pre_fields[4])
        fs_type = str(post_fields[0]).strip().lower()
        source = str(post_fields[1]).strip().lower() if len(post_fields) > 1 else ""
        entries.append((mount_point, fs_type, source))
    return entries


def _remote_mount_details_from_text(path: str, mountinfo_text: str) -> tuple[bool, str | None, str | None]:
    abspath = os.path.abspath(os.path.expanduser(str(path)))
    best_match = None
    for mount_point, fs_type, source in _parse_mountinfo_entries(mountinfo_text):
        try:
            if os.path.commonpath([abspath, mount_point]) != mount_point:
                continue
        except Exception:
            prefix = mount_point.rstrip(os.sep) + os.sep
            if abspath != mount_point and not abspath.startswith(prefix):
                continue
        if best_match is None or len(mount_point) > len(best_match[0]):
            best_match = (mount_point, fs_type, source)
    if best_match is None:
        return False, None, None
    mount_point, fs_type, source = best_match
    is_remote = (
        fs_type in REMOTE_FILESYSTEM_TYPES
        or "quobyte" in fs_type
        or "quobyte" in source
    )
    return bool(is_remote), mount_point, fs_type


def _has_remote_path_prefix(path: str) -> bool:
    abspath = os.path.abspath(os.path.expanduser(str(path)))
    for prefix in REMOTE_PATH_PREFIXES:
        if abspath == prefix or abspath.startswith(prefix.rstrip(os.sep) + os.sep):
            return True
    return False


def _detect_remote_filesystem(path: str, mountinfo_text: str | None = None) -> tuple[bool, str | None, str | None]:
    if mountinfo_text is None and os.path.isfile("/proc/self/mountinfo"):
        try:
            with open("/proc/self/mountinfo", "r", encoding="utf-8") as handle:
                mountinfo_text = handle.read()
        except Exception:
            mountinfo_text = None
    if mountinfo_text:
        is_remote, mount_point, fs_type = _remote_mount_details_from_text(path, mountinfo_text)
        if is_remote:
            return True, mount_point, fs_type
    if _has_remote_path_prefix(path):
        return True, None, "path-prefix"
    return False, None, None


def _select_plot_staging_root(env=None) -> str | None:
    env = os.environ if env is None else env
    candidates = [
        env.get("SLURM_TMPDIR"),
        env.get("PBS_JOBFS"),
        env.get("LOCAL_SCRATCH"),
        env.get("SCRATCH"),
        env.get("TMPDIR"),
        tempfile.gettempdir(),
        "/tmp",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(str(candidate)))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK):
            return path
    return None


def _resolve_plot_staging_strategy(
    final_plot_dir: str,
    *,
    env=None,
    mountinfo_text: str | None = None,
) -> dict:
    env = os.environ if env is None else env
    requested_mode = _normalize_plot_staging_mode(getattr(rt, "plot_staging", None) or env.get("PHASIS_PLOT_STAGING", "auto"))
    is_remote_output, mount_point, fs_type = _detect_remote_filesystem(final_plot_dir, mountinfo_text=mountinfo_text)
    scheduler_detected = _scheduler_environment_detected(env=env)
    staging_root = None
    mode = "direct"

    if requested_mode == "direct":
        mode = "direct"
    else:
        wants_local = requested_mode == "local" or (requested_mode == "auto" and (scheduler_detected or is_remote_output))
        if wants_local:
            staging_root = _select_plot_staging_root(env=env)
            mode = "local" if staging_root else "direct"

    return {
        "mode": mode,
        "requested_mode": requested_mode,
        "is_remote_output": bool(is_remote_output),
        "mount_point": mount_point,
        "fs_type": fs_type,
        "scheduler_detected": bool(scheduler_detected),
        "staging_root": staging_root,
    }


def _activate_plot_staging(final_plot_dir: str, strategy: dict) -> dict:
    final_plot_dir = os.path.abspath(os.path.expanduser(str(final_plot_dir)))
    strategy = dict(strategy)
    strategy["final_plot_dir"] = final_plot_dir
    strategy["plot_dir"] = final_plot_dir
    strategy["staging_run_dir"] = None

    if strategy.get("mode") == "local" and strategy.get("staging_root"):
        try:
            staging_run_dir = tempfile.mkdtemp(prefix="phasis_locusplots_", dir=strategy["staging_root"])
            plot_dir = os.path.join(staging_run_dir, os.path.basename(final_plot_dir))
            os.makedirs(plot_dir, exist_ok=True)
            strategy["staging_run_dir"] = staging_run_dir
            strategy["plot_dir"] = plot_dir
            return strategy
        except Exception:
            strategy["mode"] = "direct"
            strategy["staging_run_dir"] = None
            strategy["plot_dir"] = final_plot_dir

    shutil.rmtree(final_plot_dir, ignore_errors=True)
    os.makedirs(final_plot_dir, exist_ok=True)
    return strategy


def _persist_plot_staging_plan(plan: dict) -> None:
    rt.plot_staging_mode = str(plan.get("mode") or "direct")
    rt.plot_staging_root = plan.get("staging_root")
    try:
        if hasattr(rt, "save_snapshot"):
            rt.save_snapshot()
    except Exception:
        pass


def _finalize_staged_plot_dir(staged_plot_dir: str, final_plot_dir: str) -> str:
    staged_plot_dir = os.path.abspath(os.path.expanduser(str(staged_plot_dir)))
    final_plot_dir = os.path.abspath(os.path.expanduser(str(final_plot_dir)))
    final_parent = os.path.dirname(final_plot_dir) or "."
    os.makedirs(final_parent, exist_ok=True)
    temp_destination = tempfile.mkdtemp(prefix=".phasis_plot_copy_", dir=final_parent)
    temp_plot_dir = os.path.join(temp_destination, os.path.basename(final_plot_dir))
    try:
        shutil.copytree(staged_plot_dir, temp_plot_dir)
        shutil.rmtree(final_plot_dir, ignore_errors=True)
        os.replace(temp_plot_dir, final_plot_dir)
        return final_plot_dir
    finally:
        shutil.rmtree(temp_destination, ignore_errors=True)


def _resolved_locus_plot_ncores() -> int:
    try:
        ncores = int(getattr(rt, "ncores", 0) or 0)
    except Exception:
        ncores = 0
    if ncores <= 0:
        ncores = max(1, cpu_count())
    return int(max(1, ncores))


def _resolve_locus_plot_worker_cap(task_count: int, *, direct_remote: bool = False) -> int:
    cap = min(MAX_LOCUS_PLOT_WORKERS, _resolved_locus_plot_ncores(), max(1, int(task_count)))
    if direct_remote:
        cap = min(cap, REMOTE_DIRECT_LOCUS_PLOT_WORKER_CAP)
    return int(max(1, cap))


def _prepare_cluster_df(cluster_rows) -> pd.DataFrame:
    cluster_df = pd.DataFrame(cluster_rows)
    if cluster_df.empty:
        return cluster_df

    local = cluster_df.copy()
    local["pos"] = pd.to_numeric(local["pos"], errors="coerce")
    local["abun"] = pd.to_numeric(local["abun"], errors="coerce").fillna(0.0)
    local["len"] = pd.to_numeric(local["len"], errors="coerce")
    if "hits" in local.columns:
        local["hits"] = pd.to_numeric(local["hits"], errors="coerce")
    else:
        local["hits"] = np.nan
    if "tag_seq" not in local.columns:
        local["tag_seq"] = np.nan
    if "strand" not in local.columns:
        local["strand"] = ""
    local["strand_norm"] = local["strand"].map(_normalize_strand_code)
    local = local.dropna(subset=["pos", "len"]).reset_index(drop=True)
    return local


def _select_offset_position(pos_abundance: dict[int, float], expected_position: int) -> int | None:
    left = int(expected_position) - 1
    right = int(expected_position) + 1
    left_has = left in pos_abundance
    right_has = right in pos_abundance
    if left_has and right_has:
        return left if float(pos_abundance[left]) >= float(pos_abundance[right]) else right
    if left_has:
        return left
    if right_has:
        return right
    return None


def _export_hits_value(value):
    try:
        if pd.isna(value):
            return np.nan
        numeric = float(value)
        if abs(numeric - round(numeric)) < 1e-9:
            return int(round(numeric))
        return numeric
    except Exception:
        return np.nan


def _materialize_export_rows(
    rows_df: pd.DataFrame,
    *,
    expected_position: int,
    register_class: str,
    identifier_text: str,
    cid_value: str,
    alib_value: str,
    phase_value: int,
    strand_code: str,
    window_unit_id: str,
    window_unit_role: str,
    window_unit_rank: int,
    window_unit_shift_nt,
) -> list[dict]:
    rows = []
    for row in rows_df.itertuples(index=False):
        rows.append(
            {
                "identifier": identifier_text,
                "cID": cid_value,
                "alib": alib_value,
                "phase": int(phase_value),
                "window_unit_id": str(window_unit_id),
                "window_unit_role": str(window_unit_role),
                "window_unit_rank": int(window_unit_rank),
                "window_unit_shift_nt": window_unit_shift_nt,
                "strand": _normalize_strand_code(strand_code),
                "observed_pos": int(getattr(row, "pos")),
                "expected_register_pos": int(expected_position),
                "register_class": register_class,
                "abun": float(getattr(row, "abun", 0.0) or 0.0),
                "tag_seq": getattr(row, "tag_seq", np.nan),
                "hits": _export_hits_value(getattr(row, "hits", np.nan)),
            }
        )
    return rows


def _collect_export_rows_for_strand(
    cluster_df: pd.DataFrame,
    *,
    phase_value: int,
    strand_code: str,
    base_positions,
    extended_positions,
    identifier_text: str,
    cid_value: str,
    alib_value: str,
    window_unit_id: str = "unit_0",
    window_unit_role: str = "main_hpsp",
    window_unit_rank: int = 0,
    window_unit_shift_nt=np.nan,
) -> list[dict]:
    if cluster_df.empty:
        return []

    phase_df = cluster_df.loc[
        (cluster_df["strand_norm"] == _normalize_strand_code(strand_code))
        & (pd.to_numeric(cluster_df["len"], errors="coerce") == int(phase_value))
    ].copy()
    if phase_df.empty:
        return []

    phase_df["pos"] = pd.to_numeric(phase_df["pos"], errors="coerce")
    phase_df = phase_df.dropna(subset=["pos"]).copy()
    if phase_df.empty:
        return []
    phase_df["pos"] = phase_df["pos"].astype(int)

    rows_by_position = {}
    abundance_by_position = {}
    for pos_value, pos_df in phase_df.groupby("pos", sort=False):
        pos_local = pos_df.copy()
        pos_key = int(pos_value)
        rows_by_position[pos_key] = pos_local
        abundance_by_position[pos_key] = float(pd.to_numeric(pos_local["abun"], errors="coerce").fillna(0.0).sum())

    export_rows = []
    for expected_position in base_positions or []:
        expected_local = int(expected_position)
        exact_rows = rows_by_position.get(expected_local)
        if exact_rows is not None and not exact_rows.empty:
            export_rows.extend(
                _materialize_export_rows(
                    exact_rows,
                    expected_position=expected_local,
                    register_class="core_exact",
                    identifier_text=identifier_text,
                    cid_value=cid_value,
                    alib_value=alib_value,
                    phase_value=phase_value,
                    strand_code=strand_code,
                    window_unit_id=window_unit_id,
                    window_unit_role=window_unit_role,
                    window_unit_rank=window_unit_rank,
                    window_unit_shift_nt=window_unit_shift_nt,
                )
            )
            continue
        offset_position = _select_offset_position(abundance_by_position, expected_local)
        if offset_position is None:
            continue
        offset_rows = rows_by_position.get(int(offset_position))
        if offset_rows is None or offset_rows.empty:
            continue
        export_rows.extend(
            _materialize_export_rows(
                offset_rows,
                expected_position=expected_local,
                register_class="core_offset",
                identifier_text=identifier_text,
                cid_value=cid_value,
                alib_value=alib_value,
                phase_value=phase_value,
                strand_code=strand_code,
                window_unit_id=window_unit_id,
                window_unit_role=window_unit_role,
                window_unit_rank=window_unit_rank,
                window_unit_shift_nt=window_unit_shift_nt,
            )
        )

    for expected_position in extended_positions or []:
        expected_local = int(expected_position)
        exact_rows = rows_by_position.get(expected_local)
        if exact_rows is None or exact_rows.empty:
            continue
        export_rows.extend(
            _materialize_export_rows(
                exact_rows,
                expected_position=expected_local,
                register_class="extended_exact",
                identifier_text=identifier_text,
                cid_value=cid_value,
                alib_value=alib_value,
                phase_value=phase_value,
                strand_code=strand_code,
                window_unit_id=window_unit_id,
                window_unit_role=window_unit_role,
                window_unit_rank=window_unit_rank,
                window_unit_shift_nt=window_unit_shift_nt,
            )
        )

    return export_rows


def _unit_member_export_positions(member: dict, phase_value: int) -> tuple[list[int], list[int]]:
    peak_row = member.get("peak_row")
    strand_code = member.get("strand", "w")
    trace_rows = member.get("rows", [])
    base_positions, register_origin = _build_scored_register_positions(
        peak_row,
        phase_value,
        strand_code,
    )
    extended_positions = _collect_extended_register_positions(
        trace_rows,
        register_origin,
        base_positions,
        phase_value,
    )
    return base_positions, extended_positions


def _export_rows_for_unit_member(
    cluster_df: pd.DataFrame,
    *,
    member: dict,
    phase_value: int,
    identifier_text: str,
    cid_value: str,
    alib_value: str,
    window_unit_id: str,
    window_unit_role: str,
    window_unit_rank: int,
    window_unit_shift_nt,
) -> list[dict]:
    base_positions, extended_positions = _unit_member_export_positions(member, phase_value)
    if not base_positions and not extended_positions:
        return []
    return _collect_export_rows_for_strand(
        cluster_df,
        phase_value=phase_value,
        strand_code=str(member.get("strand", "w")),
        base_positions=base_positions,
        extended_positions=extended_positions,
        identifier_text=identifier_text,
        cid_value=cid_value,
        alib_value=alib_value,
        window_unit_id=window_unit_id,
        window_unit_role=window_unit_role,
        window_unit_rank=window_unit_rank,
        window_unit_shift_nt=window_unit_shift_nt,
    )


def _build_x_bounds(cluster_df: pd.DataFrame, identifier_text: str, howell_rows, phase_value: int):
    cluster_positions = pd.to_numeric(cluster_df["pos"], errors="coerce").dropna()
    if not cluster_positions.empty:
        return float(cluster_positions.min()), float(cluster_positions.max())

    x_values = [float(row["x"]) for row in howell_rows if row.get("x") is not None]
    xmin, xmax = _parse_identifier_interval(identifier_text)
    if xmin is None or xmax is None:
        xmin = min(x_values) if x_values else 0.0
        xmax = max(x_values) if x_values else float(phase_value)
    return float(xmin), float(xmax)


def _clean_ambiguity_count(value):
    try:
        if pd.isna(value):
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _clean_ambiguity_float(value):
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _clean_ambiguity_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        return text
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _format_origin_class(value) -> str:
    raw_value = _clean_ambiguity_text(value)
    if raw_value is None:
        return "NA"
    return ORIGIN_CLASS_LABELS.get(raw_value, raw_value.replace("_", " ").capitalize())


def _howell_peak_cutoff_text() -> str:
    try:
        return f"{float(getattr(rt, 'min_Howell_score', 12.5)):.2f}"
    except Exception:
        return "12.50"


def _coalesce_task_metric(primary_value, fallback_value):
    cleaned_primary = _clean_ambiguity_text(primary_value)
    if cleaned_primary is not None:
        return primary_value
    return fallback_value


def _wrap_strip_text(text_value: str, *, width: int = 24) -> str:
    text_local = str(text_value or "").strip()
    if not text_local:
        return ""
    return "\n".join(
        textwrap.wrap(
            text_local,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _crowding_interpretation_line(crowding_window_count: int) -> str | None:
    count_local = int(crowding_window_count)
    if count_local <= 0:
        return None
    if count_local <= 2:
        return "Sparse non-in-phase context (0-2)"
    if count_local <= 5:
        return "Moderate non-in-phase context (3-5)"
    return "Dense non-in-phase context (>5)"


def _small_count_text(value: int) -> str:
    lookup = {
        0: "Zero",
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
        11: "Eleven",
        12: "Twelve",
    }
    try:
        ivalue = int(value)
    except Exception:
        return str(value)
    return lookup.get(ivalue, str(ivalue))


def _non_in_phase_context_sentence(payload: dict) -> str | None:
    crowding_count = payload.get("crowding_window_count_value")
    if crowding_count is None:
        return None
    try:
        count_local = int(crowding_count)
    except Exception:
        return None
    if count_local <= 0:
        return None

    promoted_overlap_count = int(payload.get("overlapping_alt_count_value") or 0)
    promoted_distal_count = int(payload.get("promoted_additional_peak_count_value") or 0)
    promoted_total = int(promoted_overlap_count + promoted_distal_count)

    count_text = _small_count_text(count_local)
    noun = "window" if count_local == 1 else "windows"
    if promoted_total <= 0:
        verb = "was" if count_local == 1 else "were"
        return (
            f"{count_text} non-in-phase context {noun} {verb} detected "
            f"near the main Howell peak, but no secondary candidate unit was promoted."
        )
    verb = "remains" if count_local == 1 else "remain"
    return (
        f"{count_text} additional non-in-phase context {noun} {verb} "
        f"unpromoted near the main Howell peak."
    )


def _build_interpretation_lines(task: dict, payload: dict) -> list[str]:
    final_class = _clean_ambiguity_text(task.get("final_class")) or "non-PHAS"
    origin_class_raw = _clean_ambiguity_text(task.get("Howell_origin_class")) or ""
    relaxed_peak_score = _clean_ambiguity_float(task.get("Peak_Howell_score"))
    exact_support = payload["exact_support_value"]
    extension_window_count = payload["extension_window_count_value"] or 0
    origin_window_count = payload["origin_window_count_value"] or 0
    alt_register_count = payload["raw_alt_register_count_value"] or 0
    crowding_window_count = payload["crowding_window_count_value"] or 0

    lines = []
    if final_class == "non-PHAS" and (
        exact_support is None
        or exact_support <= 0.0
        or origin_class_raw == "insufficient_exact_support"
    ):
        return [
            "Relaxed HPSP detected",
            "Exact-only support absent",
            "No reliable exact frame",
            _format_phas_text("Classified as non-PHAS"),
        ]

    if exact_support is not None and exact_support > 0.0:
        support_label = "Strong" if exact_support >= 20.0 else "Moderate"
        lines.append(f"{support_label} exact support")
    if origin_window_count == 0:
        lines.append("No competing exact frames")
    if alt_register_count == 0:
        lines.append("No alternate strong registers")
    if origin_class_raw == "coherent_extension" and extension_window_count > 0:
        lines.append("Broad same-frame extension")
    crowding_line = _crowding_interpretation_line(crowding_window_count)
    if crowding_line:
        lines.append(crowding_line)
    overlapping_alt_count = payload["overlapping_alt_count_value"] or 0
    if overlapping_alt_count > 0:
        lines.append("Promoted secondary relaxed candidate units present")
    if final_class == "PHAS-like":
        if relaxed_peak_score is not None and relaxed_peak_score < 20.0:
            lines.append("Modest relaxed Howell support")
        lines.append(_format_phas_text("Classified as PHAS-like"))
    elif final_class == "PHAS":
        lines.append(_format_phas_text("Classified as PHAS"))
    elif final_class == "non-PHAS":
        lines.append(_format_phas_text("Classified as non-PHAS"))

    deduped = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped[:5]


def _build_ambiguity_sidebar_payload(task: dict, *, plot_mode: str = "clean") -> dict:
    exact_support = _clean_ambiguity_float(task.get("Howell_exact_support_score"))
    relaxed_peak_score = _clean_ambiguity_float(task.get("Peak_Howell_score"))
    overlap_count = _clean_ambiguity_count(task.get("Howell_ambiguity_count"))
    alt_register_count = _clean_ambiguity_count(task.get("Howell_alt_register_count"))
    overlap_margin = _clean_ambiguity_float(task.get("Howell_overlap_margin"))
    extension_window_count = _clean_ambiguity_count(task.get("Howell_extension_window_count"))
    extension_span_nt = _clean_ambiguity_count(task.get("Howell_extension_span_nt"))
    origin_window_count = _clean_ambiguity_count(task.get("Howell_origin_window_count"))
    origin_frame_count = _clean_ambiguity_count(task.get("Howell_origin_frame_count"))
    origin_margin = _clean_ambiguity_float(task.get("Howell_origin_margin"))
    origin_class = _format_origin_class(task.get("Howell_origin_class"))
    additional_peak_count = _clean_ambiguity_count(task.get("Howell_additional_peak_count"))
    additional_peak_best_score = _clean_ambiguity_float(task.get("Howell_additional_peak_best_score"))
    promoted_additional_peak_count = _clean_ambiguity_count(task.get("Howell_promoted_additional_peak_count"))
    promoted_additional_peak_best_score = _clean_ambiguity_float(task.get("Howell_promoted_additional_peak_best_score"))
    overlapping_alt_count = _clean_ambiguity_count(task.get("Howell_overlapping_alt_count"))
    overlapping_alt_best_score = _clean_ambiguity_float(task.get("Howell_overlapping_alt_best_score"))
    overlapping_alt_best_shift_nt = _clean_ambiguity_float(task.get("Howell_overlapping_alt_best_shift_nt"))
    crowding_window_count = _clean_ambiguity_count(task.get("Howell_crowding_window_count"))
    crowding_best_score = _clean_ambiguity_float(task.get("Howell_crowding_best_score"))
    crowding_score_gap = _clean_ambiguity_float(task.get("Howell_crowding_score_gap"))
    payload = {
        "exact_support": "NA" if exact_support is None else f"{exact_support:.2f}",
        "exact_support_value": exact_support,
        "relaxed_peak_score": "NA" if relaxed_peak_score is None else f"{relaxed_peak_score:.2f}",
        "relaxed_peak_score_value": relaxed_peak_score,
        "origin_class": origin_class,
        "extension_window_count": "NA" if extension_window_count is None else str(extension_window_count),
        "extension_window_count_value": extension_window_count,
        "extension_span_nt": "NA" if extension_span_nt is None else str(extension_span_nt),
        "extension_span_nt_value": extension_span_nt,
        "origin_window_count": "NA" if origin_window_count is None else str(origin_window_count),
        "origin_window_count_value": origin_window_count,
        "origin_frame_count": "NA" if origin_frame_count is None else str(origin_frame_count),
        "origin_frame_count_value": origin_frame_count,
        "origin_margin": "NA" if origin_margin is None else f"{origin_margin:.2f}",
        "origin_margin_value": origin_margin,
        "additional_peak_count": "NA" if additional_peak_count is None else str(additional_peak_count),
        "additional_peak_count_value": additional_peak_count,
        "additional_peak_best_score": "NA" if additional_peak_best_score is None else f"{additional_peak_best_score:.2f}",
        "additional_peak_best_score_value": additional_peak_best_score,
        "additional_peak_cutoff": _howell_peak_cutoff_text(),
        "promoted_additional_peak_count_value": promoted_additional_peak_count,
        "promoted_additional_peak_best_score": (
            "NA" if promoted_additional_peak_best_score is None else f"{promoted_additional_peak_best_score:.2f}"
        ),
        "promoted_additional_peak_best_score_value": promoted_additional_peak_best_score,
        "overlapping_alt_count": "NA" if overlapping_alt_count is None else str(overlapping_alt_count),
        "overlapping_alt_count_value": overlapping_alt_count,
        "overlapping_alt_best_score": "NA" if overlapping_alt_best_score is None else f"{overlapping_alt_best_score:.2f}",
        "overlapping_alt_best_score_value": overlapping_alt_best_score,
        "overlapping_alt_best_shift_nt": _format_shift_nt_text(overlapping_alt_best_shift_nt),
        "overlapping_alt_best_shift_nt_value": overlapping_alt_best_shift_nt,
        "crowding_window_count": "NA" if crowding_window_count is None else str(crowding_window_count),
        "crowding_window_count_value": crowding_window_count,
        "crowding_best_score": "NA" if crowding_best_score is None else f"{crowding_best_score:.2f}",
        "crowding_best_score_value": crowding_best_score,
        "crowding_score_gap": "NA" if crowding_score_gap is None else f"{crowding_score_gap:.2f}",
        "crowding_score_gap_value": crowding_score_gap,
        "raw_overlap_count": "NA" if overlap_count is None else str(overlap_count),
        "raw_overlap_count_value": overlap_count,
        "raw_alt_register_count": "NA" if alt_register_count is None else str(alt_register_count),
        "raw_alt_register_count_value": alt_register_count,
        "raw_overlap_margin": "NA" if overlap_margin is None else f"{overlap_margin:.2f}",
        "raw_overlap_margin_value": overlap_margin,
        "plot_mode": _normalize_locus_plot_mode(plot_mode),
    }
    payload["interpretation_lines"] = _build_interpretation_lines(task, payload)
    payload["non_in_phase_context_sentence"] = _non_in_phase_context_sentence(payload)
    return payload


def _build_ambiguity_sidebar_note(task: dict, *, plot_mode: str = "clean") -> str:
    exact_support = _clean_ambiguity_float(task.get("Howell_exact_support_score"))
    if exact_support is None or exact_support <= 0.0:
        return "Exact-only interpretation unavailable.\nRelaxed HPSP lacks exact support.\nEach Howell point summarizes a 10-cycle window anchored at that mapped sRNA."
    if _normalize_locus_plot_mode(plot_mode) == "clean":
        return "Non-in-phase phased windows are shown as hollow points.\nEach Howell point summarizes a 10-cycle window anchored at that mapped sRNA."
    return "Non-in-phase phased windows are shown as hollow points.\nEach Howell point summarizes a 10-cycle window anchored at that mapped sRNA."


def _build_strip_sections(task: dict, payload: dict, *, plot_mode: str = "clean") -> list[dict]:
    sections = [
        {
            "title": "Exact-only Howell",
            "title_fontsize": 8.6,
            "line_fontsize": 7.8,
                "lines": [
                    f"Support: {payload['exact_support']}",
                    f"Relaxed peak: {payload['relaxed_peak_score']}",
                    f"Class: {payload['origin_class']}",
                ],
            }
        ]

    if payload["interpretation_lines"]:
        sections.append(
            {
                "title": "Interpretation",
                "title_fontsize": 8.0,
                "line_fontsize": 7.1,
                "lines": payload["interpretation_lines"],
            }
        )

    if (
        payload["extension_window_count_value"] is not None
        or payload["extension_span_nt_value"] is not None
    ):
        sections.append(
            {
                "title": "Coherent extension",
                "title_fontsize": 8.0,
                "line_fontsize": 7.2,
                "lines": [
                    f"Windows: {payload['extension_window_count']}",
                    f"Span: {payload['extension_span_nt']} nt",
                ],
            }
        )

    if (
        payload["origin_window_count_value"] is not None
        or payload["origin_frame_count_value"] is not None
        or payload["origin_margin_value"] is not None
    ):
        sections.append(
            {
                "title": "Origin ambiguity",
                "title_fontsize": 8.0,
                "line_fontsize": 7.2,
                "lines": [
                    f"Windows: {payload['origin_window_count']}",
                    f"Frames: {payload['origin_frame_count']}",
                    f"Margin: {payload['origin_margin']}",
                ],
            }
        )

    if (payload["promoted_additional_peak_count_value"] or 0) > 0:
        sections.append(
            {
                "title": "Other local peaks",
                "title_fontsize": 8.0,
                "line_fontsize": 7.2,
                "lines": [
                    f"Count: {payload['promoted_additional_peak_count_value']}",
                    f"Best: {payload['promoted_additional_peak_best_score']}",
                    f"Cutoff: {payload['additional_peak_cutoff']}",
                ],
            }
        )

    if (
        (payload["overlapping_alt_count_value"] or 0) > 0
        or payload["overlapping_alt_best_score_value"] is not None
        or payload["overlapping_alt_best_shift_nt_value"] is not None
    ):
        sections.append(
            {
                "title": "Overlapping alternative candidates",
                "title_fontsize": 8.0,
                "line_fontsize": 7.2,
                "lines": [
                    f"Count: {payload['overlapping_alt_count']}",
                    f"Best: {payload['overlapping_alt_best_score']}",
                    f"Shift: {payload['overlapping_alt_best_shift_nt']}",
                ],
            }
        )

    if (
        (payload["crowding_window_count_value"] or 0) > 0
        or payload["crowding_best_score_value"] is not None
        or payload["crowding_score_gap_value"] is not None
    ):
        crowding_lines = [
            f"Windows: {payload['crowding_window_count']}",
            f"Best: {payload['crowding_best_score']}",
            f"Gap: {payload['crowding_score_gap']}",
        ]
        context_sentence = payload.get("non_in_phase_context_sentence")
        if context_sentence:
            crowding_lines.append(context_sentence)
        sections.append(
            {
                "title": "Non-in-phase context windows",
                "title_fontsize": 8.0,
                "line_fontsize": 7.2,
                "lines": crowding_lines,
            }
        )

    if _normalize_locus_plot_mode(plot_mode) == "debug":
        sections.append(
            {
                "title": "Raw context",
                "title_fontsize": 7.8,
                "line_fontsize": 7.0,
                "lines": [
                    f"Near-tied: {payload['raw_overlap_count']}",
                    f"Alt regs: {payload['raw_alt_register_count']}",
                    f"Raw margin: {payload['raw_overlap_margin']}",
                ],
            }
        )

    note_text = _build_ambiguity_sidebar_note(task, plot_mode=plot_mode)
    if note_text:
        sections.append(
            {
                "title": "Notes",
                "title_fontsize": 7.8,
                "line_fontsize": 6.4,
                "lines": note_text.splitlines(),
                "color": "#666666",
            }
        )
    return sections


def _draw_strip_line(
    ax,
    y_value: float,
    text_value: str,
    *,
    fontsize: float,
    fontweight: str = "normal",
    color: str = "#333333",
    line_gap: float = 0.032,
    paragraph_gap: float = 0.008,
) -> float:
    wrapped_text = str(text_value)
    ax.text(
        0.07,
        y_value,
        wrapped_text,
        fontsize=fontsize,
        fontweight=fontweight,
        ha="left",
        va="top",
        color=color,
        transform=ax.transAxes,
        clip_on=True,
    )
    line_count = max(wrapped_text.count("\n") + 1, 1)
    return y_value - (line_gap * line_count) - paragraph_gap


def _wrap_strip_title(text_value: str, *, width: int = 24) -> str:
    return _wrap_strip_text(text_value, width=width)


def _draw_detachable_strip(fig, task: dict, layout: dict) -> None:
    strip_left = float(layout["strip_left"])
    strip_bottom = float(layout["strip_bottom"])
    strip_width = float(layout["strip_right"]) - strip_left
    strip_height = float(layout["strip_top"]) - strip_bottom
    ax = fig.add_axes([strip_left, strip_bottom, strip_width, strip_height])
    ax.set_facecolor(DETACHABLE_STRIP_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plot_mode = _current_locus_plot_mode()
    payload = _build_ambiguity_sidebar_payload(task, plot_mode=plot_mode)
    sections = _build_strip_sections(task, payload, plot_mode=plot_mode)
    y_cursor = 0.95
    for section in sections:
        title_color = section.get("color", "#444444")
        wrapped_title = _wrap_strip_title(
            section["title"],
            width=int(section.get("title_wrap_width", 24)),
        )
        wrapped_title_lines = max(wrapped_title.count("\n") + 1, 1)
        title_fontsize = float(section.get("title_fontsize", 8.0))
        if wrapped_title_lines > 1:
            title_fontsize = min(title_fontsize, 7.6)
        y_cursor = _draw_strip_line(
            ax,
            y_cursor,
            wrapped_title,
            fontsize=title_fontsize,
            fontweight="bold",
            color=title_color,
        )
        for line in section.get("lines", []):
            wrapped = _wrap_strip_text(line, width=24)
            y_cursor = _draw_strip_line(
                ax,
                y_cursor,
                wrapped,
                fontsize=section.get("line_fontsize", 7.2),
                color=section.get("color", "#555555"),
                paragraph_gap=0.004,
            )
        y_cursor -= 0.012


def _draw_detachable_separator(fig, layout: dict) -> None:
    separator = Line2D(
        [float(layout["separator_x"]), float(layout["separator_x"])],
        [float(layout["separator_bottom"]), float(layout["separator_top"])],
        transform=fig.transFigure,
        color=DETACHABLE_SEPARATOR_COLOR,
        linewidth=1.0,
        linestyle=DETACHABLE_SEPARATOR_STYLE,
        zorder=20,
    )
    fig.add_artist(separator)


def _placeholder_payload(task: dict, message_text: str) -> dict:
    return {
        "plot_kind": "placeholder",
        "plot_path": task["plot_path"],
        "title_text": task["title_text"],
        "phase": int(task["phase"]),
        "message_text": message_text,
    }


def _analyze_single_locus(task: dict) -> dict:
    phase_value = int(task["phase"])
    cluster_rows = task.get("cluster_rows", [])

    if not cluster_rows:
        return {
            "plot_payload": _placeholder_payload(task, "No cluster rows were available for this detection."),
            "phasiRNA_rows": [],
        }

    cluster_df = _prepare_cluster_df(cluster_rows)
    if cluster_df.empty:
        return {
            "plot_payload": _placeholder_payload(task, "The locus data were empty after reconstruction."),
            "phasiRNA_rows": [],
        }

    trace = st_feat.enumerate_relaxed_howell_trace(cluster_df, phase=phase_value)
    browser_trace_context = st_feat.classify_browser_style_relaxed_trace(trace, phase=phase_value)
    additional_peak_summary = st_feat.summarize_relaxed_trace_subregions(
        trace,
        score_cutoff=float(getattr(rt, "min_Howell_score", 12.5) or 12.5),
        phase=phase_value,
    )
    main_unit = additional_peak_summary.get("main_biogenesis_unit")
    main_phase_color = _main_unit_color(phase_value)
    w_trace = trace.get("w", [])
    c_trace = trace.get("c", [])
    if not w_trace and not c_trace:
        return {
            "plot_payload": _placeholder_payload(
                task,
                "No phase-length reads were available to compute the relaxed Howell trace.",
            ),
            "phasiRNA_rows": [],
        }

    abundance_rows = _build_abundance_rows(cluster_df)
    plot_mode = _current_locus_plot_mode()
    exact_competitor_context = st_feat.collect_exact_only_peak_competitors(
        cluster_df,
        phase=phase_value,
    )
    howell_rows_w, _hpsp_w, hpsp_row_w = _build_howell_rows(
        w_trace,
        phase_value,
        "w",
        plot_mode=plot_mode,
        trace_context=browser_trace_context,
        exact_color=main_phase_color,
    )
    howell_rows_c, _hpsp_c, hpsp_row_c = _build_howell_rows(
        c_trace,
        phase_value,
        "c",
        plot_mode=plot_mode,
        trace_context=browser_trace_context,
        exact_color=main_phase_color,
    )
    howell_rows = howell_rows_w + howell_rows_c
    main_unit_guides = _build_main_unit_guide_specs(main_unit, phase_value)
    guide_specs_w = main_unit_guides["w"]
    guide_specs_c = main_unit_guides["c"]
    alternative_layers = _build_alternative_plot_layers(additional_peak_summary, phase_value)
    exact_peak_summary = exact_competitor_context.get("summary") or {}

    export_rows = []
    if main_unit:
        for member in list(main_unit.get("members") or []):
            role = str(member.get("unit_role") or "main_hpsp")
            export_rows.extend(
                _export_rows_for_unit_member(
                    cluster_df,
                    member=member,
                    phase_value=phase_value,
                    identifier_text=str(task["identifier_text"]),
                    cid_value=str(task["cid_value"]),
                    alib_value=str(task["alib_value"]),
                    window_unit_id="unit_main",
                    window_unit_role=role,
                    window_unit_rank=0,
                    window_unit_shift_nt=(0 if role == "main_hpsp" else member.get("shift_nt", np.nan)),
                )
            )
    else:
        base_positions_w, register_origin_w = _build_scored_register_positions(hpsp_row_w, phase_value, "w")
        base_positions_c, register_origin_c = _build_scored_register_positions(hpsp_row_c, phase_value, "c")
        extended_positions_w = _collect_extended_register_positions(w_trace, register_origin_w, base_positions_w, phase_value)
        extended_positions_c = _collect_extended_register_positions(c_trace, register_origin_c, base_positions_c, phase_value)
        export_rows.extend(
            _collect_export_rows_for_strand(
                cluster_df,
                phase_value=phase_value,
                strand_code="w",
                base_positions=base_positions_w,
                extended_positions=extended_positions_w,
                identifier_text=str(task["identifier_text"]),
                cid_value=str(task["cid_value"]),
                alib_value=str(task["alib_value"]),
                window_unit_id="unit_main",
                window_unit_role="main_hpsp",
                window_unit_rank=0,
                window_unit_shift_nt=0,
            )
        )
        export_rows.extend(
            _collect_export_rows_for_strand(
                cluster_df,
                phase_value=phase_value,
                strand_code="c",
                base_positions=base_positions_c,
                extended_positions=extended_positions_c,
                identifier_text=str(task["identifier_text"]),
                cid_value=str(task["cid_value"]),
                alib_value=str(task["alib_value"]),
                window_unit_id="unit_main",
                window_unit_role="main_hpsp",
                window_unit_rank=0,
                window_unit_shift_nt=0,
            )
        )

    for rank_index, unit in enumerate(additional_peak_summary.get("promoted_secondary_units", []) or [], start=1):
        unit_id = f"unit_secondary_{rank_index}"
        unit_role = str(unit.get("category") or "overlapping_alternative")
        unit_shift_nt = unit.get("shift_nt", np.nan)
        for member in _candidate_unit_members(unit):
            export_rows.extend(
                _export_rows_for_unit_member(
                    cluster_df,
                    member=member,
                    phase_value=phase_value,
                    identifier_text=str(task["identifier_text"]),
                    cid_value=str(task["cid_value"]),
                    alib_value=str(task["alib_value"]),
                    window_unit_id=unit_id,
                    window_unit_role=unit_role,
                    window_unit_rank=rank_index,
                    window_unit_shift_nt=unit_shift_nt,
                )
            )

    xmin, xmax = _build_x_bounds(cluster_df, str(task["identifier_text"]), howell_rows, phase_value)
    span = max(float(xmax) - float(xmin), float(phase_value))
    xpad = max(float(phase_value) * 4.0, span * 0.06)

    max_abun = max([abs(row["y"]) for row in abundance_rows], default=1.0)
    max_score = max([abs(row["y"]) for row in howell_rows], default=1.0)
    abun_ylim = max(1.0, max_abun * 1.15)
    score_ylim = max(1.0, max_score * 1.20)
    promoted_additional_peak_groups = additional_peak_summary.get("promoted_additional_peak_groups")
    if promoted_additional_peak_groups is None:
        promoted_additional_peak_groups = additional_peak_summary.get("additional_peak_groups", []) or []

    return {
        "plot_payload": {
            "plot_kind": "plot",
            "plot_path": task["plot_path"],
            "title_text": task["title_text"],
            "phase": phase_value,
            "abundance_rows": abundance_rows,
            "howell_rows": howell_rows,
            "guide_specs_w": guide_specs_w,
            "guide_specs_c": guide_specs_c,
            "main_unit_color": main_phase_color,
            "alternative_guide_specs_w": alternative_layers["guide_specs_w"],
            "alternative_guide_specs_c": alternative_layers["guide_specs_c"],
            "alternative_howell_overlay_rows": alternative_layers["overlay_rows"],
            "alternative_legend_groups": alternative_layers["legend_groups"],
            "xmin": float(xmin),
            "xmax": float(xmax),
            "xpad": float(xpad),
            "abun_ylim": float(abun_ylim),
            "score_ylim": float(score_ylim),
            "Howell_exact_support_score": _coalesce_task_metric(
                task.get("Howell_exact_support_score"),
                exact_peak_summary.get("Howell_exact_support_score"),
            ),
            "Howell_ambiguity_count": _coalesce_task_metric(
                task.get("Howell_ambiguity_count"),
                exact_peak_summary.get("Howell_ambiguity_count"),
            ),
            "Howell_alt_register_count": _coalesce_task_metric(
                task.get("Howell_alt_register_count"),
                exact_peak_summary.get("Howell_alt_register_count"),
            ),
            "Howell_overlap_margin": _coalesce_task_metric(
                task.get("Howell_overlap_margin"),
                exact_peak_summary.get("Howell_overlap_margin"),
            ),
            "Howell_extension_window_count": _coalesce_task_metric(
                task.get("Howell_extension_window_count"),
                exact_peak_summary.get("Howell_extension_window_count"),
            ),
            "Howell_extension_span_nt": _coalesce_task_metric(
                task.get("Howell_extension_span_nt"),
                exact_peak_summary.get("Howell_extension_span_nt"),
            ),
            "Howell_origin_window_count": _coalesce_task_metric(
                task.get("Howell_origin_window_count"),
                exact_peak_summary.get("Howell_origin_window_count"),
            ),
            "Howell_origin_frame_count": _coalesce_task_metric(
                task.get("Howell_origin_frame_count"),
                exact_peak_summary.get("Howell_origin_frame_count"),
            ),
            "Howell_origin_margin": _coalesce_task_metric(
                task.get("Howell_origin_margin"),
                exact_peak_summary.get("Howell_origin_margin"),
            ),
            "Howell_origin_class": _coalesce_task_metric(
                task.get("Howell_origin_class"),
                exact_peak_summary.get("Howell_origin_class"),
            ),
            "Howell_additional_peak_count": _coalesce_task_metric(
                task.get("Howell_additional_peak_count"),
                additional_peak_summary.get("Howell_additional_peak_count"),
            ),
            "Howell_additional_peak_best_score": _coalesce_task_metric(
                task.get("Howell_additional_peak_best_score"),
                additional_peak_summary.get("Howell_additional_peak_best_score"),
            ),
            "Howell_promoted_additional_peak_count": _coalesce_task_metric(
                task.get("Howell_promoted_additional_peak_count"),
                len(promoted_additional_peak_groups),
            ),
            "Howell_promoted_additional_peak_best_score": _coalesce_task_metric(
                task.get("Howell_promoted_additional_peak_best_score"),
                (
                    np.nan
                    if not promoted_additional_peak_groups
                    else max(
                        float(group.get("peak_score", 0.0) or 0.0)
                        for group in promoted_additional_peak_groups
                    )
                ),
            ),
            "Howell_overlapping_alt_count": _coalesce_task_metric(
                task.get("Howell_overlapping_alt_count"),
                additional_peak_summary.get("Howell_overlapping_alt_count"),
            ),
            "Howell_overlapping_alt_best_score": _coalesce_task_metric(
                task.get("Howell_overlapping_alt_best_score"),
                additional_peak_summary.get("Howell_overlapping_alt_best_score"),
            ),
            "Howell_overlapping_alt_best_shift_nt": _coalesce_task_metric(
                task.get("Howell_overlapping_alt_best_shift_nt"),
                additional_peak_summary.get("Howell_overlapping_alt_best_shift_nt"),
            ),
            "Howell_crowding_window_count": _coalesce_task_metric(
                task.get("Howell_crowding_window_count"),
                browser_trace_context.get("Howell_crowding_window_count"),
            ),
            "Howell_crowding_best_score": _coalesce_task_metric(
                task.get("Howell_crowding_best_score"),
                browser_trace_context.get("Howell_crowding_best_score"),
            ),
            "Howell_crowding_score_gap": _coalesce_task_metric(
                task.get("Howell_crowding_score_gap"),
                browser_trace_context.get("Howell_crowding_score_gap"),
            ),
            "Peak_Howell_score": task.get("Peak_Howell_score"),
            "Howell_exact_relaxed_ratio": task.get("Howell_exact_relaxed_ratio"),
            "Howell_strict_relaxed_ratio": task.get("Howell_strict_relaxed_ratio"),
            "secondary_peak_ratio": task.get("secondary_peak_ratio"),
            "final_class": task.get("final_class"),
            "report_label": task.get("report_label"),
            "qc_reason": task.get("qc_reason"),
        },
        "phasiRNA_rows": export_rows,
    }


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
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    if task.get("plot_kind") == "placeholder":
        return _write_placeholder_plot(
            plot_path,
            task["title_text"],
            task.get("message_text", "No plot payload was generated."),
        )

    phase_value = int(task["phase"])
    abundance_rows = task.get("abundance_rows", [])
    howell_rows = task.get("howell_rows", [])
    layout = dict(LOCUS_LAYOUT)

    fig = plt.figure(figsize=layout["figsize"])
    ax_abun = fig.add_axes(
        [
            layout["main_left"],
            layout["abun_bottom"],
            layout["main_right"] - layout["main_left"],
            layout["abun_height"],
        ]
    )
    ax_howell = fig.add_axes(
        [
            layout["main_left"],
            layout["howell_bottom"],
            layout["main_right"] - layout["main_left"],
            layout["howell_height"],
        ],
        sharex=ax_abun,
    )
    axes = [ax_abun, ax_howell]
    fig.patch.set_facecolor("white")
    axes[0].set_facecolor("#F1F1F1")
    axes[1].set_facecolor("#F1F1F1")
    _draw_detachable_strip(fig, task, layout)
    _draw_detachable_separator(fig, layout)

    fig.suptitle(
        task["title_text"],
        fontsize=12,
        y=layout["title_y"],
        x=(float(layout["main_left"]) + float(layout["main_right"])) / 2.0,
    )
    _add_grouped_legends(
        fig,
        phase_value,
        main_left=float(layout["main_left"]),
        main_right=float(layout["main_right"]),
        legend_y=float(layout["legend_y"]),
        plot_mode=_current_locus_plot_mode(),
        alternative_legend_groups=list((task.get("alternative_legend_groups") or {}).values()),
    )

    _draw_centerline(axes[0])
    _draw_centerline(axes[1])
    _draw_strand_guides(axes[0], task.get("guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[0], task.get("guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)
    _draw_strand_guides(axes[1], task.get("guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[1], task.get("guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)
    _draw_strand_guides(axes[0], task.get("alternative_guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[0], task.get("alternative_guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)
    _draw_strand_guides(axes[1], task.get("alternative_guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[1], task.get("alternative_guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)

    for row in abundance_rows:
        axes[0].scatter(
            row["x"],
            row["y"],
            s=18,
            marker=row.get("marker", "D"),
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

    for row in task.get("alternative_howell_overlay_rows", []) or []:
        axes[1].scatter(
            row["x"],
            row["y"],
            s=row["size"],
            marker="o",
            facecolors=row["facecolor"],
            edgecolors=row["edgecolor"],
            linewidths=row["linewidth"],
            alpha=row["alpha"],
            zorder=5,
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

    axes[0].set_xlim(float(task["xmin"]) - float(task["xpad"]), float(task["xmax"]) + float(task["xpad"]))
    axes[0].set_ylim(-float(task["abun_ylim"]), float(task["abun_ylim"]))
    axes[1].set_ylim(-float(task["score_ylim"]), float(task["score_ylim"]))

    axes[0].text(0.01, 0.92, "+", transform=axes[0].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[0].text(0.01, 0.08, "-", transform=axes[0].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[1].text(0.01, 0.92, "+", transform=axes[1].transAxes, fontsize=9, fontweight="bold", color="#444444")
    axes[1].text(0.01, 0.08, "-", transform=axes[1].transAxes, fontsize=9, fontweight="bold", color="#444444")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, linestyle="-", zorder=0)

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _write_phasirna_export(export_path: str, export_rows) -> int:
    export_path = os.path.abspath(os.path.expanduser(str(export_path)))
    parent = os.path.dirname(export_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    export_df = pd.DataFrame(export_rows or [], columns=PHASIRNA_EXPORT_COLUMNS)
    export_df.to_csv(export_path, sep="\t", index=False)
    return int(len(export_df))


def _raise_parallel_failures(results, *, stage_name: str) -> None:
    failures = [res for res in (results or []) if isinstance(res, RuntimeError)]
    if not failures:
        return
    first = failures[0]
    raise RuntimeError(f"{stage_name} failed for {len(failures)} task(s). First error:\n{first}")


def _plot_bucket_specs(phase_value: int, method_name: str) -> list[dict]:
    return [
        {
            "final_class": "PHAS",
            "display_class": "PHAS",
            "plot_dir": f"{phase_value}_{method_name}_PHAS_locus_plots",
            "export_name": f"{phase_value}_{method_name}_phasiRNAs.tsv",
        },
        {
            "final_class": "PHAS-like",
            "display_class": "PHAS-like",
            "plot_dir": f"{phase_value}_{method_name}_PHAS_like_locus_plots",
            "export_name": f"{phase_value}_{method_name}_PHAS_like_phasiRNAs.tsv",
        },
    ]


def _format_locus_title(alib_value: str, identifier_value: str, phase_value: int, display_class: str) -> str:
    suffix = f"{phase_value}-{_format_phas_text(display_class)}"
    return f"{alib_value} | {identifier_value} | {suffix}"


def write_individual_phas_locus_plots(
    method_name: str,
    labeled_features: pd.DataFrame,
    clusters_data: pd.DataFrame,
    *,
    job_outdir: str | None = None,
    job_phase: str | int | None = None,
) -> None:
    phase_value = int(job_phase) if job_phase is not None else int(getattr(rt, "phase", 21))
    bucket_specs = _plot_bucket_specs(phase_value, method_name)
    final_class_series = labeled_features.get("final_class", labeled_features.get("label", pd.Series(dtype=str))).astype(str)

    if not final_class_series.isin(["PHAS", "PHAS-like"]).any():
        for spec in bucket_specs:
            shutil.rmtree(_join_outdir(job_outdir, spec["plot_dir"]), ignore_errors=True)
            export_path = _join_outdir(job_outdir, spec["export_name"])
            _write_phasirna_export(export_path, [])
            print(f"[INFO] Wrote 0 phase-length in-register phasiRNA row(s) to {export_path}")
        print("[INFO] No PHAS or PHAS-like calls were detected; skipping individual locus plots.")
        return

    print("#### Plotting individual PHAS and PHAS-like loci ######")

    call_cols = ["identifier", "alib", "cID"]
    missing_cols = [col for col in call_cols if col not in labeled_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for locus plots: {missing_cols}")

    cluster_keep_cols = ["clusterID", "identifier", "alib", "pos", "abun", "len", "strand", "tag_seq"]
    if "hits" in clusters_data.columns:
        cluster_keep_cols.append("hits")
    missing_cluster_cols = [col for col in cluster_keep_cols if col not in clusters_data.columns and col != "hits"]
    if missing_cluster_cols:
        raise ValueError(f"clusters_data missing required columns for locus plots: {missing_cluster_cols}")

    trimmed_clusters = clusters_data[cluster_keep_cols].copy()
    grouped_by_cid = {}
    grouped_by_identifier = {}
    for (cid_value, alib_value), subdf in trimmed_clusters.groupby(["clusterID", "alib"], sort=False):
        grouped_by_cid[(str(cid_value).strip(), str(alib_value).strip())] = subdf.copy()
    for (identifier_value, alib_value), subdf in trimmed_clusters.groupby(["identifier", "alib"], sort=False):
        grouped_by_identifier[(str(identifier_value).strip(), str(alib_value).strip())] = subdf.copy()

    for spec in bucket_specs:
        final_plot_dir = _join_outdir(job_outdir, spec["plot_dir"])
        phasirna_out = _join_outdir(job_outdir, spec["export_name"])
        bucket_calls = labeled_features.loc[final_class_series == spec["final_class"]].copy()
        if bucket_calls.empty:
            shutil.rmtree(final_plot_dir, ignore_errors=True)
            _write_phasirna_export(phasirna_out, [])
            print(f"[INFO] Wrote 0 phase-length in-register phasiRNA row(s) to {phasirna_out}")
            continue

        bucket_calls = bucket_calls.drop_duplicates(subset=call_cols, keep="first").reset_index(drop=True)
        raw_tasks = []
        for row in bucket_calls.itertuples(index=False):
            identifier_value = str(getattr(row, "identifier")).strip()
            alib_value = str(getattr(row, "alib")).strip()
            cid_value = str(getattr(row, "cID")).strip()

            cluster_df = grouped_by_cid.get((cid_value, alib_value))
            if cluster_df is None:
                cluster_df = grouped_by_identifier.get((identifier_value, alib_value))

            raw_tasks.append(
                {
                    "filename": f"{_sanitize_plot_name(alib_value)}__{_sanitize_plot_name(identifier_value)}.png",
                    "title_text": _format_locus_title(
                        alib_value,
                        identifier_value,
                        phase_value,
                        spec["display_class"],
                    ),
                    "identifier_text": identifier_value,
                    "cid_value": cid_value,
                    "alib_value": alib_value,
                    "phase": int(phase_value),
                    "Howell_exact_support_score": getattr(row, "Howell_exact_support_score", np.nan),
                    "Peak_Howell_score": getattr(row, "Peak_Howell_score", np.nan),
                    "Howell_ambiguity_count": getattr(row, "Howell_ambiguity_count", np.nan),
                    "Howell_alt_register_count": getattr(row, "Howell_alt_register_count", np.nan),
                    "Howell_overlap_margin": getattr(row, "Howell_overlap_margin", np.nan),
                    "Howell_extension_window_count": getattr(row, "Howell_extension_window_count", np.nan),
                    "Howell_extension_span_nt": getattr(row, "Howell_extension_span_nt", np.nan),
                    "Howell_origin_window_count": getattr(row, "Howell_origin_window_count", np.nan),
                    "Howell_origin_frame_count": getattr(row, "Howell_origin_frame_count", np.nan),
                    "Howell_origin_margin": getattr(row, "Howell_origin_margin", np.nan),
                    "Howell_origin_class": getattr(row, "Howell_origin_class", np.nan),
                    "Howell_additional_peak_count": getattr(row, "Howell_additional_peak_count", np.nan),
                    "Howell_additional_peak_best_score": getattr(row, "Howell_additional_peak_best_score", np.nan),
                    "Howell_overlapping_alt_count": getattr(row, "Howell_overlapping_alt_count", np.nan),
                    "Howell_overlapping_alt_best_score": getattr(row, "Howell_overlapping_alt_best_score", np.nan),
                    "Howell_overlapping_alt_best_shift_nt": getattr(row, "Howell_overlapping_alt_best_shift_nt", np.nan),
                    "Howell_crowding_window_count": getattr(row, "Howell_crowding_window_count", np.nan),
                    "Howell_crowding_best_score": getattr(row, "Howell_crowding_best_score", np.nan),
                    "Howell_crowding_score_gap": getattr(row, "Howell_crowding_score_gap", np.nan),
                    "Howell_exact_relaxed_ratio": getattr(row, "Howell_exact_relaxed_ratio", np.nan),
                    "Howell_strict_relaxed_ratio": getattr(row, "Howell_strict_relaxed_ratio", np.nan),
                    "secondary_peak_ratio": getattr(row, "secondary_peak_ratio", np.nan),
                    "final_class": getattr(row, "final_class", getattr(row, "label", "non-PHAS")),
                    "report_label": getattr(row, "report_label", getattr(row, "label", "non-PHAS")),
                    "qc_reason": getattr(row, "qc_reason", ""),
                    "cluster_rows": [] if cluster_df is None else cluster_df.to_dict("records"),
                }
            )

        if not raw_tasks:
            shutil.rmtree(final_plot_dir, ignore_errors=True)
            _write_phasirna_export(phasirna_out, [])
            print(f"[INFO] Wrote 0 phase-length in-register phasiRNA row(s) to {phasirna_out}")
            continue

        strategy = _resolve_plot_staging_strategy(final_plot_dir)
        activated_plan = _activate_plot_staging(final_plot_dir, strategy)
        _persist_plot_staging_plan(activated_plan)

        if activated_plan["mode"] == "local":
            print(
                f"[INFO] Plot staging mode={activated_plan['mode']} "
                f"(scratch={activated_plan['staging_root']}, final={activated_plan['final_plot_dir']})."
            )
        elif strategy["requested_mode"] == "local" and not strategy.get("staging_root"):
            print("[WARN] Local plot staging was requested but no writable scratch directory was found; writing plots directly.")

        tasks = []
        for task in raw_tasks:
            payload = dict(task)
            payload["plot_path"] = os.path.join(activated_plan["plot_dir"], task["filename"])
            tasks.append(payload)

        analysis_worker_cap = _resolve_locus_plot_worker_cap(len(tasks), direct_remote=False)
        prepared_results = run_parallel_with_progress(
            _analyze_single_locus,
            tasks,
            desc=f"Preparing {spec['display_class']} locus plot data",
            min_chunk=1,
            batch_factor=1.0,
            unit="file",
            kind="compute",
            initial_worker_cap=analysis_worker_cap,
            max_worker_cap=analysis_worker_cap,
            adaptive_recovery=False,
        )
        _raise_parallel_failures(prepared_results, stage_name=f"{spec['display_class']} locus analysis")

        plot_payloads = []
        export_rows = []
        for res in prepared_results or []:
            if not isinstance(res, dict):
                continue
            plot_payloads.append(res.get("plot_payload", {}))
            export_rows.extend(res.get("phasiRNA_rows", []) or [])

        export_count = _write_phasirna_export(phasirna_out, export_rows)
        print(f"[INFO] Wrote {export_count} phase-length in-register phasiRNA row(s) to {phasirna_out}")

        render_worker_cap = _resolve_locus_plot_worker_cap(
            len(plot_payloads),
            direct_remote=activated_plan["mode"] == "direct" and activated_plan.get("is_remote_output", False),
        )

        try:
            render_results = run_parallel_with_progress(
                _write_single_locus_plot,
                plot_payloads,
                desc=f"Writing {spec['display_class']} locus plots",
                min_chunk=1,
                batch_factor=1.0,
                unit="file",
                kind="plot",
                initial_worker_cap=render_worker_cap,
                max_worker_cap=render_worker_cap,
                adaptive_recovery=False,
            )
            _raise_parallel_failures(render_results, stage_name=f"{spec['display_class']} locus plot rendering")

            if activated_plan["mode"] == "local":
                _finalize_staged_plot_dir(activated_plan["plot_dir"], activated_plan["final_plot_dir"])
        finally:
            if activated_plan.get("staging_run_dir"):
                shutil.rmtree(activated_plan["staging_run_dir"], ignore_errors=True)

        print(
            f"[INFO] Wrote {len(plot_payloads)} individual {spec['display_class']} locus plot(s) "
            f"to {activated_plan['final_plot_dir']}"
        )
