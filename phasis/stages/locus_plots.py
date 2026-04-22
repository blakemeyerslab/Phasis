from __future__ import annotations

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
from matplotlib.ticker import FuncFormatter

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.stages import feature_assembly as st_feat

MAX_LOCUS_PLOT_WORKERS = 10
REMOTE_DIRECT_LOCUS_PLOT_WORKER_CAP = 4
GUIDE_CYCLES = 10
PLOT_STAGING_CHOICES = frozenset({"auto", "local", "direct"})
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
                "strand": _normalize_strand_code(strand_code),
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


def _add_grouped_legends(fig, phase_value: int, *, main_left: float, main_right: float, legend_y: float) -> None:
    legend_groups = _build_plot_legend_groups(phase_value)
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
) -> list[dict]:
    rows = []
    for row in rows_df.itertuples(index=False):
        rows.append(
            {
                "identifier": identifier_text,
                "cID": cid_value,
                "alib": alib_value,
                "phase": int(phase_value),
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
            )
        )

    return export_rows


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


def _build_ambiguity_sidebar_payload(task: dict) -> dict:
    exact_support = _clean_ambiguity_float(task.get("Howell_exact_support_score"))
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

    return {
        "exact_support": "NA" if exact_support is None else f"{exact_support:.2f}",
        "origin_class": origin_class,
        "extension_window_count": "NA" if extension_window_count is None else str(extension_window_count),
        "extension_span_nt": "NA" if extension_span_nt is None else str(extension_span_nt),
        "origin_window_count": "NA" if origin_window_count is None else str(origin_window_count),
        "origin_frame_count": "NA" if origin_frame_count is None else str(origin_frame_count),
        "origin_margin": "NA" if origin_margin is None else f"{origin_margin:.2f}",
        "additional_peak_count": "NA" if additional_peak_count is None else str(additional_peak_count),
        "additional_peak_best_score": "NA" if additional_peak_best_score is None else f"{additional_peak_best_score:.2f}",
        "additional_peak_cutoff": _howell_peak_cutoff_text(),
        "raw_overlap_count": "NA" if overlap_count is None else str(overlap_count),
        "raw_alt_register_count": "NA" if alt_register_count is None else str(alt_register_count),
        "raw_overlap_margin": "NA" if overlap_margin is None else f"{overlap_margin:.2f}",
    }


def _build_ambiguity_sidebar_note(task: dict) -> str:
    exact_support = _clean_ambiguity_float(task.get("Howell_exact_support_score"))
    if exact_support is None or exact_support <= 0.0:
        return "Exact-only interpretation unavailable.\nRelaxed HPSP lacks exact support."
    return "Howell panel is relaxed.\nOffsets are excluded here."


def _draw_strip_line(
    ax,
    y_value: float,
    text_value: str,
    *,
    fontsize: float,
    fontweight: str = "normal",
    color: str = "#333333",
    line_gap: float = 0.037,
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

    payload = _build_ambiguity_sidebar_payload(task)
    class_line = f"Class: {payload['origin_class']}"
    if len(class_line) > 23:
        class_line = f"Class:\n{_wrap_strip_text(payload['origin_class'], width=21)}"

    y_cursor = 0.95
    y_cursor = _draw_strip_line(ax, y_cursor, "Exact-only Howell", fontsize=8.6, fontweight="bold")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Support: {payload['exact_support']}", fontsize=8.0, fontweight="bold")
    y_cursor = _draw_strip_line(ax, y_cursor, class_line, fontsize=7.6, fontweight="bold", paragraph_gap=0.022)

    y_cursor = _draw_strip_line(ax, y_cursor, "Coherent extension", fontsize=8.0, fontweight="bold", color="#444444")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Windows: {payload['extension_window_count']}", fontsize=7.2, color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Span: {payload['extension_span_nt']} nt", fontsize=7.2, color="#555555", paragraph_gap=0.022)

    y_cursor = _draw_strip_line(ax, y_cursor, "Origin ambiguity", fontsize=8.0, fontweight="bold", color="#444444")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Windows: {payload['origin_window_count']}", fontsize=7.2, color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Frames: {payload['origin_frame_count']}", fontsize=7.2, color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Margin: {payload['origin_margin']}", fontsize=7.2, color="#555555", paragraph_gap=0.022)

    y_cursor = _draw_strip_line(ax, y_cursor, "Other local peaks", fontsize=8.0, fontweight="bold", color="#444444")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Count: {payload['additional_peak_count']}", fontsize=7.2, color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Best: {payload['additional_peak_best_score']}", fontsize=7.2, color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Cutoff: {payload['additional_peak_cutoff']}", fontsize=7.2, color="#555555", paragraph_gap=0.022)

    y_cursor = _draw_strip_line(ax, y_cursor, "Raw context", fontsize=7.8, fontweight="bold", color="#555555")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Near-tied: {payload['raw_overlap_count']}", fontsize=7.0, color="#666666")
    y_cursor = _draw_strip_line(ax, y_cursor, f"Alt regs: {payload['raw_alt_register_count']}", fontsize=7.0, color="#666666")
    _draw_strip_line(ax, y_cursor, f"Raw margin: {payload['raw_overlap_margin']}", fontsize=7.0, color="#666666")

    ax.text(
        0.07,
        0.02,
        _build_ambiguity_sidebar_note(task),
        fontsize=6.2,
        ha="left",
        va="bottom",
        color="#666666",
        transform=ax.transAxes,
        clip_on=True,
    )


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
            "plot_payload": _placeholder_payload(task, "No cluster rows were available for this PHAS call."),
            "phasiRNA_rows": [],
        }

    cluster_df = _prepare_cluster_df(cluster_rows)
    if cluster_df.empty:
        return {
            "plot_payload": _placeholder_payload(task, "The PHAS locus data were empty after reconstruction."),
            "phasiRNA_rows": [],
        }

    trace = st_feat.enumerate_relaxed_howell_trace(cluster_df, phase=phase_value)
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
    howell_rows_w, _hpsp_w, hpsp_row_w = _build_howell_rows(w_trace, phase_value, "w")
    howell_rows_c, _hpsp_c, hpsp_row_c = _build_howell_rows(c_trace, phase_value, "c")
    howell_rows = howell_rows_w + howell_rows_c
    guide_specs_w = _build_score_exact_guide_specs(hpsp_row_w, phase_value, w_trace, "w")
    guide_specs_c = _build_score_exact_guide_specs(hpsp_row_c, phase_value, c_trace, "c")
    additional_peak_summary = st_feat.summarize_relaxed_trace_subregions(
        trace,
        score_cutoff=float(getattr(rt, "min_Howell_score", 12.5) or 12.5),
    )

    base_positions_w, register_origin_w = _build_scored_register_positions(hpsp_row_w, phase_value, "w")
    base_positions_c, register_origin_c = _build_scored_register_positions(hpsp_row_c, phase_value, "c")
    extended_positions_w = _collect_extended_register_positions(w_trace, register_origin_w, base_positions_w, phase_value)
    extended_positions_c = _collect_extended_register_positions(c_trace, register_origin_c, base_positions_c, phase_value)

    export_rows = []
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
        )
    )

    xmin, xmax = _build_x_bounds(cluster_df, str(task["identifier_text"]), howell_rows, phase_value)
    span = max(float(xmax) - float(xmin), float(phase_value))
    xpad = max(float(phase_value) * 4.0, span * 0.06)

    max_abun = max([abs(row["y"]) for row in abundance_rows], default=1.0)
    max_score = max([abs(row["y"]) for row in howell_rows], default=1.0)
    abun_ylim = max(1.0, max_abun * 1.15)
    score_ylim = max(1.0, max_score * 1.20)

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
            "xmin": float(xmin),
            "xmax": float(xmax),
            "xpad": float(xpad),
            "abun_ylim": float(abun_ylim),
            "score_ylim": float(score_ylim),
            "Howell_exact_support_score": task.get("Howell_exact_support_score"),
            "Howell_ambiguity_count": task.get("Howell_ambiguity_count"),
            "Howell_alt_register_count": task.get("Howell_alt_register_count"),
            "Howell_overlap_margin": task.get("Howell_overlap_margin"),
            "Howell_extension_window_count": task.get("Howell_extension_window_count"),
            "Howell_extension_span_nt": task.get("Howell_extension_span_nt"),
            "Howell_origin_window_count": task.get("Howell_origin_window_count"),
            "Howell_origin_frame_count": task.get("Howell_origin_frame_count"),
            "Howell_origin_margin": task.get("Howell_origin_margin"),
            "Howell_origin_class": task.get("Howell_origin_class"),
            "Howell_additional_peak_count": _coalesce_task_metric(
                task.get("Howell_additional_peak_count"),
                additional_peak_summary.get("Howell_additional_peak_count"),
            ),
            "Howell_additional_peak_best_score": _coalesce_task_metric(
                task.get("Howell_additional_peak_best_score"),
                additional_peak_summary.get("Howell_additional_peak_best_score"),
            ),
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
    )

    _draw_centerline(axes[0])
    _draw_centerline(axes[1])
    _draw_strand_guides(axes[0], task.get("guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[0], task.get("guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)
    _draw_strand_guides(axes[1], task.get("guide_specs_w", []), _phase_color_hex(phase_value), upper_half=True)
    _draw_strand_guides(axes[1], task.get("guide_specs_c", []), _phase_color_hex(phase_value), upper_half=False)

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
    final_plot_dir = _join_outdir(job_outdir, plot_dir_name)
    phasirna_out = _join_outdir(job_outdir, f"{phase_value}_{method_name}_phasiRNAs.tsv")

    phas_calls = labeled_features.loc[labeled_features["label"] == "PHAS"].copy()
    if phas_calls.empty:
        shutil.rmtree(final_plot_dir, ignore_errors=True)
        _write_phasirna_export(phasirna_out, [])
        print("[INFO] No PHAS calls were detected; skipping individual locus plots.")
        print(f"[INFO] Wrote 0 phase-length in-register phasiRNA row(s) to {phasirna_out}")
        return

    print("#### Plotting individual PHAS loci ######")

    call_cols = ["identifier", "alib", "cID"]
    missing_cols = [col for col in call_cols if col not in phas_calls.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for locus plots: {missing_cols}")

    cluster_keep_cols = ["clusterID", "identifier", "alib", "pos", "abun", "len", "strand", "tag_seq"]
    if "hits" in clusters_data.columns:
        cluster_keep_cols.append("hits")
    missing_cluster_cols = [col for col in cluster_keep_cols if col not in clusters_data.columns and col != "hits"]
    if missing_cluster_cols:
        raise ValueError(f"clusters_data missing required columns for locus plots: {missing_cluster_cols}")

    strategy = _resolve_plot_staging_strategy(final_plot_dir)
    activated_plan = None

    trimmed_clusters = clusters_data[cluster_keep_cols].copy()
    grouped_by_cid = {}
    grouped_by_identifier = {}
    for (cid_value, alib_value), subdf in trimmed_clusters.groupby(["clusterID", "alib"], sort=False):
        grouped_by_cid[(str(cid_value).strip(), str(alib_value).strip())] = subdf.copy()
    for (identifier_value, alib_value), subdf in trimmed_clusters.groupby(["identifier", "alib"], sort=False):
        grouped_by_identifier[(str(identifier_value).strip(), str(alib_value).strip())] = subdf.copy()

    phas_calls = phas_calls.drop_duplicates(subset=call_cols, keep="first").reset_index(drop=True)
    raw_tasks = []
    for row in phas_calls.itertuples(index=False):
        identifier_value = str(getattr(row, "identifier")).strip()
        alib_value = str(getattr(row, "alib")).strip()
        cid_value = str(getattr(row, "cID")).strip()

        cluster_df = grouped_by_cid.get((cid_value, alib_value))
        if cluster_df is None:
            cluster_df = grouped_by_identifier.get((identifier_value, alib_value))

        raw_tasks.append(
            {
                "filename": f"{_sanitize_plot_name(alib_value)}__{_sanitize_plot_name(identifier_value)}.png",
                "title_text": f"{alib_value} | {identifier_value} | {phase_value}-$\\it{{PHAS}}$",
                "identifier_text": identifier_value,
                "cid_value": cid_value,
                "alib_value": alib_value,
                "phase": int(phase_value),
                "Howell_exact_support_score": getattr(row, "Howell_exact_support_score", np.nan),
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
                "cluster_rows": [] if cluster_df is None else cluster_df.to_dict("records"),
            }
        )

    if not raw_tasks:
        shutil.rmtree(final_plot_dir, ignore_errors=True)
        _write_phasirna_export(phasirna_out, [])
        print("[INFO] No PHAS locus plot tasks were produced.")
        print(f"[INFO] Wrote 0 phase-length in-register phasiRNA row(s) to {phasirna_out}")
        return

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
        desc="Preparing PHAS locus plot data",
        min_chunk=1,
        batch_factor=1.0,
        unit="file",
        kind="compute",
        initial_worker_cap=analysis_worker_cap,
        max_worker_cap=analysis_worker_cap,
        adaptive_recovery=False,
    )
    _raise_parallel_failures(prepared_results, stage_name="PHAS locus analysis")

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
            desc="Writing PHAS locus plots",
            min_chunk=1,
            batch_factor=1.0,
            unit="file",
            kind="plot",
            initial_worker_cap=render_worker_cap,
            max_worker_cap=render_worker_cap,
            adaptive_recovery=False,
        )
        _raise_parallel_failures(render_results, stage_name="PHAS locus plot rendering")

        if activated_plan["mode"] == "local":
            _finalize_staged_plot_dir(activated_plan["plot_dir"], activated_plan["final_plot_dir"])
    finally:
        if activated_plan.get("staging_run_dir"):
            shutil.rmtree(activated_plan["staging_run_dir"], ignore_errors=True)

    print(f"[INFO] Wrote {len(plot_payloads)} individual PHAS locus plot(s) to {activated_plan['final_plot_dir']}")
