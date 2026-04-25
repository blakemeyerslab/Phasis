"""phasis.stages.output

Output writer + plotting stage extracted from legacy.py.

Goals:
- No behavior drift: writes the same TSV/GFF outputs and runs the same plots.
- Spawn-safe on macOS: plotting happens in a dedicated pool (kind="plot") and
  workers can rehydrate outdir/phase via runtime snapshot.
- No nested functions; no imports inside functions.
"""

from __future__ import annotations

import os
import re
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.parallel import make_pool
from phasis import ids as st_ids

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Module globals used by plotting and GFF writing (set by finalize_and_write_results or _plot_wrapper)
outdir: str | None = None
phase: str | int | None = None


def _join_outdir(dirpath: str | None, name: str) -> str:
    if not dirpath:
        return name
    return dirpath + name if dirpath.endswith("/") else dirpath + "/" + name


def _fallback_series(nrows: int) -> pd.Series:
    return pd.Series([np.nan] * int(nrows))


def _fallback_text_series(nrows: int, value: str = "") -> pd.Series:
    return pd.Series([value] * int(nrows), dtype="object")


def _format_phas_plot_text(text_value: str) -> str:
    text_local = str(text_value or "")
    text_local = text_local.replace("non-PHAS", "__NON_PHAS__")
    text_local = text_local.replace("PHAS", r"$\it{PHAS}$")
    text_local = text_local.replace("__NON_PHAS__", r"non-$\it{PHAS}$")
    return text_local


def _set_colorbar_ticks_and_labels(cbar, ticks, labels) -> None:
    ticks_list = list(ticks)
    labels_list = list(labels)
    cbar.set_ticks(ticks_list)
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks_list))
    cbar.set_ticklabels(labels_list)


def _compute_finite_color_scale(data, *, fallback_min: float = 0.0, fallback_max: float = 1.0):
    values = np.asarray(data, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float(fallback_min), float(fallback_max)

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.01, 1e-6)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _build_colorbar_ticks_and_labels(vmin: float, vmax: float, *, nticks: int = 4):
    if nticks <= 1:
        ticks = [vmin]
    else:
        ticks = np.linspace(vmin, vmax, nticks).tolist()
    labels = [f"{tick:.1f}" for tick in ticks]
    return ticks, labels


def _build_discrete_chromosome_cmap(unique_chromosomes):
    nchrom = max(int(len(unique_chromosomes)), 1)
    base_cmap = plt.get_cmap("Greys")
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, nchrom))
    if nchrom == 1:
        return LinearSegmentedColormap.from_list("CustomGreysSingle", [chrom_cmap[0], chrom_cmap[0]], 2)
    return LinearSegmentedColormap.from_list("CustomGreys", chrom_cmap, nchrom)


def _format_runtime_parameter_lines() -> list[str]:
    libs_value = getattr(rt, "libs", None)
    if isinstance(libs_value, (list, tuple)):
        libs_list = [str(x) for x in libs_value]
    elif libs_value is None:
        libs_list = []
    else:
        libs_list = [str(libs_value)]

    class_files = getattr(rt, "class_cluster_file", None)
    if isinstance(class_files, (list, tuple)):
        class_files_text = ", ".join(str(x) for x in class_files) if class_files else "None"
    elif class_files:
        class_files_text = str(class_files)
    else:
        class_files_text = "None"

    libs_text = ", ".join(libs_list) if libs_list else "None"

    threshold_parts = [
        f"mindepth={getattr(rt, 'mindepth', 'NA')}",
        f"maxhits={getattr(rt, 'maxhits', 'NA')}",
        f"mismat={getattr(rt, 'mismat', 'NA')}",
        f"uniqueRatioCut={getattr(rt, 'uniqueRatioCut', 'NA')}",
        f"clustbuffer={getattr(rt, 'clustbuffer', 'NA')}",
        f"minClusterLength={getattr(rt, 'minClusterLength', 'NA')}",
        f"window_len={getattr(rt, 'window_len', 'NA')}",
        f"sliding={getattr(rt, 'sliding', 'NA')}",
        f"phasisScoreCutoff={getattr(rt, 'phasisScoreCutoff', 'NA')}",
        f"min_Howell_score={getattr(rt, 'min_Howell_score', 'NA')}",
        f"max_complexity={getattr(rt, 'max_complexity', 'NA')}",
        f"norm={getattr(rt, 'norm', 'NA')}",
    ]
    if getattr(rt, "norm", False):
        threshold_parts.append(f"norm_factor={getattr(rt, 'norm_factor', 'NA')}")
    threshold_parts.extend(
        [
            f"cores={getattr(rt, 'cores', 'NA')}",
            f"ncores={getattr(rt, 'ncores', 'NA')}",
        ]
    )

    return [
        (
            "  - Parameters: "
            f"steps={getattr(rt, 'steps', 'NA')}, "
            f"phase={getattr(rt, 'phase', 'NA')}, "
            f"classifier={getattr(rt, 'classifier', 'NA')}, "
            f"concat_libs={getattr(rt, 'concat_libs', 'NA')}, "
            f"outdir={getattr(rt, 'outdir', 'NA')}"
        ),
        (
            "  - Inputs: "
            f"reference={getattr(rt, 'reference', 'NA')}, "
            f"libs_count={len(libs_list)}, "
            f"libs={libs_text}, "
            f"class_cluster_file={class_files_text}"
        ),
        (
            "  - Thresholds/resources: "
            + ", ".join(threshold_parts)
        ),
    ]


def _print_final_detection_summary(phas_df: pd.DataFrame, *, wrote_line: str | None = None) -> None:
    unique_loci = int(phas_df['identifier'].nunique()) if 'identifier' in phas_df.columns else 0
    total_detections = int(len(phas_df))
    phase_label = phase if phase is not None else getattr(rt, "phase", "NA")

    summary_lines = []
    if wrote_line:
        summary_lines.append(wrote_line)

    summary_lines.append(
        f"  - Detected {unique_loci} unique {phase_label}-PHAS loci candidates across {total_detections} {phase_label}-PHAS library-specific detections."
    )

    if total_detections > 0 and 'alib' in phas_df.columns:
        per_lib = phas_df['alib'].astype(str).value_counts().sort_index()
        per_lib_text = ", ".join(f"{lib}={int(count)}" for lib, count in per_lib.items())
        summary_lines.append(f"  - Library detections: {per_lib_text}")
    else:
        summary_lines.append("  - Library detections: none")

    summary_lines.extend(_format_runtime_parameter_lines())

    print("")
    print("\n\n".join(summary_lines))


def _parse_identifiers_and_alib(features: pd.DataFrame, job_phase: str | int | None):
    """
    Return achr, start, end, cleaned alib arrays from features['identifier']/['alib'].

    If 'identifier' is not in 'chr:start..end' form, try to resolve via rt.mergedClusterReverse
    using the row's 'cID'. Falls back to blanks to avoid IndexError.
    """
    # Ensure reverse map exists (spawn-safe)
    rmap = getattr(rt, "mergedClusterReverse", None)
    if not isinstance(rmap, dict) or not rmap:
        try:
            st_ids.ensure_mergedClusterDict(str(job_phase) if job_phase is not None else None)
        except Exception:
            pass
        rmap = getattr(rt, "mergedClusterReverse", {}) or {}

    achr, start, end = [], [], []

    # iterate row-wise so we can look at both identifier and cID
    cids = features.get("cID", pd.Series([None] * len(features)))
    for id_str, cID in zip(features["identifier"].astype(str), cids):
        u = None
        if ":" in id_str and ".." in id_str:
            u = id_str
        else:
            # try reverse map by cID first, then by the identifier string itself
            key = None
            if cID is not None and str(cID) != "nan":
                key = str(cID).strip()
            if key and key in rmap:
                u = rmap.get(key)
            elif id_str in rmap:
                u = rmap.get(id_str)

        if u and ":" in u and ".." in u:
            left, right = u.split(":", 1)
            achr.append(left)
            s_val, e_val = right.split("..", 1)
            start.append(s_val)
            end.append(e_val)
        else:
            achr.append("")
            start.append("")
            end.append("")

    # keep alib as-is unless it ends with ".{phase}-PHAS.candidate"
    alib_src = features["alib"].astype(str).tolist()
    alib_ids = [re.sub(rf"\.{re.escape(str(job_phase))}-PHAS\.candidate$", "", x) for x in alib_src]
    return achr, start, end, alib_ids


def format_attributes(row):
    attributes = {
        'id':row['identifier'],
        'complexity': row['complexity'],
        'strand_bias': row['strand_bias'],
        'log_clust_len_norm_counts': row['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': row['ratio_abund_len_phase']
    }
    attr_str = ';'.join([f"{key}={value}" for key, value in attributes.items()])
    return attr_str


def write_gff(phasis_result_df,gff_filename):
    unique_df = phasis_result_df.groupby('identifier').first().reset_index()
    # Write to GFF3 file
    
    with open(gff_filename, 'w') as gff_file:
        for index, row in unique_df.iterrows():
            seq_id = row['achr']
            source = 'Phasis'
            feature_type = f'{phase}-PHAS'
            start_pos = str(row['start'])
            end_pos = str(row['end'])
            score = str(row['phasis_score'])
            strand = '.'
            t_phase = '.'
            attributes = format_attributes(row)
            gff_line = f"{seq_id}\t{source}\t{feature_type}\t{start_pos}\t{end_pos}\t{score}\t{strand}\t{t_phase}\t{attributes}\n"
            gff_file.write(gff_line)

    return None



def _filter_plot_df(phasis_result_df):
    df = phasis_result_df
    if str(phase) == "24":
        ids_with_phas = set(df.loc[df["label"] == "PHAS", "identifier"].unique())
        df = df[df["identifier"].isin(ids_with_phas)].copy()
    return df


def _empty_plot_placeholder(outfile, message):
    f, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(0.01, 0.5, _format_phas_plot_text(message), fontsize=12)
    f.savefig(outfile, dpi=300)
    plt.close(f)
    return None


def _chrom_sort_key(chrom_value):
    chrom_text = str(chrom_value)
    return (0, int(chrom_text)) if chrom_text.isdigit() else (1, chrom_text)



def _identifier_sort_tuple(identifier):
    text = str(identifier)
    chrom = text.split(":", 1)[0]
    start_val = float('inf')
    end_val = float('inf')

    if ":" in text and ".." in text:
        try:
            coords = text.split(":", 1)[1]
            start_text, end_text = coords.split("..", 1)
            start_val = int(start_text)
            end_val = int(end_text)
        except Exception:
            start_val = float('inf')
            end_val = float('inf')

    return (_chrom_sort_key(chrom), start_val, end_val, text)



def _sort_heatmap_frame(data):
    sorted_identifiers = sorted(list(data.index), key=_identifier_sort_tuple)
    data = data.loc[sorted_identifiers]
    chrom_data = [str(identifier).split(":", 1)[0] for identifier in data.index]
    unique_chromosomes = sorted(np.unique(chrom_data), key=_chrom_sort_key)
    return data, chrom_data, unique_chromosomes


def _build_score_heatmap_data(df, value_column):
    all_libs = sorted(list(df["alib"].astype(str).unique()))
    identifiers = list(df["identifier"].astype(str).unique())

    data = pd.DataFrame(0.0, index=identifiers, columns=all_libs)
    phas_mask = pd.DataFrame(False, index=identifiers, columns=all_libs)

    it = tqdm(identifiers, desc="Processing Rows") if tqdm else identifiers
    for identifier in it:
        temp_rows = df[df["identifier"] == identifier]
        for alib in all_libs:
            subset = temp_rows[temp_rows["alib"] == alib]
            if subset.empty:
                continue
            if value_column in subset.columns and len(subset[value_column]) > 0:
                value = pd.to_numeric(subset[value_column], errors="coerce").iloc[0]
                if pd.notna(value):
                    data.loc[identifier, alib] = float(value)
            if "PHAS" in subset["label"].values:
                phas_mask.loc[identifier, alib] = True

    data, chrom_data, unique_chromosomes = _sort_heatmap_frame(data)
    phas_mask = phas_mask.loc[data.index, data.columns]
    return data, phas_mask, chrom_data, unique_chromosomes


def _draw_chromosome_bar(fig, ref_ax, heat_data, chrom_data, unique_chromosomes, left_pad=0.055):
    pos = ref_ax.get_position()
    bar_width = 0.02
    bar_left = max(0.02, pos.x0 - left_pad)
    cax = fig.add_axes([bar_left, pos.y0, bar_width, pos.height])

    chrom_cmap = _build_discrete_chromosome_cmap(unique_chromosomes)
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax.imshow(
        chrom_colors,
        cmap=chrom_cmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, 1, len(heat_data.index), 0),
    )
    cax.set_xticks([])
    cax.set_yticks([])

    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    cax.set_ylim(len(heat_data.index), 0)

    chrom_positions = np.array([
        np.mean(np.where(np.array(chrom_data) == chrom)[0]) + 0.5
        for chrom in unique_chromosomes
    ])
    cax.set_yticks(chrom_positions)
    cax.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha="center")
    cax.yaxis.set_tick_params(labelsize=16, pad=11)
    return cax


def _draw_phas_borders(ax, phas_mask, edgecolor="#D4AF37", linewidth=0.1):
    rows, cols = np.where(phas_mask.to_numpy(dtype=bool))
    for row_idx, col_idx in zip(rows, cols):
        ax.add_patch(
            Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                joinstyle="miter",
                zorder=10,
            )
        )


def _format_score_colorbar_ticks(max_value):
    safe_max = float(max_value) if pd.notna(max_value) else 0.0
    if safe_max <= 0:
        return [0.0, 1.0], ["0.0", "1.0"]
    ticks = [0.0, safe_max / 3.0, (2.0 * safe_max) / 3.0, safe_max]
    labels = [f"{tick:.1f}" for tick in ticks]
    return ticks, labels


def _plot_single_score_heatmap(ax, data, phas_mask, cmap, title_text):
    max_value = float(np.nanmax(data.to_numpy())) if data.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    norm = Normalize(vmin=0, vmax=max_value)

    heat = sns.heatmap(
        data,
        square=False,
        cmap=cmap,
        cbar=False,
        norm=norm,
        xticklabels=True,
        yticklabels=False,
        ax=ax,
    )
    #_draw_phas_borders(ax, phas_mask)
    ax.set_facecolor("#D9D9D9")

    ax.set_title(title_text, fontsize=13, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    return heat, norm, max_value


def _add_score_colorbar(fig, ax, cmap, norm, max_value, colorbar_label, side="left"):
    pos = ax.get_position()
    cbar_width = 0.014
    cbar_height = 0.10
    cbar_bottom = min(.98 - cbar_height, pos.y1 + 0.06)

    if side == "left":
        cbar_left = max(0.02, pos.x0 + 0.005)
    else:
        cbar_left = min(0.98 - cbar_width, pos.x1 - cbar_width - 0.005)

    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax,
        orientation="vertical"
    )
    ticks, labels = _format_score_colorbar_ticks(max_value)
    _set_colorbar_ticks_and_labels(cbar, ticks, labels)
    cbar.set_label(colorbar_label, rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def plot_howell_score_heat_maps(phasis_result_df, plot_type):
    print("#### Plotting Howell score heatmaps ######")

    global outdir, phase

    outfile = _join_outdir(outdir, f"{plot_type}_{phase}_Howell_scores.pdf")
    df = _filter_plot_df(phasis_result_df)

    if df.empty:
        return _empty_plot_placeholder(outfile, f"No {phase}-PHAS loci detected.")

    data_howell, phas_mask, chrom_data, unique_chromosomes = _build_score_heatmap_data(df, "Peak_Howell_score")
    data_howell_strict, phas_mask_strict, _, _ = _build_score_heatmap_data(df, "Peak_Howell_score_strict")

    purple_cmap = LinearSegmentedColormap.from_list("HowellPurple", ["#8C8C8C", "#6D28D9"], N=256)
    teal_cmap = LinearSegmentedColormap.from_list("HowellStrict", ["#8C8C8C", "#0F766E"], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
    fig.patch.set_facecolor("#D9D9D9")

    plt.subplots_adjust(left=0.18, right=0.97, top=0.72, bottom=0.1, wspace=0.18)

    _, norm_howell, max_howell = _plot_single_score_heatmap(
        axes[0],
        data_howell,
        phas_mask,
        purple_cmap,
        "Peak Howell score",
    )
    _, norm_howell_strict, max_howell_strict = _plot_single_score_heatmap(
        axes[1],
        data_howell_strict,
        phas_mask_strict,
        teal_cmap,
        "Peak Howell score (strict)",
    )

    _draw_chromosome_bar(fig, axes[0], data_howell, chrom_data, unique_chromosomes, left_pad=0.06)

    for ax in axes:
        ax.set_ylabel("")

    _add_score_colorbar(fig, axes[0], purple_cmap, norm_howell, max_howell, "Peak Howell score", side="left")
    _add_score_colorbar(fig, axes[1], teal_cmap, norm_howell_strict, max_howell_strict, "Peak Howell score (strict)", side="left")
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    return None

def plot_report_heat_map(phasis_result_df, plot_type):
    print("#### Plotting heatmap ######")

    global outdir, phase

    # Keep full library columns (even if no PHAS in some libs)
    all_libs = sorted(list(phasis_result_df["alib"].unique()))

    # For phase==24: drop loci (rows) that have no PHAS call in any library
    df = phasis_result_df
    if str(phase) == "24":
        ids_with_phas = set(df.loc[df["label"] == "PHAS", "identifier"].unique())
        df = df[df["identifier"].isin(ids_with_phas)].copy()

        if df.empty:
            # Nothing to plot; write a small placeholder PDF to avoid errors
            f, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")
            ax.text(0.01, 0.5, _format_phas_plot_text("No 24-PHAS loci detected."), fontsize=12)
            f.savefig(_join_outdir(outdir, f"{phase}_{plot_type}_PHAS.pdf"), dpi=300)
            plt.close(f)
            return

    data = pd.DataFrame(
        data=0.0,
        columns=all_libs,  # Sorted columns alphanumerically
        index=list(df["identifier"].unique())
    )

    it = tqdm(data.index, desc="Processing Rows") if tqdm else data.index

    # Iterate over rows and columns to fill heatmap data
    for i in it:
        tempRows = df[df["identifier"] == i]
        for j in data.columns:
            # Default: Not detected (0) for 21; for 24 keep-row, missing cells should appear as non-PHAS cluster (1)
            k = 1.0 if str(phase) == "24" else 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "non-PHAS" in subSetData["label"].values:
                    k = 1.0
                elif "PHAS" in subSetData["label"].values:
                    k = 2.0
            data.loc[i, j] = k

    # Extract chromosome data from the identifier
    chrom_data = [identifier.split(':')[0] for identifier in data.index]

    # Sort chromosomes numerically (1, 2, 3 ... not 1, 10, 2 ...)
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx]
    )

    # Reorder the DataFrame based on sorted chromosomes
    data = data.iloc[sorted_indices]

    # Re-extract the sorted chromosome data
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    # Create the heatmap figure
    f, ax = plt.subplots(figsize=(11, 11))

    # Define color map for the heatmap (exact original)
    colors = ["#C3D8EA", "#3662A5", "#C24F4E"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, len(colors))

    # Create the actual heatmap
    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, xticklabels=True, yticklabels=False, ax=ax)

    # Add a new axis for the chromosome bar to the left (exact original placement)
    cax = f.add_axes([0.17, 0.1, 0.02, 0.8])

    # Discrete grayscale colormap for chromosomes
    chrom_cmap = _build_discrete_chromosome_cmap(unique_chromosomes)
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax.imshow(
        chrom_colors,
        cmap=chrom_cmap,
        aspect="auto"
    )
    cax.set_xticks([])
    cax.set_yticks([])

    # Remove border
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    # IMPORTANT: for phase==24, do NOT force ylim; this avoids half-row offsets when few rows remain
    if str(phase) != "24":
        cax.set_ylim(len(data.index), 0)

    # Midpoints for chromosome labels (exact original logic)
    chrom_ranges = {}
    for chrom in unique_chromosomes:
        chrom_loci_indices = np.where(np.array(chrom_data) == chrom)[0]
        chrom_midpoint = (chrom_loci_indices[0] + chrom_loci_indices[-1]) / 2
        chrom_ranges[chrom] = chrom_midpoint

    midpoints = [chrom_ranges[chrom] for chrom in unique_chromosomes]
    cax.set_yticks(midpoints)
    cax.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha="center")
    cax.yaxis.set_tick_params(labelsize=16, pad=11)

    # Legend (exact original placement)
    cax2 = inset_axes(
        ax,
        width="2.5%",
        height="9%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 1.01, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cax2.spines["top"].set_visible(False)
    cax2.spines["right"].set_visible(False)
    cax2.spines["bottom"].set_visible(False)
    cax2.spines["left"].set_visible(False)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=2)), cax=cax2, orientation="vertical")
    _set_colorbar_ticks_and_labels(cbar, [0, 1, 2], ["Not detected", r"non-$\it{PHAS}$ cluster", r"$\it{PHAS}$"])

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = heat.get_figure()
    fig.savefig(_join_outdir(outdir, f"{phase}_{plot_type}_PHAS.pdf"), dpi=300)
    plt.close(fig)
    return

def plot_phasAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting phasAbundance heatmap ######")

    global outdir, phase

    # Keep full library columns (even if no PHAS in some libs)
    all_libs = sorted(list(phasis_result_df["alib"].unique()))

    # For phase==24: drop loci (rows) that have no PHAS call in any library
    df = phasis_result_df
    if str(phase) == "24":
        ids_with_phas = set(df.loc[df["label"] == "PHAS", "identifier"].unique())
        df = df[df["identifier"].isin(ids_with_phas)].copy()

        if df.empty:
            f, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")
            ax.text(0.01, 0.5, _format_phas_plot_text("No 24-PHAS loci detected."), fontsize=12)
            f.savefig(_join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS.pdf"), dpi=300)
            plt.close(f)
            return None

    data = pd.DataFrame(
        data=0.0,
        columns=all_libs,
        index=list(df["identifier"].unique())
    )

    it = tqdm(data.index, desc="Processing Rows") if tqdm else data.index
    for i in it:
        tempRows = df[df["identifier"] == i]
        for j in data.columns:
            k = 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "PHAS" in subSetData["label"].values:
                    if "log_clust_len_norm_counts" in subSetData.columns and len(subSetData["log_clust_len_norm_counts"]) > 0:
                        k = float(subSetData["log_clust_len_norm_counts"].iloc[0])
            data.loc[i, j] = k

    # Sort chromosomes numerically
    chrom_data = [identifier.split(":")[0] for identifier in data.index]
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx]
    )
    data = data.iloc[sorted_indices]

    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    min_value, max_value = _compute_finite_color_scale(data)

    f, ax = plt.subplots(figsize=(11, 11))

    # Define custom colors (exact original)
    colors = ["#C3D8EA", "#3F2F13"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, 10)
    norm = Normalize(vmin=min_value, vmax=max_value)
    ticks, labels = _build_colorbar_ticks_and_labels(min_value, max_value)

    # Colorbar axis (exact original placement)
    cax = inset_axes(
        ax,
        width="5%",
        height="9%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 1.01, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="vertical")
    _set_colorbar_ticks_and_labels(cbar, ticks, labels)

    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, norm=norm, xticklabels=True, yticklabels=False, ax=ax)

    # Chromosome bar
    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])

    chrom_cmap = _build_discrete_chromosome_cmap(unique_chromosomes)
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax2.imshow(
        chrom_colors,
        cmap=chrom_cmap,
        aspect="auto"
    )
    cax2.set_xticks([])
    cax2.set_yticks([])

    cax2.spines["top"].set_visible(False)
    cax2.spines["right"].set_visible(False)
    cax2.spines["bottom"].set_visible(False)
    cax2.spines["left"].set_visible(False)

    # IMPORTANT: for phase==24, do NOT force ylim; this avoids half-row offsets when few rows remain
    if str(phase) != "24":
        cax2.set_ylim(len(data.index), 0)

    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    cax2.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha="center")
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = heat.get_figure()
    fig.savefig(_join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS.pdf"), dpi=300)
    plt.close(fig)
    return None

def plot_totalAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting PHAS and non-PHAS Heatmaps ######")

    global outdir, phase

    # Keep full library columns
    all_libs = sorted(list(phasis_result_df["alib"].unique()))

    # For phase==24: drop loci (rows) that have no PHAS call in any library
    df = phasis_result_df
    if str(phase) == "24":
        ids_with_phas = set(df.loc[df["label"] == "PHAS", "identifier"].unique())
        df = df[df["identifier"].isin(ids_with_phas)].copy()

        if df.empty:
            f, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")
            ax.text(0.01, 0.5, _format_phas_plot_text("No 24-PHAS loci detected."), fontsize=12)
            f.savefig(_join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS_and_nonPHAS.pdf"), dpi=300)
            plt.close(f)
            return None

    data_phas = pd.DataFrame(
        data=0.0,
        columns=all_libs,
        index=list(df["identifier"].unique())
    )
    data_non_phas = data_phas.copy()

    it = tqdm(data_phas.index, desc="Processing Rows") if tqdm else data_phas.index
    for i in it:
        tempRows = df[df["identifier"] == i]
        for j in data_phas.columns:
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty and "total_abund" in subSetData.columns and len(subSetData["total_abund"]) > 0:
                if "PHAS" in subSetData["label"].values:
                    data_phas.loc[i, j] = float(subSetData["total_abund"].iloc[0])
                elif "non-PHAS" in subSetData["label"].values:
                    data_non_phas.loc[i, j] = float(subSetData["total_abund"].iloc[0])

    # Apply log10 normalization (exact original)
    data_phas = np.log10(data_phas.replace(0, np.nan).fillna(1e-10))
    data_non_phas = np.log10(data_non_phas.replace(0, np.nan).fillna(1e-10))

    chrom_data = [identifier.split(":")[0] for identifier in data_phas.index]
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx]
    )
    data_phas = data_phas.iloc[sorted_indices]
    data_non_phas = data_non_phas.iloc[sorted_indices]

    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    min_value_phas, max_value_phas = _compute_finite_color_scale(data_phas)
    min_value_non_phas, max_value_non_phas = _compute_finite_color_scale(data_non_phas)

    f, ax = plt.subplots(figsize=(11, 11))

    phas_colors = ["#D5E6D6", "#FF9999", "#FF0000"]
    non_phas_colors = ["#D5E6D6", "#9999FF", "#0000FF"]

    norm_phas = Normalize(vmin=min_value_phas, vmax=max_value_phas)
    norm_non_phas = Normalize(vmin=min_value_non_phas, vmax=max_value_non_phas)

    cmap_phas = LinearSegmentedColormap.from_list("PHAS", phas_colors, N=256)
    sns.heatmap(data_phas, square=False, cmap=cmap_phas, cbar=False, norm=norm_phas, xticklabels=True, yticklabels=False, ax=ax)

    cmap_non_phas = LinearSegmentedColormap.from_list("non-PHAS", non_phas_colors, N=256)
    sns.heatmap(data_non_phas, square=False, cmap=cmap_non_phas, cbar=False, norm=norm_non_phas, xticklabels=True, yticklabels=False, ax=ax, alpha=0.5)

    cax_phas = inset_axes(
        ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 0.6, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_phas, norm=norm_phas), cax=cax_phas, orientation="vertical")
    ticks_phas, labels_phas = _build_colorbar_ticks_and_labels(min_value_phas, max_value_phas)
    _set_colorbar_ticks_and_labels(cbar_phas, ticks_phas, labels_phas)
    cbar_phas.set_label(r"log of $\it{PHAS}$ abundance", rotation=90, labelpad=15)

    cax_non_phas = inset_axes(
        ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar_non_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_non_phas, norm=norm_non_phas), cax=cax_non_phas, orientation="vertical")
    ticks_non_phas, labels_non_phas = _build_colorbar_ticks_and_labels(min_value_non_phas, max_value_non_phas)
    _set_colorbar_ticks_and_labels(cbar_non_phas, ticks_non_phas, labels_non_phas)
    cbar_non_phas.set_label(r"log of non-$\it{PHAS}$ abundance", rotation=90, labelpad=10)

    # Chromosome bar (exact original, and this one was already correct)
    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])
    chrom_cmap = _build_discrete_chromosome_cmap(unique_chromosomes)
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax2.imshow(chrom_colors, cmap=chrom_cmap, aspect="auto")
    cax2.set_xticks([])
    cax2.set_yticks([])

    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    cax2.set_yticklabels(unique_chromosomes, fontsize=12, rotation=0, ha="center")
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = ax.get_figure()
    fig.savefig(_join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS_and_nonPHAS.pdf"), dpi=300)
    plt.close(fig)
    return None

def _plot_wrapper(job):
    """
    Worker-safe plot wrapper.

    Accepts:
      - (fn, df, mname)                              [legacy format]
      - (fn, df, mname, job_outdir, job_phase)       [spawn-safe format]

    Ensures `outdir` and `phase` globals are defined in the worker
    without importing inside the function (macOS spawn-safe).
    """
    if not isinstance(job, (tuple, list)):
        raise TypeError(f"_plot_wrapper expected tuple/list, got: {type(job)}")

    n = len(job)
    if n == 3:
        fn, df, mname = job
        job_outdir, job_phase = None, None
    elif n == 5:
        fn, df, mname, job_outdir, job_phase = job
    else:
        raise ValueError(f"Unexpected plot job tuple size: {n}")

    # Ensure globals exist in worker (spawn/fork safe)
    global outdir, phase

    # Prefer explicit values passed in the job (most reliable across spawn)
    if job_outdir is not None:
        outdir = job_outdir
    elif outdir is None:
        # Fallback to runtime snapshot loaded in worker initializer
        outdir = getattr(rt, "outdir", outdir)

    if job_phase is not None:
        phase = job_phase
    elif phase is None:
        phase = getattr(rt, "phase", phase)

    # Ensure output directory exists (safe in parallel)
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            pass

    return fn(df, mname)


def finalize_and_write_results(method_name: str, features: pd.DataFrame, *, job_outdir: str | None = None, job_phase: str | int | None = None):
    
    """
    Build result dataframe (all clusters + filtered PHAS),
    write standardized outputs, run 3 tree plots in parallel, and write GFF.
    """
    # Resolve config (prefer explicit args; fallback to runtime)
    global outdir, phase
    if job_outdir is not None:
        outdir = job_outdir
    elif outdir is None:
        outdir = getattr(rt, "outdir", outdir)

    if job_phase is not None:
        phase = job_phase
    elif phase is None:
        phase = getattr(rt, "phase", phase)

    try:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    # Decompose identifiers and clean alib tags
    achr, start, end, alib_ids = _parse_identifiers_and_alib(features, phase)

    nrows = len(features)

    # ---- Full 'all clusters' table (unchanged, includes strict columns) ----
    all_df = pd.DataFrame({
        'identifier': features['identifier'],
        'phasis_score': features['phasis_score'],
        'achr': achr,
        'start': start,
        'end': end,
        'complexity': features['complexity'],
        'strand_bias': features['strand_bias'],
        'log_clust_len_norm_counts': features['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': features['ratio_abund_len_phase'],
        'label': features['label'],
        'alib': alib_ids,
        'combined_fishers': features.get('combined_fishers', _fallback_series(nrows)),
        'total_abund': features.get('total_abund', _fallback_series(nrows)),
        'w_Howell_score': features.get('w_Howell_score', _fallback_series(nrows)),
        'w_window_start': features.get('w_window_start', _fallback_series(nrows)),
        'w_window_end': features.get('w_window_end', _fallback_series(nrows)),
        'c_Howell_score': features.get('c_Howell_score', _fallback_series(nrows)),
        'c_window_start': features.get('c_window_start', _fallback_series(nrows)),
        'c_window_end': features.get('c_window_end', _fallback_series(nrows)),
        'Peak_Howell_score': features.get('Peak_Howell_score', _fallback_series(nrows)),
        'Howell_exact_support_score': features.get('Howell_exact_support_score', _fallback_series(nrows)),
        'Howell_ambiguity_count': features.get('Howell_ambiguity_count', _fallback_series(nrows)),
        'Howell_alt_register_count': features.get('Howell_alt_register_count', _fallback_series(nrows)),
        'Howell_overlap_margin': features.get('Howell_overlap_margin', _fallback_series(nrows)),
        'Howell_extension_window_count': features.get('Howell_extension_window_count', _fallback_series(nrows)),
        'Howell_extension_span_nt': features.get('Howell_extension_span_nt', _fallback_series(nrows)),
        'Howell_origin_window_count': features.get('Howell_origin_window_count', _fallback_series(nrows)),
        'Howell_origin_frame_count': features.get('Howell_origin_frame_count', _fallback_series(nrows)),
        'Howell_origin_margin': features.get('Howell_origin_margin', _fallback_series(nrows)),
        'Howell_origin_class': features.get('Howell_origin_class', pd.Series([np.nan] * int(nrows))),
        'Howell_additional_peak_count': features.get('Howell_additional_peak_count', _fallback_series(nrows)),
        'Howell_additional_peak_best_score': features.get('Howell_additional_peak_best_score', _fallback_series(nrows)),
        'Howell_crowding_window_count': features.get('Howell_crowding_window_count', _fallback_series(nrows)),
        'Howell_crowding_best_score': features.get('Howell_crowding_best_score', _fallback_series(nrows)),
        'Howell_crowding_score_gap': features.get('Howell_crowding_score_gap', _fallback_series(nrows)),
        # strict (classic) Howell
        'w_Howell_score_strict': features.get('w_Howell_score_strict', _fallback_series(nrows)),
        'w_window_start_strict': features.get('w_window_start_strict', _fallback_series(nrows)),
        'w_window_end_strict': features.get('w_window_end_strict', _fallback_series(nrows)),
        'c_Howell_score_strict': features.get('c_Howell_score_strict', _fallback_series(nrows)),
        'c_window_start_strict': features.get('c_window_start_strict', _fallback_series(nrows)),
        'c_window_end_strict': features.get('c_window_end_strict', _fallback_series(nrows)),
        'Peak_Howell_score_strict': features.get('Peak_Howell_score_strict', _fallback_series(nrows)),
    })

    # Standardized filenames
    all_out   = _join_outdir(outdir, f"{phase}_{method_name}_all_clusters.tsv")
    calls_out = _join_outdir(outdir, f"{phase}_{method_name}_calls.tsv")
    gff_out   = _join_outdir(outdir, f"{phase}_PHAS.gff")
    qc_out    = _join_outdir(outdir, f"{phase}_{method_name}_classification_qc.tsv")
 
    # Write all clusters with labels
    all_df.to_csv(all_out, sep="\t", index=False)

    qc_df = pd.DataFrame({
        "identifier": features["identifier"],
        "alib": alib_ids,
        "cID": features.get("cID", _fallback_text_series(nrows)),
        "pre_qc_label": features.get("pre_qc_label", features.get("label", _fallback_text_series(nrows, "non-PHAS"))),
        "report_label": features.get("report_label", features.get("label", _fallback_text_series(nrows, "non-PHAS"))),
        "final_class": features.get("final_class", features.get("label", _fallback_text_series(nrows, "non-PHAS"))),
        "qc_reason": features.get("qc_reason", _fallback_text_series(nrows, "")),
        "Peak_Howell_score": features.get("Peak_Howell_score", _fallback_series(nrows)),
        "Howell_exact_support_score": features.get("Howell_exact_support_score", _fallback_series(nrows)),
        "Howell_origin_class": features.get("Howell_origin_class", _fallback_text_series(nrows)),
        "Howell_origin_window_count": features.get("Howell_origin_window_count", _fallback_series(nrows)),
        "Howell_origin_frame_count": features.get("Howell_origin_frame_count", _fallback_series(nrows)),
        "Howell_alt_register_count": features.get("Howell_alt_register_count", _fallback_series(nrows)),
        "Howell_additional_peak_count": features.get("Howell_additional_peak_count", _fallback_series(nrows)),
        "Howell_additional_peak_best_score": features.get("Howell_additional_peak_best_score", _fallback_series(nrows)),
        "Howell_crowding_window_count": features.get("Howell_crowding_window_count", _fallback_series(nrows)),
        "Howell_crowding_best_score": features.get("Howell_crowding_best_score", _fallback_series(nrows)),
        "Howell_crowding_score_gap": features.get("Howell_crowding_score_gap", _fallback_series(nrows)),
        "secondary_peak_ratio": features.get("secondary_peak_ratio", _fallback_series(nrows)),
        "override_note": features.get("override_note", _fallback_text_series(nrows)),
    })
    qc_df.to_csv(qc_out, sep="\t", index=False)

    final_class_series = features.get("final_class", features["label"]).astype(str)
    phas_mask = final_class_series == "PHAS"
    phas_df = all_df.loc[phas_mask].copy()

    # Write GFF
    write_gff(phas_df,gff_out)

    # ---- Compact calls table (same as before + Peak_Howell_score_strict) ----
    calls_cols = [
        'identifier', 'phasis_score', 'achr', 'start', 'end', 'alib',
        'Peak_Howell_score', 'Peak_Howell_score_strict',
        'Howell_exact_support_score',
        'Howell_ambiguity_count', 'Howell_alt_register_count', 'Howell_overlap_margin',
        'Howell_extension_window_count', 'Howell_extension_span_nt',
        'Howell_origin_window_count', 'Howell_origin_frame_count',
        'Howell_origin_margin', 'Howell_origin_class',
        'Howell_additional_peak_count', 'Howell_additional_peak_best_score',
    ]
    # Ensure missing columns are created as NaN so write doesn't fail
    for col in calls_cols:
        if col not in phas_df.columns:
            phas_df[col] = np.nan
    compact_calls = phas_df[calls_cols].copy()
    compact_calls.to_csv(calls_out, sep="\t", index=False)

    # --- Run the 4 plots in parallel (each on a core) ---
    plot_jobs = [
        (plot_report_heat_map, all_df, method_name, outdir, phase),
        (plot_phasAbundance_heat_map, all_df, method_name, outdir, phase),
        (plot_totalAbundance_heat_map, all_df, method_name, outdir, phase),
        (plot_howell_score_heat_maps, all_df, method_name, outdir, phase),
    ]
    plot_pool = make_pool(min(4, cpu_count()), kind="plot")
    try:
        plot_pool.map(_plot_wrapper, plot_jobs)
        plot_pool.close()
        plot_pool.join()
    except Exception:
        plot_pool.terminate()
        plot_pool.join()
        raise
    _print_final_detection_summary(
        phas_df,
        wrote_line=f"  - Wrote: {all_out}, {calls_out}, {qc_out}, and {gff_out}",
    )
