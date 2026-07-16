"""
Classification stage.

IMPORTANT:
- Pure stage: does NOT write files and does NOT plot.
- Phasis 2.8.1 uses GMM classification followed by the
  Register-Resolved Locus Interpretation Layer.
- No nested functions; no imports inside functions.
"""

import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

# Feature set used by the GMM classifier.
GMM_FEATURE_COLS = [
    "complexity",
    "strand_bias",
    "log_clust_len_norm_counts",
    "ratio_abund_len_phase",
    "phasis_score",
]
FINAL_CLASS_VALUES = {"PHAS", "PHAS-like", "non-PHAS"}
NON_PHAS = "non-PHAS"
PHAS = "PHAS"
PHAS_LIKE = "PHAS-like"
EVIDENCE_REASON_PASS = "pass"
EVIDENCE_REASON_CLASSIFIER_NON_PHAS = "classifier_non_phas"
EVIDENCE_REASON_INSUFFICIENT_EXACT_SUPPORT = "insufficient_exact_support"
EVIDENCE_REASON_WEAK_EXACT_SUPPORT = "weak_exact_support"
EVIDENCE_REASON_LOW_SCORE_CROWDED = "low_score_crowded_window_context"
EVIDENCE_REASON_WEAK_SCAFFOLD_CONTEXT = "weak_scaffold_alternative_context"
EVIDENCE_REASON_LEGACY = "legacy_classification"
EVIDENCE_REASON_MANUAL_OVERRIDE = "manual_override"
PHAS_LIKE_MIN_EXACT_SUPPORT_SCORE = 5.0
PHAS_LIKE_MIN_RELAXED_SCORE = 12.5
PHAS_LIKE_MAX_RELAXED_SCORE = 20.0
PHAS_LIKE_MIN_CROWDING_WINDOWS = 5
PHAS_LIKE_WEAK_CONTEXT_MIN_RELAXED_SCORE = 20.0
PHAS_LIKE_WEAK_CONTEXT_MAX_RELAXED_SCORE = 30.0
PHAS_LIKE_MAX_EXACT_RELAXED_RATIO = 0.20
PHAS_LIKE_MAX_STRICT_RELAXED_RATIO = 0.40
PHAS_LIKE_MIN_HIGH_CROWDING_WINDOWS = 7
PHAS_LIKE_MIN_OVERLAPPING_ALT_COUNT = 2
PHAS_LIKE_MAX_OVERLAPPING_ALT_GAP = 4.0
PHAS_LIKE_MIN_COMPLEXITY = 0.20


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value, *, default: int = 0) -> int:
    try:
        fvalue = float(value)
    except Exception:
        return int(default)
    if np.isnan(fvalue):
        return int(default)
    return int(fvalue)


def _safe_ratio(numerator, denominator) -> float:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if np.isnan(num) or np.isnan(den) or den <= 0.0:
        return float("nan")
    return float(num / den)


def _normalize_override_class(value) -> str:
    text = str(value or "").strip()
    lookup = {
        "phas": PHAS,
        "phas-like": PHAS_LIKE,
        "phas_like": PHAS_LIKE,
        "phas like": PHAS_LIKE,
        "non-phas": NON_PHAS,
        "non_phas": NON_PHAS,
        "non phas": NON_PHAS,
    }
    normalized = lookup.get(text.lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported final_class override value {value!r}. "
            f"Allowed values: {sorted(FINAL_CLASS_VALUES)}"
        )
    return normalized


def _normalize_evidence_alib(alib_value, *, phase) -> str:
    if alib_value is None:
        return ""
    if phase is None:
        return str(alib_value)
    return re.sub(
        rf"\.{re.escape(str(phase))}-PHAS\.candidate$",
        "",
        str(alib_value),
    )


def _build_detection_key(identifier_value, alib_value, *, phase) -> tuple[str, str]:
    return (
        str(identifier_value or "").strip(),
        _normalize_evidence_alib(alib_value, phase=phase).strip(),
    )


def _read_classification_overrides(
    overrides_path: Optional[str],
    *,
    phase,
) -> dict[tuple[str, str], dict]:
    if not overrides_path:
        return {}

    path = os.path.abspath(os.path.expanduser(str(overrides_path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classification override file not found: {path}")
    if os.path.getsize(path) == 0:
        return {}

    overrides_df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"identifier", "alib", "final_class"}
    missing = sorted(required - set(overrides_df.columns))
    if missing:
        raise ValueError(
            f"Classification override file {path} is missing required columns: {missing}"
        )

    seen = set()
    lookup = {}
    for row in overrides_df.itertuples(index=False):
        key = _build_detection_key(
            getattr(row, "identifier", ""),
            getattr(row, "alib", ""),
            phase=phase,
        )
        if key in seen:
            raise ValueError(
                f"Duplicate classification override key detected for identifier={key[0]!r}, alib={key[1]!r}"
            )
        seen.add(key)
        evidence_reason = (
            str(getattr(row, "evidence_reason", "") or "").strip()
            or EVIDENCE_REASON_MANUAL_OVERRIDE
        )
        note = str(getattr(row, "note", "") or "").strip()
        lookup[key] = {
            "final_class": _normalize_override_class(getattr(row, "final_class", "")),
            "evidence_reason": evidence_reason,
            "note": note,
        }
    return lookup


def _automatic_evidence_classification(row) -> tuple[str, str]:
    initial_classifier_label = str(row.get("initial_classifier_label", NON_PHAS) or NON_PHAS)
    if initial_classifier_label != PHAS:
        return NON_PHAS, EVIDENCE_REASON_CLASSIFIER_NON_PHAS

    exact_support = _safe_float(row.get("Howell_exact_support_score"))
    origin_class = str(row.get("Howell_origin_class", "") or "").strip()
    if np.isnan(exact_support) or exact_support <= 0.0 or origin_class == EVIDENCE_REASON_INSUFFICIENT_EXACT_SUPPORT:
        return NON_PHAS, EVIDENCE_REASON_INSUFFICIENT_EXACT_SUPPORT
    if exact_support < PHAS_LIKE_MIN_EXACT_SUPPORT_SCORE:
        return PHAS_LIKE, EVIDENCE_REASON_WEAK_EXACT_SUPPORT

    relaxed_peak_score = _safe_float(row.get("Peak_Howell_score"))
    crowding_window_count = _safe_int(row.get("Howell_crowding_window_count"), default=0)
    if (
        not np.isnan(exact_support)
        and exact_support > 0.0
        and not np.isnan(relaxed_peak_score)
        and PHAS_LIKE_MIN_RELAXED_SCORE < relaxed_peak_score < PHAS_LIKE_MAX_RELAXED_SCORE
        and crowding_window_count >= PHAS_LIKE_MIN_CROWDING_WINDOWS
    ):
        return PHAS_LIKE, EVIDENCE_REASON_LOW_SCORE_CROWDED

    exact_relaxed_ratio = _safe_float(row.get("Howell_exact_relaxed_ratio"))
    if np.isnan(exact_relaxed_ratio):
        exact_relaxed_ratio = _safe_ratio(exact_support, relaxed_peak_score)

    strict_relaxed_ratio = _safe_float(row.get("Howell_strict_relaxed_ratio"))
    if np.isnan(strict_relaxed_ratio):
        strict_relaxed_ratio = _safe_ratio(
            row.get("Peak_Howell_score_strict"),
            relaxed_peak_score,
        )

    overlapping_alt_count = _safe_int(row.get("Howell_overlapping_alt_count"), default=0)
    overlapping_alt_best_score = _safe_float(row.get("Howell_overlapping_alt_best_score"))
    complexity = _safe_float(row.get("complexity"))

    support_weakness = (
        (not np.isnan(exact_relaxed_ratio) and exact_relaxed_ratio < PHAS_LIKE_MAX_EXACT_RELAXED_RATIO)
        or (not np.isnan(strict_relaxed_ratio) and strict_relaxed_ratio < PHAS_LIKE_MAX_STRICT_RELAXED_RATIO)
    )
    overlapping_alt_close = (
        not np.isnan(overlapping_alt_best_score)
        and not np.isnan(relaxed_peak_score)
        and float(relaxed_peak_score) - float(overlapping_alt_best_score) <= PHAS_LIKE_MAX_OVERLAPPING_ALT_GAP
    )
    context_weakness = (
        origin_class in {"ambiguous_origin", "mixed_extension_and_ambiguity"}
        or crowding_window_count >= PHAS_LIKE_MIN_HIGH_CROWDING_WINDOWS
        or overlapping_alt_count >= PHAS_LIKE_MIN_OVERLAPPING_ALT_COUNT
        or overlapping_alt_close
        or (not np.isnan(complexity) and complexity >= PHAS_LIKE_MIN_COMPLEXITY)
    )
    if (
        not np.isnan(relaxed_peak_score)
        and PHAS_LIKE_WEAK_CONTEXT_MIN_RELAXED_SCORE <= relaxed_peak_score < PHAS_LIKE_WEAK_CONTEXT_MAX_RELAXED_SCORE
        and support_weakness
        and context_weakness
    ):
        return PHAS_LIKE, EVIDENCE_REASON_WEAK_SCAFFOLD_CONTEXT

    return PHAS, EVIDENCE_REASON_PASS


def apply_evidence_classification(
    features: pd.DataFrame,
    *,
    phase=None,
    legacy_classification: bool = False,
    overrides_path: Optional[str] = None,
) -> pd.DataFrame:
    out = features.copy()
    if "label" not in out.columns:
        raise ValueError("Evidence classification requires a 'label' column from the classifier stage")

    out["initial_classifier_label"] = out["label"].astype(str)
    out["Howell_exact_relaxed_ratio"] = [
        _safe_ratio(exact_support, peak_score)
        for exact_support, peak_score in zip(
            out.get("Howell_exact_support_score", pd.Series(index=out.index, dtype=float)),
            out.get("Peak_Howell_score", pd.Series(index=out.index, dtype=float)),
        )
    ]
    out["Howell_strict_relaxed_ratio"] = [
        _safe_ratio(strict_score, peak_score)
        for strict_score, peak_score in zip(
            out.get("Peak_Howell_score_strict", pd.Series(index=out.index, dtype=float)),
            out.get("Peak_Howell_score", pd.Series(index=out.index, dtype=float)),
        )
    ]
    out["secondary_peak_ratio"] = [
        _safe_ratio(best_score, peak_score)
        for best_score, peak_score in zip(
            out.get("Howell_additional_peak_best_score", pd.Series(index=out.index, dtype=float)),
            out.get("Peak_Howell_score", pd.Series(index=out.index, dtype=float)),
        )
    ]

    final_classes = []
    evidence_reasons = []
    for row in out.to_dict("records"):
        if legacy_classification:
            final_class = PHAS if str(row.get("initial_classifier_label", NON_PHAS) or NON_PHAS) == PHAS else NON_PHAS
            evidence_reason = EVIDENCE_REASON_LEGACY
        else:
            final_class, evidence_reason = _automatic_evidence_classification(row)
        final_classes.append(final_class)
        evidence_reasons.append(evidence_reason)

    out["final_class"] = final_classes
    out["evidence_reason"] = evidence_reasons
    out["override_note"] = ""

    overrides = _read_classification_overrides(overrides_path, phase=phase)
    if overrides:
        for idx, row in out.iterrows():
            key = _build_detection_key(row.get("identifier"), row.get("alib"), phase=phase)
            override = overrides.get(key)
            if not override:
                continue
            out.at[idx, "final_class"] = override["final_class"]
            out.at[idx, "evidence_reason"] = override["evidence_reason"]
            out.at[idx, "override_note"] = override["note"]

    out["report_label"] = np.where(out["final_class"] == PHAS, PHAS, NON_PHAS)
    out["label"] = out["report_label"]
    return out


def _apply_post_filters(
    df: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
) -> pd.DataFrame:
    """
    Apply score, Howell-support, and complexity post-filters after GMM labeling.
    """
    out = df.copy()

    # 1) Score + Howell filter
    out.loc[
        (out["phasis_score"] < phasisScoreCutoff)
        | (out["Peak_Howell_score"] < min_Howell_score),
        "label",
    ] = "non-PHAS"

    # 2) Complexity filter
    out.loc[(out["complexity"] > max_complexity), "label"] = "non-PHAS"

    return out


def gmm_classify(
    features: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
    n_clusters: int = 2,
) -> pd.DataFrame:
    """
    GMM clustering with post-filters.
    Chooses the GMM cluster with the highest mean phasis_score as PHAS.
    """
    cols_for_model = list(GMM_FEATURE_COLS)
    X = features[cols_for_model].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    try:
        cluster_labels = gmm.fit_predict(X_scaled)
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'split'" not in str(exc):
            raise
        warnings.warn(
            "GMM k-means initialization failed while inspecting BLAS thread pools; "
            "retrying with deterministic random_from_data initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=0,
            init_params="random_from_data",
        )
        cluster_labels = gmm.fit_predict(X_scaled)

    tmp = pd.DataFrame(
        {"cluster": cluster_labels, "phasis_score": features["phasis_score"].values}
    )
    phas_cluster = tmp.groupby("cluster")["phasis_score"].mean().idxmax()

    out = features.copy()
    out["label"] = np.where(cluster_labels == phas_cluster, "PHAS", "non-PHAS")

    out = _apply_post_filters(out, phasisScoreCutoff, min_Howell_score, max_complexity)
    return out


def resolve_pipeline_classification_args(
    *,
    cfg=None,
    phasisScoreCutoff=None,
    min_Howell_score=None,
    max_complexity=None,
    job_outdir=None,
    job_phase=None,
    default_phasisScoreCutoff=None,
    default_min_Howell_score=None,
    default_max_complexity=None,
    default_job_outdir=None,
    default_job_phase=None,
):
    """
    Resolve legacy/runtime-facing classification args outside legacy.py.

    This keeps stage-owned parameter normalization together with the
    classification stage while remaining pure (no file writing).
    """
    if cfg is not None:
        phasisScoreCutoff = cfg.phasisScoreCutoff
        min_Howell_score = cfg.min_Howell_score
        max_complexity = cfg.max_complexity
        job_outdir = cfg.outdir
        job_phase = cfg.phase

    if phasisScoreCutoff is None:
        phasisScoreCutoff = default_phasisScoreCutoff
    if min_Howell_score is None:
        min_Howell_score = default_min_Howell_score
    if max_complexity is None:
        max_complexity = default_max_complexity
    if job_outdir is None:
        job_outdir = default_job_outdir
    if job_phase is None:
        job_phase = default_job_phase

    return (
        float(phasisScoreCutoff),
        float(min_Howell_score),
        float(max_complexity),
        job_outdir,
        job_phase,
    )


def gmm_classify_for_pipeline(
    features: pd.DataFrame,
    n_clusters: int = 2,
    *,
    cfg=None,
    phasisScoreCutoff=None,
    min_Howell_score=None,
    max_complexity=None,
    job_outdir=None,
    job_phase=None,
    default_phasisScoreCutoff=None,
    default_min_Howell_score=None,
    default_max_complexity=None,
    default_job_outdir=None,
    default_job_phase=None,
):
    """
    Legacy/pipeline-facing GMM helper.

    Returns:
        (labeled_df, job_outdir, job_phase)
    """
    (
        phasisScoreCutoff,
        min_Howell_score,
        max_complexity,
        job_outdir,
        job_phase,
    ) = resolve_pipeline_classification_args(
        cfg=cfg,
        phasisScoreCutoff=phasisScoreCutoff,
        min_Howell_score=min_Howell_score,
        max_complexity=max_complexity,
        job_outdir=job_outdir,
        job_phase=job_phase,
        default_phasisScoreCutoff=default_phasisScoreCutoff,
        default_min_Howell_score=default_min_Howell_score,
        default_max_complexity=default_max_complexity,
        default_job_outdir=default_job_outdir,
        default_job_phase=default_job_phase,
    )

    labeled = gmm_classify(
        features,
        phasisScoreCutoff=phasisScoreCutoff,
        min_Howell_score=min_Howell_score,
        max_complexity=max_complexity,
        n_clusters=int(n_clusters),
    )
    return labeled, job_outdir, job_phase
