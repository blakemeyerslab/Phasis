"""
Classification stage extracted from legacy.py (migration-safe).

IMPORTANT:
- Pure stage: does NOT write files and does NOT plot.
- Keeps legacy semantics: same scaling, same model path, same post-filters.
- No nested functions; no imports inside functions.
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import joblib
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but KNeighborsClassifier was fitted with feature names",
    category=UserWarning,
)

# Keep the exact feature set that legacy.py uses for KNN/GMM
KNN_FEATURE_COLS = [
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
QC_REASON_PASS = "pass"
QC_REASON_CLASSIFIER_NON_PHAS = "classifier_non_phas"
QC_REASON_INSUFFICIENT_EXACT_SUPPORT = "insufficient_exact_support"
QC_REASON_LOW_SCORE_CROWDED = "low_score_crowded_window_context"
QC_REASON_LEGACY = "legacy_classification"
QC_REASON_MANUAL_OVERRIDE = "manual_override"
PHAS_LIKE_MIN_RELAXED_SCORE = 12.5
PHAS_LIKE_MAX_RELAXED_SCORE = 20.0
PHAS_LIKE_MIN_CROWDING_WINDOWS = 5


def _default_knn_model_path() -> str:
    """
    legacy.py uses:
        script_dir = os.path.dirname(os.path.realpath(__file__))  # legacy.py lives in phasis/
        model_path = os.path.join(script_dir, "data", "knn_model.pkl")

    This file lives in phasis/stages/, so we go one directory up to phasis/.
    """
    phasis_pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(phasis_pkg_dir, "data", "knn_model.pkl")


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


def _normalize_qc_alib(alib_value, *, phase) -> str:
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
        _normalize_qc_alib(alib_value, phase=phase).strip(),
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
        qc_reason = str(getattr(row, "qc_reason", "") or "").strip() or QC_REASON_MANUAL_OVERRIDE
        note = str(getattr(row, "note", "") or "").strip()
        lookup[key] = {
            "final_class": _normalize_override_class(getattr(row, "final_class", "")),
            "qc_reason": qc_reason,
            "note": note,
        }
    return lookup


def _automatic_qc_classification(row) -> tuple[str, str]:
    pre_qc_label = str(row.get("pre_qc_label", NON_PHAS) or NON_PHAS)
    if pre_qc_label != PHAS:
        return NON_PHAS, QC_REASON_CLASSIFIER_NON_PHAS

    exact_support = _safe_float(row.get("Howell_exact_support_score"))
    origin_class = str(row.get("Howell_origin_class", "") or "").strip()
    if (not np.isnan(exact_support) and exact_support <= 0.0) or origin_class == QC_REASON_INSUFFICIENT_EXACT_SUPPORT:
        return NON_PHAS, QC_REASON_INSUFFICIENT_EXACT_SUPPORT

    relaxed_peak_score = _safe_float(row.get("Peak_Howell_score"))
    crowding_window_count = _safe_int(row.get("Howell_crowding_window_count"), default=0)
    if (
        not np.isnan(exact_support)
        and exact_support > 0.0
        and not np.isnan(relaxed_peak_score)
        and PHAS_LIKE_MIN_RELAXED_SCORE < relaxed_peak_score < PHAS_LIKE_MAX_RELAXED_SCORE
        and crowding_window_count >= PHAS_LIKE_MIN_CROWDING_WINDOWS
    ):
        return PHAS_LIKE, QC_REASON_LOW_SCORE_CROWDED

    return PHAS, QC_REASON_PASS


def apply_qc_reclassification(
    features: pd.DataFrame,
    *,
    phase=None,
    legacy_classification: bool = False,
    overrides_path: Optional[str] = None,
) -> pd.DataFrame:
    out = features.copy()
    if "label" not in out.columns:
        raise ValueError("QC reclassification requires a 'label' column from the classifier stage")

    out["pre_qc_label"] = out["label"].astype(str)
    out["secondary_peak_ratio"] = [
        _safe_ratio(best_score, peak_score)
        for best_score, peak_score in zip(
            out.get("Howell_additional_peak_best_score", pd.Series(index=out.index, dtype=float)),
            out.get("Peak_Howell_score", pd.Series(index=out.index, dtype=float)),
        )
    ]

    final_classes = []
    qc_reasons = []
    for row in out.to_dict("records"):
        if legacy_classification:
            final_class = PHAS if str(row.get("pre_qc_label", NON_PHAS) or NON_PHAS) == PHAS else NON_PHAS
            qc_reason = QC_REASON_LEGACY
        else:
            final_class, qc_reason = _automatic_qc_classification(row)
        final_classes.append(final_class)
        qc_reasons.append(qc_reason)

    out["final_class"] = final_classes
    out["qc_reason"] = qc_reasons
    out["override_note"] = ""

    overrides = _read_classification_overrides(overrides_path, phase=phase)
    if overrides:
        for idx, row in out.iterrows():
            key = _build_detection_key(row.get("identifier"), row.get("alib"), phase=phase)
            override = overrides.get(key)
            if not override:
                continue
            out.at[idx, "final_class"] = override["final_class"]
            out.at[idx, "qc_reason"] = override["qc_reason"]
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
    Apply the exact same post-filters as legacy.KNN_phas_clustering/GMM_phas_clustering.
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


def knn_classify(
    features: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    KNN classifier (pure): returns `features` with `label` column set.
    """
    # Scale the same KNN feature set as legacy
    X = features[KNN_FEATURE_COLS].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model and predict
    if model_path is None:
        model_path = _default_knn_model_path()
    knn_clf = joblib.load(model_path)
    y_pred = knn_clf.predict(X_scaled)

    out = features.copy()
    out["label"] = np.where(y_pred == 1, "PHAS", "non-PHAS")

    # Post-filters (identical semantics)
    out = _apply_post_filters(out, phasisScoreCutoff, min_Howell_score, max_complexity)
    return out


def gmm_classify(
    features: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
    n_clusters: int = 2,
) -> pd.DataFrame:
    """
    GMM clustering aligned with KNN feature scaling + post-filters.
    Chooses the GMM cluster with the highest mean phasis_score as PHAS.
    """
    cols_for_model = list(KNN_FEATURE_COLS)
    X = features[cols_for_model].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
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


def knn_classify_for_pipeline(
    features: pd.DataFrame,
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
    Legacy/pipeline-facing KNN helper.

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

    labeled = knn_classify(
        features,
        phasisScoreCutoff=phasisScoreCutoff,
        min_Howell_score=min_Howell_score,
        max_complexity=max_complexity,
    )
    return labeled, job_outdir, job_phase


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
