from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import phasis.runtime as rt


@dataclass(frozen=True)
class Phase2Config:
    # identity/output
    phase: str
    outdir: Optional[str]
    concat_libs: bool

    # orchestration
    steps: str
    class_cluster_file: Any
    libs: Any = None

    # window selection / clustering params (Phase II)
    # NOTE: These live in runtime as rt.window_len / rt.sliding / rt.minClusterLength.
    # We give them defaults so older call sites that instantiate Phase2Config manually
    # won't break during migration.
    window_len: int = 0
    sliding: int = 0
    minClusterLength: int = 0

    # classification
    classifier: str = ""
    classifier_aliases: Any = None
    phasisScoreCutoff: float = 0.0
    min_Howell_score: float = 0.0
    max_complexity: float = 1.0
    legacy_classification: bool = False
    classification_overrides: Optional[str] = None
    locus_plot_mode: str = "clean"

    # cache
    memFile: Optional[str] = None
    compress_intermediates: bool = True

    @classmethod
    def from_runtime(cls) -> "Phase2Config":
        return cls(
            phase=str(getattr(rt, "phase", "")),
            outdir=getattr(rt, "outdir", None),
            concat_libs=bool(getattr(rt, "concat_libs", False)),
            libs=getattr(rt, "libs", None),
            steps=str(getattr(rt, "steps", "")),
            class_cluster_file=getattr(rt, "class_cluster_file", None),
            window_len=int(getattr(rt, "window_len", 0) or 0),
            sliding=int(getattr(rt, "sliding", 0) or 0),
            minClusterLength=int(getattr(rt, "minClusterLength", 0) or 0),
            classifier=str(getattr(rt, "classifier", "")),
            classifier_aliases=getattr(rt, "classifier_aliases", None),
            phasisScoreCutoff=float(getattr(rt, "phasisScoreCutoff", 0.0) or 0.0),
            min_Howell_score=float(getattr(rt, "min_Howell_score", 0.0) or 0.0),
            max_complexity=float(getattr(rt, "max_complexity", 1.0) or 1.0),
            legacy_classification=bool(getattr(rt, "legacy_classification", False)),
            classification_overrides=getattr(rt, "classification_overrides", None),
            locus_plot_mode=str(getattr(rt, "locus_plot_mode", "clean") or "clean"),
            memFile=getattr(rt, "memFile", None),
            compress_intermediates=bool(getattr(rt, "compress_intermediates", True)),
        )
