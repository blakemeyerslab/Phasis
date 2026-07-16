# phasis/deps_check.py
from __future__ import annotations

import importlib.util
import shutil
from typing import Iterable, List

from phasis.samtools import SamtoolsError, validate_runtime_samtools


PY_DEPS: List[str] = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",      # scikit-learn
    "matplotlib",
    "seaborn",
    "tqdm",
]

TOOLS: List[str] = [
    "hisat2",
]


def _missing_python(pkgs: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for name in pkgs:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def _missing_tools(tools: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for t in tools:
        if shutil.which(t) is None:
            missing.append(t)
    return missing


def require_dependencies() -> None:
    missing_py = _missing_python(PY_DEPS)
    missing_tools = _missing_tools(TOOLS)

    if not missing_py and not missing_tools:
        try:
            validate_runtime_samtools(announce=True)
            return
        except SamtoolsError as exc:
            print(f"\n[Phasis] {exc}\n")
            raise SystemExit(2) from exc

    print("\n[Phasis] Missing dependencies detected:\n")
    if missing_py:
        print("  Python packages (pip/conda):")
        for m in missing_py:
            print(f"    - {m}")
        print("")
    if missing_tools:
        print("  Executables (PATH):")
        for m in missing_tools:
            print(f"    - {m}")
        print("")

    print("Fix the missing items and re-run. Exiting.\n")
    raise SystemExit(2)
