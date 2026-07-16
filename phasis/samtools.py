"""Resolve, validate, and reuse the samtools executable for one Phasis run."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess

from phasis import runtime as rt


MIN_SAMTOOLS_VERSION = (1, 10)


class SamtoolsError(RuntimeError):
    """Raised when samtools cannot be resolved or validated."""


@dataclass(frozen=True)
class SamtoolsInfo:
    path: str
    version: tuple[int, ...]
    raw_version_output: str

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.version)


def _format_minimum_version() -> str:
    return ".".join(str(part) for part in MIN_SAMTOOLS_VERSION)


def _actionable_path_message() -> str:
    return (
        "Activate the intended Conda environment or adjust PATH, then verify with "
        "'which samtools' and 'samtools --version'."
    )


def _parse_version(raw_output: str) -> tuple[int, ...] | None:
    """Extract a samtools release number from ``samtools --version`` output."""
    match = re.search(r"\bsamtools\s+(\d+(?:\.\d+)+)", raw_output, flags=re.IGNORECASE)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _is_supported(version: tuple[int, ...]) -> bool:
    padded = tuple(version) + (0,) * max(0, len(MIN_SAMTOOLS_VERSION) - len(version))
    return padded[: len(MIN_SAMTOOLS_VERSION)] >= MIN_SAMTOOLS_VERSION


def validate_samtools(executable: str | None = None) -> SamtoolsInfo:
    """Resolve and validate samtools >= 1.10, returning the absolute executable path."""
    requested = executable or "samtools"
    resolved = shutil.which(requested)
    if not resolved:
        raise SamtoolsError(
            f"samtools executable was not found (requested: {requested!r}). "
            f"{_actionable_path_message()}"
        )

    resolved = os.path.abspath(resolved)
    try:
        result = subprocess.run(
            [resolved, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SamtoolsError(
            f"Unable to execute samtools at {resolved!r}: {exc}. {_actionable_path_message()}"
        ) from exc

    raw_output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    version = _parse_version(raw_output)
    if result.returncode != 0 or version is None:
        detail = raw_output or "<no version output>"
        raise SamtoolsError(
            f"Could not determine a supported samtools version from {resolved!r}.\n"
            f"Raw 'samtools --version' output:\n{detail}\n"
            f"Phasis requires samtools >= {_format_minimum_version()}. {_actionable_path_message()}"
        )
    if not _is_supported(version):
        raise SamtoolsError(
            f"Unsupported samtools version {'.'.join(map(str, version))} at {resolved!r}; "
            f"Phasis requires samtools >= {_format_minimum_version()}. {_actionable_path_message()}"
        )
    return SamtoolsInfo(path=resolved, version=version, raw_version_output=raw_output)


def validate_runtime_samtools(*, announce: bool = True) -> SamtoolsInfo:
    """Validate the runtime path once and publish it for mapping and parser workers."""
    info = validate_samtools(getattr(rt, "samtools_path", None))
    rt.samtools_path = info.path
    rt.samtools_version = info.version_text
    if announce:
        print(f"--Samtools{'':<22}: found")
        print(f"  executable: {info.path}")
        print(f"  version: {info.version_text} (minimum {_format_minimum_version()})")
    return info


def runtime_samtools_path() -> str:
    """Return the already validated samtools path; never resolve PATH downstream."""
    path = getattr(rt, "samtools_path", None)
    if not path:
        raise SamtoolsError(
            "samtools was not validated during startup, so Phasis will not resolve it again downstream. "
            f"{_actionable_path_message()}"
        )
    return str(path)
