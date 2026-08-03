"""Compare and combine high-confidence loci from two PHASIS result bundles.

The comparison deliberately keeps one of the source loci as the representative
instead of manufacturing new coordinates or identifiers.  All loci from the
preferred run are retained.  A locus from the other run supports every
preferred locus that it directly overlaps and is added as a new representative
only when it overlaps none of them.  This is important when a broad pooled call
overlaps more than one finer, non-pooled call.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_CALLS_RE = re.compile(r"^(\d+)_calls\.tsv$")
_CANONICAL_SUFFIXES = {
    "calls": "calls.tsv",
    "all_clusters": "all_clusters.tsv",
    "classification_evidence": "classification_evidence.tsv",
    "phasiRNAs": "phasiRNAs.tsv",
}


class ResultComparisonError(ValueError):
    """Raised when result bundles cannot be compared safely."""


@dataclass(frozen=True)
class Locus:
    """One unique high-confidence locus from a source result bundle."""

    run: str
    identifier: str
    chromosome: str
    start: int
    end: int


@dataclass(frozen=True)
class Overlap:
    """A direct same-chromosome overlap between one locus from each run."""

    run_a: Locus
    run_b: Locus
    overlap_nt: int


@dataclass(frozen=True)
class CombinedLocus:
    """A selected output locus and all source loci that support it."""

    representative: Locus
    category: str
    members: tuple[Locus, ...]


@dataclass(frozen=True)
class ComparisonResult:
    """Files and summary returned by :func:`compare_result_directories`."""

    phase: int
    outdir: Path
    summary: Mapping[str, Any]
    paths: Mapping[str, Path]


@dataclass
class _Bundle:
    run: str
    label: str
    directory: Path
    phase: int
    paths: dict[str, Path]
    schemas: dict[str, tuple[str, ...]]
    call_rows: list[dict[str, str]]
    loci: dict[str, Locus]
    mode: str


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", str(value)):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part.casefold()))
    return tuple(parts)


def _locus_key(locus: Locus) -> tuple[Any, ...]:
    return (
        _natural_key(locus.chromosome),
        locus.start,
        locus.end,
        locus.identifier,
        locus.run,
    )


def _phase_in(directory: Path) -> int:
    phases = {
        int(match.group(1))
        for path in directory.iterdir()
        if path.is_file() and (match := _CALLS_RE.match(path.name))
    }
    if not phases:
        raise ResultComparisonError(
            f"No top-level <phase>_calls.tsv was found in {directory}"
        )
    if len(phases) != 1:
        found = ", ".join(str(value) for value in sorted(phases))
        raise ResultComparisonError(
            f"Multiple call-table phases were found in {directory}: {found}"
        )
    return phases.pop()


def _read_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
    except StopIteration as exc:
        raise ResultComparisonError(f"Required table is empty: {path}") from exc
    if not header or len(set(header)) != len(header):
        raise ResultComparisonError(f"Invalid or duplicate columns in {path}")
    return tuple(header)


def _read_table(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    header = _read_header(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return header, rows


def _clean_label(value: str, option: str) -> str:
    text = str(value).strip()
    if not text:
        raise ResultComparisonError(f"{option} cannot be empty")
    if any(char in text for char in "\t\r\n"):
        raise ResultComparisonError(f"{option} cannot contain tabs or newlines")
    return text


def _mode_from_rows(
    call_rows: Sequence[Mapping[str, str]], all_clusters_path: Path
) -> str:
    libraries = [str(row.get("alib", "")).strip() for row in call_rows]
    if not libraries:
        with all_clusters_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                libraries.append(str(row.get("alib", "")).strip())
                if libraries[-1] != "ALL_LIBS":
                    break
    # An empty, otherwise valid bundle has no evidence that it is pooled.
    return "pooled" if libraries and all(lib == "ALL_LIBS" for lib in libraries) else "non-pooled"


def _load_bundle(directory: Path, run: str, label: str) -> _Bundle:
    if not directory.is_dir():
        raise ResultComparisonError(f"Result directory does not exist: {directory}")
    phase = _phase_in(directory)
    paths = {
        name: directory / f"{phase}_{suffix}"
        for name, suffix in _CANONICAL_SUFFIXES.items()
    }
    paths["gff"] = directory / f"{phase}_PHAS.gff"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ResultComparisonError(
            "Incomplete PHASIS result directory; missing: " + ", ".join(missing)
        )

    schemas = {name: _read_header(path) for name, path in paths.items() if name != "gff"}
    required_calls = {"identifier", "achr", "start", "end", "alib"}
    missing_calls = required_calls.difference(schemas["calls"])
    if missing_calls:
        raise ResultComparisonError(
            f"{paths['calls']} is missing columns: {', '.join(sorted(missing_calls))}"
        )
    for name in ("all_clusters", "classification_evidence", "phasiRNAs"):
        if "identifier" not in schemas[name]:
            raise ResultComparisonError(f"{paths[name]} is missing column: identifier")

    _, call_rows = _read_table(paths["calls"])
    loci: dict[str, Locus] = {}
    for line_number, row in enumerate(call_rows, start=2):
        identifier = str(row["identifier"]).strip()
        chromosome = str(row["achr"]).strip()
        if not identifier or not chromosome:
            raise ResultComparisonError(
                f"Blank identifier or chromosome in {paths['calls']} line {line_number}"
            )
        try:
            start = int(str(row["start"]).strip())
            end = int(str(row["end"]).strip())
        except ValueError as exc:
            raise ResultComparisonError(
                f"Non-integer coordinates in {paths['calls']} line {line_number}"
            ) from exc
        if start > end:
            raise ResultComparisonError(
                f"Start exceeds end in {paths['calls']} line {line_number}"
            )
        locus = Locus(run, identifier, chromosome, start, end)
        previous = loci.get(identifier)
        if previous is not None and (
            previous.chromosome,
            previous.start,
            previous.end,
        ) != (chromosome, start, end):
            raise ResultComparisonError(
                f"Identifier {identifier!r} has inconsistent coordinates in {paths['calls']}"
            )
        loci[identifier] = locus

    return _Bundle(
        run=run,
        label=label,
        directory=directory,
        phase=phase,
        paths=paths,
        schemas=schemas,
        call_rows=call_rows,
        loci=loci,
        mode=_mode_from_rows(call_rows, paths["all_clusters"]),
    )


def _validate_compatible(a: _Bundle, b: _Bundle) -> None:
    if a.phase != b.phase:
        raise ResultComparisonError(
            f"PHASIS phases differ: run_a is {a.phase}, run_b is {b.phase}"
        )
    for name in _CANONICAL_SUFFIXES:
        if a.schemas[name] != b.schemas[name]:
            raise ResultComparisonError(
                f"Incompatible {name} schemas between {a.paths[name]} and {b.paths[name]}"
            )


def _find_overlaps(
    loci_a: Iterable[Locus], loci_b: Iterable[Locus], buffer: int
) -> list[Overlap]:
    by_chr_a: dict[str, list[Locus]] = defaultdict(list)
    by_chr_b: dict[str, list[Locus]] = defaultdict(list)
    for locus in loci_a:
        by_chr_a[locus.chromosome].append(locus)
    for locus in loci_b:
        by_chr_b[locus.chromosome].append(locus)

    overlaps: list[Overlap] = []
    for chromosome in sorted(set(by_chr_a).intersection(by_chr_b), key=_natural_key):
        a_loci = sorted(by_chr_a[chromosome], key=_locus_key)
        b_loci = sorted(by_chr_b[chromosome], key=_locus_key)
        active: list[Locus] = []
        b_index = 0
        for locus_a in a_loci:
            while b_index < len(b_loci) and b_loci[b_index].start <= locus_a.end + buffer:
                active.append(b_loci[b_index])
                b_index += 1
            active = [locus for locus in active if locus.end + buffer >= locus_a.start]
            for locus_b in active:
                if locus_b.start <= locus_a.end + buffer and locus_a.start <= locus_b.end + buffer:
                    overlap_nt = max(
                        0,
                        min(locus_a.end, locus_b.end)
                        - max(locus_a.start, locus_b.start)
                        + 1,
                    )
                    overlaps.append(Overlap(locus_a, locus_b, overlap_nt))
    return sorted(overlaps, key=lambda item: (_locus_key(item.run_a), _locus_key(item.run_b)))


def _preferred_run(a: _Bundle, b: _Bundle, prefer: str) -> str:
    normalized = str(prefer).strip().lower().replace("-", "_")
    aliases = {"a": "run_a", "run_a": "run_a", "b": "run_b", "run_b": "run_b"}
    if normalized in aliases:
        return aliases[normalized]
    if normalized != "auto":
        raise ResultComparisonError("prefer must be one of: auto, a, b, run_a, run_b")
    if a.mode == "pooled" and b.mode == "non-pooled":
        return "run_b"
    if b.mode == "pooled" and a.mode == "non-pooled":
        return "run_a"
    return "run_a"


def _combine(
    a: _Bundle, b: _Bundle, overlaps: Sequence[Overlap], preferred_run: str
) -> list[CombinedLocus]:
    by_a: dict[str, list[Locus]] = defaultdict(list)
    by_b: dict[str, list[Locus]] = defaultdict(list)
    for overlap in overlaps:
        by_a[overlap.run_a.identifier].append(overlap.run_b)
        by_b[overlap.run_b.identifier].append(overlap.run_a)

    combined: list[CombinedLocus] = []
    if preferred_run == "run_a":
        matched_other = set(by_b)
        for locus in sorted(a.loci.values(), key=_locus_key):
            support = tuple(sorted(by_a.get(locus.identifier, ()), key=_locus_key))
            category = "shared" if support else "run_a_only"
            combined.append(CombinedLocus(locus, category, (locus, *support)))
        for locus in sorted(b.loci.values(), key=_locus_key):
            if locus.identifier not in matched_other:
                combined.append(CombinedLocus(locus, "run_b_only", (locus,)))
    else:
        matched_other = set(by_a)
        for locus in sorted(b.loci.values(), key=_locus_key):
            support = tuple(sorted(by_b.get(locus.identifier, ()), key=_locus_key))
            category = "shared" if support else "run_b_only"
            combined.append(CombinedLocus(locus, category, (locus, *support)))
        for locus in sorted(a.loci.values(), key=_locus_key):
            if locus.identifier not in matched_other:
                combined.append(CombinedLocus(locus, "run_a_only", (locus,)))
    return sorted(combined, key=lambda item: _locus_key(item.representative))


def _write_tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _combined_rows(
    combined: Sequence[CombinedLocus], labels: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in combined:
        locus = item.representative
        members_a = [member.identifier for member in item.members if member.run == "run_a"]
        members_b = [member.identifier for member in item.members if member.run == "run_b"]
        rows.append(
            {
                "identifier": locus.identifier,
                "achr": locus.chromosome,
                "start": locus.start,
                "end": locus.end,
                "category": item.category,
                "representative_run": locus.run,
                "representative_label": labels[locus.run],
                "source_support_count": int(bool(members_a)) + int(bool(members_b)),
                "supporting_locus_count": len(item.members),
                "run_a_identifiers": ";".join(members_a),
                "run_b_identifiers": ";".join(members_b),
            }
        )
    return rows


_COMBINED_COLUMNS = (
    "identifier",
    "achr",
    "start",
    "end",
    "category",
    "representative_run",
    "representative_label",
    "source_support_count",
    "supporting_locus_count",
    "run_a_identifiers",
    "run_b_identifiers",
)


def _source_map_rows(
    combined: Sequence[CombinedLocus], labels: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in combined:
        representative = item.representative
        for member in item.members:
            overlap_nt = max(
                0,
                min(representative.end, member.end)
                - max(representative.start, member.start)
                + 1,
            )
            is_representative = member == representative
            rows.append(
                {
                    "combined_identifier": representative.identifier,
                    "combined_achr": representative.chromosome,
                    "combined_start": representative.start,
                    "combined_end": representative.end,
                    "category": item.category,
                    "source_run": member.run,
                    "source_label": labels[member.run],
                    "source_identifier": member.identifier,
                    "source_achr": member.chromosome,
                    "source_start": member.start,
                    "source_end": member.end,
                    "is_representative": "true" if is_representative else "false",
                    "relationship": "representative" if is_representative else "overlap_support",
                    "overlap_nt": overlap_nt,
                }
            )
    return rows


_SOURCE_MAP_COLUMNS = (
    "combined_identifier",
    "combined_achr",
    "combined_start",
    "combined_end",
    "category",
    "source_run",
    "source_label",
    "source_identifier",
    "source_achr",
    "source_start",
    "source_end",
    "is_representative",
    "relationship",
    "overlap_nt",
)


def _overlap_rows(overlaps: Sequence[Overlap], buffer: int) -> list[dict[str, Any]]:
    matches_per_a = Counter(item.run_a.identifier for item in overlaps)
    matches_per_b = Counter(item.run_b.identifier for item in overlaps)
    rows: list[dict[str, Any]] = []
    for item in overlaps:
        a, b = item.run_a, item.run_b
        match_count_a = matches_per_a[a.identifier]
        match_count_b = matches_per_b[b.identifier]
        if match_count_a == 1 and match_count_b == 1:
            relationship_shape = "one_to_one"
        elif match_count_a > 1 and match_count_b == 1:
            relationship_shape = "run_a_one_to_many_run_b"
        elif match_count_a == 1 and match_count_b > 1:
            relationship_shape = "run_a_many_to_one_run_b"
        else:
            relationship_shape = "many_to_many"
        length_a = a.end - a.start + 1
        length_b = b.end - b.start + 1
        rows.append(
            {
                "run_a_identifier": a.identifier,
                "run_a_achr": a.chromosome,
                "run_a_start": a.start,
                "run_a_end": a.end,
                "run_b_identifier": b.identifier,
                "run_b_achr": b.chromosome,
                "run_b_start": b.start,
                "run_b_end": b.end,
                "overlap_start": max(a.start, b.start) if item.overlap_nt else "",
                "overlap_end": min(a.end, b.end) if item.overlap_nt else "",
                "overlap_nt": item.overlap_nt,
                "run_a_overlap_fraction": item.overlap_nt / length_a,
                "run_b_overlap_fraction": item.overlap_nt / length_b,
                "run_a_match_count": match_count_a,
                "run_b_match_count": match_count_b,
                "relationship_shape": relationship_shape,
                "overlap_buffer": buffer,
            }
        )
    return rows


_OVERLAP_COLUMNS = (
    "run_a_identifier",
    "run_a_achr",
    "run_a_start",
    "run_a_end",
    "run_b_identifier",
    "run_b_achr",
    "run_b_start",
    "run_b_end",
    "overlap_start",
    "overlap_end",
    "overlap_nt",
    "run_a_overlap_fraction",
    "run_b_overlap_fraction",
    "run_a_match_count",
    "run_b_match_count",
    "relationship_shape",
    "overlap_buffer",
)


def _selected_by_run(combined: Sequence[CombinedLocus]) -> dict[str, set[str]]:
    selected = {"run_a": set(), "run_b": set()}
    for item in combined:
        selected[item.representative.run].add(item.representative.identifier)
    return selected


def _write_canonical_table(
    output: Path,
    name: str,
    bundles: Sequence[_Bundle],
    selected: Mapping[str, set[str]],
    order: Mapping[tuple[str, str], int],
) -> int:
    schema = bundles[0].schemas[name]
    retained: list[tuple[int, int, dict[str, str]]] = []
    serial = 0
    for bundle in bundles:
        with bundle.paths[name].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                identifier = str(row.get("identifier", ""))
                key = (bundle.run, identifier)
                if identifier in selected[bundle.run]:
                    retained.append((order[key], serial, row))
                    serial += 1
    retained.sort(key=lambda item: (item[0], item[1]))
    _write_tsv(output, schema, (item[2] for item in retained))
    return len(retained)


def _gff_identifier(fields: Sequence[str], path: Path, line_number: int) -> str:
    if len(fields) != 9:
        raise ResultComparisonError(f"Expected 9 GFF columns in {path} line {line_number}")
    for attribute in fields[8].split(";"):
        key, separator, value = attribute.partition("=")
        if separator and key.strip().lower() == "id":
            return value.strip()
    raise ResultComparisonError(f"Missing id attribute in {path} line {line_number}")


def _read_gff_features(bundle: _Bundle) -> dict[str, str]:
    features: dict[str, str] = {}
    with bundle.paths["gff"].open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            identifier = _gff_identifier(line.split("\t"), bundle.paths["gff"], line_number)
            if identifier in features:
                raise ResultComparisonError(
                    f"Duplicate GFF feature id {identifier!r} in {bundle.paths['gff']}"
                )
            features[identifier] = line
    return features


def _write_gff(
    output: Path, bundles: Sequence[_Bundle], combined: Sequence[CombinedLocus]
) -> int:
    feature_maps = {bundle.run: _read_gff_features(bundle) for bundle in bundles}
    lines: list[str] = []
    for item in combined:
        locus = item.representative
        try:
            lines.append(feature_maps[locus.run][locus.identifier])
        except KeyError as exc:
            bundle = next(bundle for bundle in bundles if bundle.run == locus.run)
            raise ResultComparisonError(
                f"No GFF feature for called locus {locus.identifier!r} in {bundle.paths['gff']}"
            ) from exc
    with output.open("w", encoding="utf-8", newline="") as handle:
        for line in lines:
            handle.write(line + "\n")
    return len(lines)


def _output_paths(outdir: Path, phase: int) -> dict[str, Path]:
    return {
        "calls": outdir / f"{phase}_calls.tsv",
        "all_clusters": outdir / f"{phase}_all_clusters.tsv",
        "classification_evidence": outdir / f"{phase}_classification_evidence.tsv",
        "phasiRNAs": outdir / f"{phase}_phasiRNAs.tsv",
        "gff": outdir / f"{phase}_PHAS.gff",
        "combined_loci": outdir / f"{phase}_combined_loci.tsv",
        "shared_loci": outdir / f"{phase}_shared_loci.tsv",
        "run_a_only_loci": outdir / f"{phase}_run_a_only_loci.tsv",
        "run_b_only_loci": outdir / f"{phase}_run_b_only_loci.tsv",
        "source_locus_map": outdir / f"{phase}_source_locus_map.tsv",
        "overlap_pairs": outdir / f"{phase}_overlap_pairs.tsv",
        "summary": outdir / "comparison_summary.txt",
        "manifest": outdir / "comparison_manifest.json",
    }


def _prepare_outdir(outdir: Path, input_dirs: Sequence[Path], force: bool) -> None:
    resolved = outdir.resolve()
    for directory in input_dirs:
        input_resolved = directory.resolve()
        if resolved == input_resolved or input_resolved in resolved.parents:
            raise ResultComparisonError(
                "The output directory cannot be either input directory or be created inside one"
            )
    if outdir.exists() and not outdir.is_dir():
        raise ResultComparisonError(f"Output path is not a directory: {outdir}")
    if outdir.is_dir() and any(outdir.iterdir()) and not force:
        raise ResultComparisonError(
            f"Output directory is not empty: {outdir}; use force=True/--force to overwrite known outputs"
        )
    outdir.mkdir(parents=True, exist_ok=True)


def compare_result_directories(
    dir_a: str | Path,
    dir_b: str | Path,
    outdir: str | Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    prefer: str = "auto",
    overlap_buffer: int = 0,
    force: bool = False,
) -> ComparisonResult:
    """Compare two complete PHASIS result directories and write a combined bundle.

    Only loci in the top-level ``<phase>_calls.tsv`` tables participate; PHAS-like
    calls are intentionally excluded.  With ``prefer='auto'``, a non-pooled run
    is preferred over a pooled run (detected from the ``alib``/``ALL_LIBS``
    values).  When both have the same mode, run A is preferred.
    """

    path_a = Path(dir_a).expanduser().resolve()
    path_b = Path(dir_b).expanduser().resolve()
    output_dir = Path(outdir).expanduser().resolve()
    try:
        buffer = int(overlap_buffer)
    except (TypeError, ValueError) as exc:
        raise ResultComparisonError("overlap_buffer must be a non-negative integer") from exc
    if buffer < 0:
        raise ResultComparisonError("overlap_buffer must be a non-negative integer")

    run_a_label = _clean_label(label_a or path_a.name or "run_a", "label_a")
    run_b_label = _clean_label(label_b or path_b.name or "run_b", "label_b")
    a = _load_bundle(path_a, "run_a", run_a_label)
    b = _load_bundle(path_b, "run_b", run_b_label)
    _validate_compatible(a, b)
    preferred = _preferred_run(a, b, prefer)
    overlaps = _find_overlaps(a.loci.values(), b.loci.values(), buffer)
    combined = _combine(a, b, overlaps, preferred)
    _prepare_outdir(output_dir, (path_a, path_b), bool(force))

    paths = _output_paths(output_dir, a.phase)
    labels = {"run_a": a.label, "run_b": b.label}
    locus_rows = _combined_rows(combined, labels)
    _write_tsv(paths["combined_loci"], _COMBINED_COLUMNS, locus_rows)
    _write_tsv(
        paths["shared_loci"],
        _COMBINED_COLUMNS,
        (row for row in locus_rows if row["category"] == "shared"),
    )
    _write_tsv(
        paths["run_a_only_loci"],
        _COMBINED_COLUMNS,
        (row for row in locus_rows if row["category"] == "run_a_only"),
    )
    _write_tsv(
        paths["run_b_only_loci"],
        _COMBINED_COLUMNS,
        (row for row in locus_rows if row["category"] == "run_b_only"),
    )
    _write_tsv(paths["source_locus_map"], _SOURCE_MAP_COLUMNS, _source_map_rows(combined, labels))
    _write_tsv(paths["overlap_pairs"], _OVERLAP_COLUMNS, _overlap_rows(overlaps, buffer))

    selected = _selected_by_run(combined)
    order = {
        (item.representative.run, item.representative.identifier): rank
        for rank, item in enumerate(combined)
    }
    canonical_counts: dict[str, int] = {}
    for name in _CANONICAL_SUFFIXES:
        canonical_counts[name] = _write_canonical_table(
            paths[name], name, (a, b), selected, order
        )
    canonical_counts["gff"] = _write_gff(paths["gff"], (a, b), combined)

    counts = {category: sum(item.category == category for item in combined) for category in (
        "shared", "run_a_only", "run_b_only"
    )}
    summary: dict[str, Any] = {
        "phase": a.phase,
        "run_a": str(path_a),
        "run_b": str(path_b),
        "label_a": a.label,
        "label_b": b.label,
        "mode_a": a.mode,
        "mode_b": b.mode,
        "preferred_run": preferred,
        "overlap_buffer": buffer,
        "input_loci_a": len(a.loci),
        "input_loci_b": len(b.loci),
        "overlap_pairs": len(overlaps),
        "shared_loci": counts["shared"],
        "run_a_only_loci": counts["run_a_only"],
        "run_b_only_loci": counts["run_b_only"],
        "combined_loci": len(combined),
        "collapsed_source_loci": len(a.loci) + len(b.loci) - len(combined),
        "canonical_rows": canonical_counts,
    }
    with paths["summary"].open("w", encoding="utf-8", newline="") as handle:
        for key, value in summary.items():
            if key == "canonical_rows":
                continue
            handle.write(f"{key}\t{value}\n")
        for name, count in canonical_counts.items():
            handle.write(f"{name}_rows\t{count}\n")

    manifest = {
        "format": "phasis-result-comparison",
        "format_version": 1,
        **summary,
        "summary": summary,
        "schemas": {name: list(schema) for name, schema in a.schemas.items()},
        "outputs": {name: path.name for name, path in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return ComparisonResult(a.phase, output_dir, summary, paths)


# A concise alias for callers that do not need to mirror the command name.
compare_results = compare_result_directories


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phasis-compare",
        description="Compare and combine high-confidence loci from two PHASIS result directories.",
    )
    parser.add_argument("dir_a", help="First completed PHASIS result directory")
    parser.add_argument("dir_b", help="Second completed PHASIS result directory")
    parser.add_argument("--outdir", required=True, help="Directory for combined results")
    parser.add_argument("--label-a", default=None, help="Display label for the first run")
    parser.add_argument("--label-b", default=None, help="Display label for the second run")
    parser.add_argument(
        "--prefer",
        choices=("auto", "a", "b", "run_a", "run_b"),
        default="auto",
        help="Representative-source preference (default: auto, preferring non-pooled)",
    )
    parser.add_argument(
        "--overlap-buffer",
        type=int,
        default=0,
        help="Maximum gap in nucleotides still treated as an overlap (default: 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite this tool's known outputs in a non-empty directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        result = compare_result_directories(
            args.dir_a,
            args.dir_b,
            args.outdir,
            label_a=args.label_a,
            label_b=args.label_b,
            prefer=args.prefer,
            overlap_buffer=args.overlap_buffer,
            force=args.force,
        )
    except (OSError, ResultComparisonError) as exc:
        raise SystemExit(f"phasis-compare: error: {exc}") from exc
    print(
        f"Combined {result.summary['input_loci_a']} and "
        f"{result.summary['input_loci_b']} input loci into "
        f"{result.summary['combined_loci']} loci: "
        f"{result.summary['shared_loci']} shared, "
        f"{result.summary['run_a_only_loci']} run-A-only, and "
        f"{result.summary['run_b_only_loci']} run-B-only. "
        f"Results: {result.outdir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
