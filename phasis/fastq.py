"""Streaming FASTQ parsing and sRNA tag-count preparation shared by Phasis tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import os
from typing import Callable, Iterator, TextIO


MIN_SRNA_LENGTH = 18
MAX_SRNA_LENGTH = 35
RAW_SAMPLE_SIZE = 1_000
RAW_LONG_FRACTION = 0.80


class FastqFormatError(ValueError):
    """Raised for malformed or incomplete FASTQ input."""


class RawFastqInputError(ValueError):
    """Raised when input strongly resembles untrimmed, non-sRNA FASTQ."""


@dataclass
class FastqStats:
    reads_examined: int = 0
    reads_retained: int = 0
    reads_rejected_length: int = 0
    reads_rejected_ambiguous: int = 0
    reads_chopped_at_n: int = 0
    sampled_reads: int = 0
    sampled_long_reads: int = 0

    def as_dict(self, unique_tags: int) -> dict[str, int]:
        return {
            "reads_examined": self.reads_examined,
            "reads_retained": self.reads_retained,
            "reads_rejected_length": self.reads_rejected_length,
            "reads_rejected_ambiguous": self.reads_rejected_ambiguous,
            "reads_chopped_at_n": self.reads_chopped_at_n,
            "unique_retained_tags": unique_tags,
        }


ProgressCallback = Callable[[FastqStats, int, bool], None]


def open_fastq_text(path: str) -> TextIO:
    """Open plain or gzip-compressed FASTQ as text without pre-counting records."""
    path = str(path)
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def fastq_output_stem(path: str) -> str:
    base = os.path.basename(str(path))
    if base.lower().endswith(".gz"):
        base = base[:-3]
    for suffix in (".fastq", ".fq"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    stem, dot, _ = base.rpartition(".")
    return stem if dot else base


def iter_fastq_records(path: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield validated FASTQ records and fail clearly on malformed input."""
    with open_fastq_text(path) as handle:
        record_number = 0
        while True:
            header = handle.readline()
            if not header:
                return
            record_number += 1
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()
            if not sequence or not plus or not quality:
                raise FastqFormatError(
                    f"Incomplete FASTQ record {record_number} in {path!r}; expected four lines per record."
                )
            if not header.startswith("@") or not plus.startswith("+"):
                raise FastqFormatError(
                    f"Malformed FASTQ record {record_number} in {path!r}; expected '@' header and '+' separator."
                )
            sequence = sequence.rstrip("\r\n")
            quality = quality.rstrip("\r\n")
            if len(sequence) != len(quality):
                raise FastqFormatError(
                    f"Malformed FASTQ record {record_number} in {path!r}; sequence and quality lengths differ."
                )
            yield header.rstrip("\r\n"), sequence, plus.rstrip("\r\n"), quality


def preprocess_srna_sequence(sequence: str) -> tuple[str | None, str | None, bool]:
    """Apply legacy N handling, then reject invalid or out-of-range sRNA sequences."""
    sequence = sequence.strip().upper()
    chopped = False
    half_length = len(sequence) // 2
    n_position = sequence.find("N", 0, half_length)
    if n_position != -1:
        sequence = sequence[:n_position]
        chopped = True
    if any(base not in {"A", "C", "G", "T"} for base in sequence):
        return None, "ambiguous", chopped
    if not MIN_SRNA_LENGTH <= len(sequence) <= MAX_SRNA_LENGTH:
        return None, "length", chopped
    return sequence, None, chopped


def _raw_input_message(path: str, stats: FastqStats) -> str:
    fraction = stats.sampled_long_reads / max(1, stats.sampled_reads)
    return (
        f"FASTQ {path!r} appears to contain untrimmed, non-sRNA reads: "
        f"{stats.sampled_long_reads}/{stats.sampled_reads} sampled reads ({fraction:.0%}) exceed "
        f"{MAX_SRNA_LENGTH} nt. -libformat Q expects preprocessed sRNA reads ({MIN_SRNA_LENGTH}-{MAX_SRNA_LENGTH} nt). "
        "Do not truncate raw reads in Phasis. Perform adapter/quality trimming externally, convert with "
        "support_scripts/fastqToTag.py, then rerun with -libformat T."
    )


def _raw_input_likely(stats: FastqStats) -> bool:
    return bool(stats.sampled_reads) and (
        stats.sampled_long_reads / stats.sampled_reads >= RAW_LONG_FRACTION
    )


def count_fastq_tags(
    path: str,
    *,
    progress_every: int = 1_000_000,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Counter[str], FastqStats]:
    """Stream FASTQ records, filter before counting, and return retained tag counts."""
    counter: Counter[str] = Counter()
    stats = FastqStats()
    previous_progress = 0

    for _header, raw_sequence, _plus, _quality in iter_fastq_records(path):
        stats.reads_examined += 1
        if stats.sampled_reads < RAW_SAMPLE_SIZE:
            stats.sampled_reads += 1
            if len(raw_sequence.strip()) > MAX_SRNA_LENGTH:
                stats.sampled_long_reads += 1
        sequence, reason, chopped = preprocess_srna_sequence(raw_sequence)
        if chopped:
            stats.reads_chopped_at_n += 1
        if reason == "length":
            stats.reads_rejected_length += 1
        elif reason == "ambiguous":
            stats.reads_rejected_ambiguous += 1
        else:
            counter[sequence] += 1
            stats.reads_retained += 1

        if stats.reads_examined == RAW_SAMPLE_SIZE and _raw_input_likely(stats):
            raise RawFastqInputError(_raw_input_message(path, stats))
        if progress_callback and progress_every > 0 and stats.reads_examined - previous_progress >= progress_every:
            progress_callback(stats, stats.reads_examined - previous_progress, False)
            previous_progress = stats.reads_examined

    if _raw_input_likely(stats):
        raise RawFastqInputError(_raw_input_message(path, stats))
    if progress_callback:
        progress_callback(stats, stats.reads_examined - previous_progress, True)
    return counter, stats


def write_tag_count(counter: Counter[str], output_path: str) -> str:
    """Write deterministic Phasis-compatible ``sequence<TAB>count`` output."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for sequence, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{sequence}\t{count}\n")
    return output_path
