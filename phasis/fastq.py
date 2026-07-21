"""Streaming FASTQ parsing and sRNA tag-count preparation shared by Phasis tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import heapq
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


def preflight_fastq(path: str, *, sample_size: int = RAW_SAMPLE_SIZE) -> FastqStats:
    """Validate a small FASTQ prefix and reject likely raw input before indexing."""
    stats = FastqStats()
    for _header, raw_sequence, _plus, _quality in iter_fastq_records(path):
        stats.reads_examined += 1
        stats.sampled_reads += 1
        if len(raw_sequence.strip()) > MAX_SRNA_LENGTH:
            stats.sampled_long_reads += 1
        if stats.sampled_reads >= sample_size:
            break
    if _raw_input_likely(stats):
        raise RawFastqInputError(_raw_input_message(path, stats))
    return stats


class FastqTagChunker:
    """Yield bounded tag counters while keeping only one chunk in RAM."""

    def __init__(
        self,
        path: str,
        *,
        max_unique_tags: int | None,
        progress_every: int = 1_000_000,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if max_unique_tags is not None and max_unique_tags < 1:
            raise ValueError("max_unique_tags must be positive or None")
        self.path = path
        self.max_unique_tags = max_unique_tags
        self.progress_every = progress_every
        self.progress_callback = progress_callback
        self.stats = FastqStats()

    def __iter__(self) -> Iterator[Counter[str]]:
        counter: Counter[str] = Counter()
        previous_progress = 0

        for _header, raw_sequence, _plus, _quality in iter_fastq_records(self.path):
            self.stats.reads_examined += 1
            if self.stats.sampled_reads < RAW_SAMPLE_SIZE:
                self.stats.sampled_reads += 1
                if len(raw_sequence.strip()) > MAX_SRNA_LENGTH:
                    self.stats.sampled_long_reads += 1
            sequence, reason, chopped = preprocess_srna_sequence(raw_sequence)
            if chopped:
                self.stats.reads_chopped_at_n += 1
            if reason == "length":
                self.stats.reads_rejected_length += 1
            elif reason == "ambiguous":
                self.stats.reads_rejected_ambiguous += 1
            else:
                counter[sequence] += 1
                self.stats.reads_retained += 1

            if self.stats.reads_examined == RAW_SAMPLE_SIZE and _raw_input_likely(self.stats):
                raise RawFastqInputError(_raw_input_message(self.path, self.stats))
            if (
                self.progress_callback
                and self.progress_every > 0
                and self.stats.reads_examined - previous_progress >= self.progress_every
            ):
                self.progress_callback(self.stats, self.stats.reads_examined - previous_progress, False)
                previous_progress = self.stats.reads_examined
            if self.max_unique_tags is not None and len(counter) >= self.max_unique_tags:
                yield counter
                counter = Counter()

        if _raw_input_likely(self.stats):
            raise RawFastqInputError(_raw_input_message(self.path, self.stats))
        if counter:
            yield counter
        if self.progress_callback:
            self.progress_callback(self.stats, self.stats.reads_examined - previous_progress, True)


def count_fastq_tags(
    path: str,
    *,
    progress_every: int = 1_000_000,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Counter[str], FastqStats]:
    """Stream FASTQ records, filter before counting, and return retained tag counts."""
    chunker = FastqTagChunker(
        path,
        max_unique_tags=None,
        progress_every=progress_every,
        progress_callback=progress_callback,
    )
    counter = Counter()
    for counter in chunker:
        # No chunk boundary is used in this compatibility helper, so this loop
        # runs once and does not duplicate the completed counter in memory.
        pass
    return counter, chunker.stats


def write_tag_count(counter: Counter[str], output_path: str) -> str:
    """Write deterministic Phasis-compatible ``sequence<TAB>count`` output."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for sequence, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{sequence}\t{count}\n")
    return output_path


def write_tag_count_chunk(counter: Counter[str], output_path: str) -> str:
    """Write one lexically sorted tag-count chunk for an external merge."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for sequence in sorted(counter):
            handle.write(f"{sequence}\t{counter[sequence]}\n")
    return output_path


def _next_tag_count(handle: TextIO) -> tuple[str, int] | None:
    line = handle.readline()
    if not line:
        return None
    sequence, count = line.rstrip("\n").split("\t")
    return sequence, int(count)


def merge_tag_count_chunks(chunk_paths: list[str], output_path: str) -> int:
    """Externally merge lexically sorted FASTQ tag-count chunks.

    The output is sorted by sequence rather than by abundance. Tag-count inputs
    have no ordering requirement, and this keeps memory bounded by one tag per
    chunk instead of by every distinct sequence in the library.
    """
    handles = [open(path, "r", encoding="utf-8") for path in chunk_paths]
    heap: list[tuple[str, int, int]] = []
    unique_tags = 0
    try:
        for index, handle in enumerate(handles):
            item = _next_tag_count(handle)
            if item is not None:
                sequence, count = item
                heapq.heappush(heap, (sequence, index, count))

        with open(output_path, "w", encoding="utf-8") as output:
            while heap:
                sequence, index, count = heapq.heappop(heap)
                total_count = count
                item = _next_tag_count(handles[index])
                if item is not None:
                    next_sequence, next_count = item
                    heapq.heappush(heap, (next_sequence, index, next_count))

                while heap and heap[0][0] == sequence:
                    _same_sequence, same_index, same_count = heapq.heappop(heap)
                    total_count += same_count
                    item = _next_tag_count(handles[same_index])
                    if item is not None:
                        next_sequence, next_count = item
                        heapq.heappush(heap, (next_sequence, same_index, next_count))

                output.write(f"{sequence}\t{total_count}\n")
                unique_tags += 1
    finally:
        for handle in handles:
            handle.close()
    return unique_tags
