from __future__ import annotations

import gzip
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from phasis import libprep
from phasis.fastq import FastqFormatError, RawFastqInputError, count_fastq_tags, preflight_fastq


def write_fastq(path: Path, sequences: list[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences, start=1):
            handle.write(f"@read_{index}\n{sequence}\n+\n{'I' * len(sequence)}\n")


def read_tag_table(path: Path) -> dict[str, int]:
    return {
        sequence: int(count)
        for sequence, count in (line.rstrip("\n").split("\t") for line in path.read_text().splitlines())
    }


class FastqProcessingTests(unittest.TestCase):
    def test_plain_and_gzip_fastq_variants_have_identical_counts(self):
        sequences = ["A" * 21, "C" * 24, "A" * 21, "G" * 17, "T" * 36, "A" * 20 + "R"]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [
                Path(tmpdir) / "sample.fastq",
                Path(tmpdir) / "sample.fq",
                Path(tmpdir) / "sample.fastq.gz",
                Path(tmpdir) / "sample.fq.gz",
            ]
            for path in paths:
                write_fastq(path, sequences)

            results = [count_fastq_tags(str(path)) for path in paths]

        expected = {"A" * 21: 2, "C" * 24: 1}
        for counts, stats in results:
            self.assertEqual(dict(counts), expected)
            self.assertEqual(stats.reads_examined, 6)
            self.assertEqual(stats.reads_retained, 3)
            self.assertEqual(stats.reads_rejected_length, 2)
            self.assertEqual(stats.reads_rejected_ambiguous, 1)

    def test_malformed_fastq_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.fastq"
            path.write_text("@read\nACGT\n+\n", encoding="utf-8")
            with self.assertRaisesRegex(FastqFormatError, "Incomplete FASTQ record"):
                count_fastq_tags(str(path))

    def test_likely_raw_reads_are_rejected_instead_of_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.fastq"
            write_fastq(path, ["A" * 76] * 4)
            with self.assertRaisesRegex(RawFastqInputError, "adapter/quality trimming"):
                count_fastq_tags(str(path))

    def test_preflight_rejects_raw_reads_without_counting_the_full_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.fastq"
            write_fastq(path, ["A" * 76] * 1_005)
            with self.assertRaisesRegex(RawFastqInputError, "1000/1000"):
                preflight_fastq(str(path))

    def test_reports_progress_while_streaming(self):
        updates = []
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.fastq"
            write_fastq(path, ["A" * 21] * 3)
            count_fastq_tags(
                str(path), progress_every=1, progress_callback=lambda stats, delta, final: updates.append(
                    (stats.reads_examined, delta, final)
                )
            )

        self.assertEqual(updates[:3], [(1, 1, False), (2, 1, False), (3, 1, False)])
        self.assertEqual(updates[-1], (3, 0, True))

    def test_internal_q_conversion_matches_fastq_to_tag_helper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fastq = tmp / "sample.fastq.gz"
            write_fastq(fastq, ["A" * 21, "C" * 24, "A" * 21, "G" * 17])
            fas = tmp / "internal.fas"
            summary = tmp / "internal.sum"
            original_mindepth = libprep.mindepth
            try:
                libprep.mindepth = 1
                libprep.fastq_process(str(fastq), str(fas), str(summary))
            finally:
                libprep.mindepth = original_mindepth

            helper = Path(__file__).resolve().parents[1] / "support_scripts" / "fastqToTag.py"
            subprocess.run(
                [sys.executable, str(helper), str(fastq), "--chunk-unique-tags", "2"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            internal = {sequence: count for sequence, count in libprep.fas_records(str(fas))}
            self.assertEqual(internal, read_tag_table(tmp / "sample.tag"))
            self.assertFalse(list(tmp.glob(".sample.fastq_chunks_*")))

    def test_chunked_q_conversion_merges_counts_without_retaining_all_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fastq = tmp / "sample.fastq"
            sequences = ["A" * 21, "C" * 21, "G" * 21, "A" * 21, "C" * 21]
            write_fastq(fastq, sequences)
            fas = tmp / "chunked.fas"
            summary = tmp / "chunked.sum"
            original_mindepth = libprep.mindepth
            original_chunk_size = getattr(libprep.rt, "fastq_chunk_unique_tags", None)
            try:
                libprep.mindepth = 1
                libprep.rt.fastq_chunk_unique_tags = None
                with mock.patch.dict(os.environ, {"PHASIS_FASTQ_CHUNK_UNIQUE_TAGS": "2"}, clear=False):
                    libprep.fastq_process(str(fastq), str(fas), str(summary))
            finally:
                libprep.mindepth = original_mindepth
                libprep.rt.fastq_chunk_unique_tags = original_chunk_size

            self.assertEqual(
                {sequence: count for sequence, count in libprep.fas_records(str(fas))},
                {"A" * 21: 2, "C" * 21: 2, "G" * 21: 1},
            )
            self.assertFalse(list(tmp.glob(".sample.fastq_chunks_*")))


if __name__ == "__main__":
    unittest.main()
