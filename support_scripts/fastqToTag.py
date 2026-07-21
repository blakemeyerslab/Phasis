from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasis.fastq import (
    FastqTagChunker,
    fastq_output_stem,
    merge_tag_count_chunks,
    write_tag_count_chunk,
)


FASTQ_CHUNK_UNIQUE_TAGS_DEFAULT = 250_000


class TqdmFastqProgress:
    def __init__(self, description):
        self._bar = tqdm(total=None, desc=description, unit="reads")

    def __call__(self, stats, delta, final):
        if delta:
            self._bar.update(delta)
        self._bar.set_postfix(
            retained=stats.reads_retained,
            length_rejects=stats.reads_rejected_length,
            ambiguous_rejects=stats.reads_rejected_ambiguous,
        )
        if final:
            self._bar.close()

    def close(self):
        self._bar.close()

def fastq_to_count_table(fastq_file, *, chunk_unique_tags=FASTQ_CHUNK_UNIQUE_TAGS_DEFAULT):
    """
    Converts plain or gzip-compressed preprocessed sRNA FASTQ to a
    Phasis-compatible tab-separated tag-count table with bounded RAM.
    """
    if int(chunk_unique_tags) < 1:
        raise ValueError("--chunk-unique-tags must be a positive integer")

    output_file = str(Path(fastq_file).with_name(f"{fastq_output_stem(fastq_file)}.tag"))
    progress = TqdmFastqProgress(f"Processing {os.path.basename(fastq_file)}")
    chunk_dir = tempfile.mkdtemp(
        prefix=f".{fastq_output_stem(fastq_file)}.fastq_chunks_",
        dir=str(Path(output_file).parent),
    )
    chunk_paths = []
    try:
        chunker = FastqTagChunker(
            fastq_file,
            max_unique_tags=int(chunk_unique_tags),
            progress_callback=progress,
        )
        for index, counter in enumerate(chunker, start=1):
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:06d}.tsv")
            write_tag_count_chunk(counter, chunk_path)
            chunk_paths.append(chunk_path)
        unique_tags = merge_tag_count_chunks(chunk_paths, output_file)
        stats = chunker.stats
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        progress.close()

    print(f"Output written to {output_file}")
    print(
        "Reads examined:{0} | retained:{1} | rejected_length:{2} | rejected_ambiguous:{3} | "
        "chopped_at_N:{4} | unique_retained_tags:{5}".format(
            stats.reads_examined,
            stats.reads_retained,
            stats.reads_rejected_length,
            stats.reads_rejected_ambiguous,
            stats.reads_chopped_at_n,
            unique_tags,
        )
    )
    return output_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a preprocessed sRNA FASTQ file to a Phasis tag-count table."
    )
    parser.add_argument("fastq_file", help="Input .fastq/.fq file, optionally gzip-compressed")
    parser.add_argument(
        "--chunk-unique-tags",
        type=int,
        default=FASTQ_CHUNK_UNIQUE_TAGS_DEFAULT,
        metavar="N",
        help=(
            "Maximum unique tags held in RAM before writing a temporary chunk "
            f"[default {FASTQ_CHUNK_UNIQUE_TAGS_DEFAULT}]"
        ),
    )
    args = parser.parse_args(argv)
    if args.chunk_unique_tags < 1:
        parser.error("--chunk-unique-tags must be a positive integer")
    return args


if __name__ == "__main__":
    args = parse_args()
    fastq_to_count_table(args.fastq_file, chunk_unique_tags=args.chunk_unique_tags)
