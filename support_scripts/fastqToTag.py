from __future__ import annotations

import os
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasis.fastq import count_fastq_tags, fastq_output_stem, write_tag_count


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

def fastq_to_count_table(fastq_file):
    """
    Converts plain or gzip-compressed preprocessed sRNA FASTQ to a
    Phasis-compatible tab-separated tag-count table.
    """
    output_file = str(Path(fastq_file).with_name(f"{fastq_output_stem(fastq_file)}.tag"))
    progress = TqdmFastqProgress(f"Processing {os.path.basename(fastq_file)}")
    seq_counter, stats = count_fastq_tags(fastq_file, progress_callback=progress)
    write_tag_count(seq_counter, output_file)

    print(f"Output written to {output_file}")
    print(
        "Reads examined:{0} | retained:{1} | rejected_length:{2} | rejected_ambiguous:{3} | "
        "chopped_at_N:{4} | unique_retained_tags:{5}".format(
            stats.reads_examined,
            stats.reads_retained,
            stats.reads_rejected_length,
            stats.reads_rejected_ambiguous,
            stats.reads_chopped_at_n,
            len(seq_counter),
        )
    )
    return output_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fastqToTag.py <input.fastq|input.fq|input.fastq.gz|input.fq.gz>")
    else:
        fastq_file = sys.argv[1]
        fastq_to_count_table(fastq_file)
