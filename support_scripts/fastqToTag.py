import sys
import os
from collections import defaultdict
from tqdm import tqdm

def chop_sequence(sequence):
    """
    Chops the sequence at the first occurrence of 'N' if it appears
    within the first half of the sequence.
    """
    half_length = len(sequence) // 2
    n_position = sequence.find('N', 0, half_length)
    if n_position != -1:
        return sequence[:n_position]
    return sequence

def fastq_to_count_table(fastq_file):
    """
    Converts a FASTQ file to a tab-separated count table (.tag file),
    handling sequences with 'N' as described.
    """
    base_name = os.path.splitext(fastq_file)[0]
    output_file = f"{base_name}.tag"

    seq_counter = defaultdict(int)
    removed_sequences = 0

    with open(fastq_file, "r") as fq:
        # Count total reads for progress bar (optional heavy):
        total = None
        try:
            # Attempt to estimate record count for tqdm
            lines = sum(1 for _ in open(fastq_file, 'r'))
            total = lines // 4
        except:
            total = None

        fq.seek(0)
        reader = tqdm(total=total, desc="Processing reads") if total else None
        try:
            while True:
                header = fq.readline().strip()
                if not header:
                    break
                seq = fq.readline().strip()
                fq.readline()  # plus line
                fq.readline()  # quality line

                processed_sequence = chop_sequence(seq)
                if 18 <= len(processed_sequence) <= 35 and 'N' not in processed_sequence:
                    seq_counter[processed_sequence] += 1
                else:
                    removed_sequences += 1

                if reader:
                    reader.update(1)
        finally:
            if reader:
                reader.close()

    with open(output_file, "w") as out:
        for seq, count in sorted(seq_counter.items(), key=lambda x: x[1], reverse=True):
            out.write(f"{seq}\t{count}\n")

    print(f"Output written to {output_file}")
    print(f"Sequences removed or chopped due to 'N' presence: {removed_sequences}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input.fastq>")
    else:
        fastq_file = sys.argv[1]
        fastq_to_count_table(fastq_file)
