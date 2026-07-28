import sys
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasis.progress import tqdm

def chop_sequence(sequence):
    """
    Chops the sequence at the first occurrence of 'N' if it appears
    within the first half of the sequence.

    Args:
    sequence (str): The DNA sequence to chop.

    Returns:
    str: The chopped sequence or the original if no 'N' found in the first half.
    """
    half_length = len(sequence) // 2
    n_position = sequence.find('N', 0, half_length)
    if n_position != -1:
        return sequence[:n_position]
    return sequence

def fasta_to_count_table(fasta_file):
    """
    Converts a FASTA file to a tab-separated count table file,
    handling sequences with 'N' as described.

    Args:
    fasta_file (str): Path to the input FASTA file.
    """
    # Output file name
    base_name = os.path.splitext(fasta_file)[0]
    output_file = f"{base_name}.tag"

    # Read sequences and count them
    seq_counter = defaultdict(int)
    removed_sequences = 0

    with open(fasta_file, "r") as fasta:
        sequence = ""
        for line in tqdm(fasta, desc="Processing sequences"):
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    processed_sequence = chop_sequence(sequence)
                    if 18 <= len(processed_sequence) <= 35 and 'N' not in processed_sequence:
                        seq_counter[processed_sequence] += 1
                    else:
                        removed_sequences += 1
                    sequence = ""
            else:
                sequence += line
        if sequence:
            processed_sequence = chop_sequence(sequence)
            if 18 <= len(processed_sequence) <= 35 and 'N' not in processed_sequence:
                seq_counter[processed_sequence] += 1
            else:
                removed_sequences += 1

    # Write counts to the output file
    with open(output_file, "w") as out:
        for seq, count in sorted(seq_counter.items(), key=lambda x: x[1], reverse=True):
            out.write(f"{seq}\t{count}\n")

    print(f"Output written to {output_file}")
    print(f"Sequences removed or chopped due to 'N' presence: {removed_sequences}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input.fasta>")
    else:
        fasta_file = sys.argv[1]
        fasta_to_count_table(fasta_file)
