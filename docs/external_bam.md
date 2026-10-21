# External BAM to native Phasis candidates

`support_scripts/bam_to_candidates.py` (also `python -m phasis.external_bam`)
imports collapsed, single-end short-RNA BAM evidence and calls the existing
`cluster_build.clusterprocess` and `cluster_scoring.scoringprocess` functions.
It does not replace Phasis clustering, filtering, window statistics or scoring.
Calling remains the normal `phasis -steps class` operation. No pooling is done.

Install Phasis from this branch and the optional `pysam` dependency (tested with
pysam 0.23.3, Python 3.11.9). BAM sorting is performed in private temporary files;
input BAMs may be coordinate sorted, query-name sorted or unsorted and need no
index. A matching FASTA with a `.fai` is required. Create it once with
`samtools faidx reference.fa`. Output directories must be new/empty. Input BAM,
reference, abundance-map and chromosome-map SHA-256 values and all parameters
are recorded; stale native caches are never implicitly adopted.

## Example

Use a separate output directory for every library and aligner:

```bash
python support_scripts/bam_to_candidates.py \
  --bam W23_2.0_1.bam --aligner bowtie1 --library W23_2.0_1 \
  --qname-map W23_2.0_1.qnames.tsv \
  --reference B73.clean.fa --phase 24 --maxhits 25 --mismatches 0 \
  --mindepth 1 --cores 4 --output-dir bowtie1/W23_2.0_1

phasis -steps class -phase 24 -libformat F \
  -reference B73.clean.fa --reference_id_mode preserve \
  -libs W23_2.0_1.fas -cores 4 -mindepth 1 -maxhits 25 -mismat 0 \
  -phasisScoreCutoff 250 -min_Howell_score 12.5 -max_complexity 0.3 \
  -class_cluster_files bowtie1/W23_2.0_1/W23_2.0_1.24-PHAS.candidate.clusters \
  --outdir bowtie1_calls
```

For the six-library comparison supply all six FASTAs and all six candidate
files explicitly to `-libs` and `-class_cluster_files`. Individual libraries
remain separate; do not set `--pool_libraries`. Use a fresh calling working
directory per aligner. The candidate format includes all 18 native fields and
can be moved without the importer pickles. Preserve `IMPORT_SUCCESS.json`, the
import manifest, resolved parameters, `.sum`, and candidate checksum alongside
it. A zero-byte candidate can represent no candidates; downstream behavior with
an entirely empty experiment must be reviewed rather than reported as success.

## Abundance and reference contract

The preferred QNAME map is a tab-separated file with columns:

```
qname	sequence	abundance
seq_1|12	ACGTTGCAACGTTGCAACGTTGCA	12
```

The sequence is in the original read orientation and abundance is a positive
integer. Unknown/duplicate QNAMEs, sequence conflicts, and multiple accepted
QNAMEs representing the same collapsed sequence fail. If original abundance is
reliably encoded in native terminal `|count` names, explicitly choose
`--abundance-in-qname` instead. Abundance is never silently set to one. The map
also supplies denominators when unmapped reads are absent from a BAM.

BAM reference names and lengths must match the supplied FASTA. An explicit
`--chrom-map` TSV with `old_id` and `clean_id` can translate BAM names to FASTA
names; ambiguous many-to-one mappings and length mismatches fail. No heuristic
name stripping is performed. SAM zero-based internal positions become native
one-based LEFTMOST positions, including reverse-strand reads; the native
scoring stage handles its own phase offsets. Normalized reverse SEQ is the
reverse complement of the original read, as required by SAM.

## Strict alignment contract

This first importer supports **zero mismatches only** (`--mismatches 0`). It
requires full-read matches, allowing only CIGAR M or = with full query length.
Soft/hard clips, insertions, deletions, skipped regions, padding and X operations
are excluded. Accepted sequence is checked against the indexed FASTA; NM, when
present, must also be zero. Ambiguous bases are rejected. Thus acceptance does
not depend on aligner-specific MAPQ or on the order of optional SAM tags.

Only flags 0, 16, 256 and 272 supply evidence, matching native single-end flag
handling. Secondary alignments count as alternative placements; supplementary,
QC-failed and duplicate-flagged records do not. Paired-end mapped input fails.
Repeated identical placements are deduplicated; conflicting duplicate tags fail.
Each collapsed tag's abundance is retained at each accepted placement, matching
native Phasis evidence, without multiplying the distinct-tag denominator.

**Native maxhits is an exclusive bound:** the current parser accepts `hits <
maxhits`. Therefore default 25 retains 1–24 reported placements and excludes
25 or more. It is not a promise to retain exactly 25 multimappers. Native
`uniqueRatioCut` scoring uses the native hit definition (including its existing
`UNIQRATIO_HIT=2` criterion); the descriptive mapping metric `unique_sequences`
in this importer means exactly one reported placement.

Choose `--aligner` explicitly:

- `hisat2`: named NH is required. All reported primary/secondary placements must
  be supplied; NH and their deduplicated count must agree below the cutoff.
- `bowtie1`: multiplicity is the count of deduplicated reported placements. The
  BAM MUST retain all alignments reported by Bowtie using `-k >= maxhits` (or
  exhaustive reporting), without primary-only/MAPQ filtering. Missing secondary
  evidence cannot be inferred from a Bowtie BAM, so the generation command is a
  required provenance constraint.
- `bwa-aln`: use BWA-backtrack `aln`/`samse`, not MEM/MEM2. X0 is required. Below
  the cutoff, XA alternatives are expanded and validated; primary plus unique
  XA/secondary placements must equal X0. Missing XA evidence fails rather than
  being misclassified as unique. X0 at/above the cutoff is rejected even when
  the repetitive read has no XA list.

When importing non-exact raw BAMs, reported multiplicity is evaluated before
mismatch/CIGAR filtering, matching the native parser's reported-hit convention.
The controlled comparison avoids this complication by generating exact-only
alignments in all arms. A BAM with incomplete reporting cannot establish the
true genomic multiplicity of a read; do not prefilter it before import.

## Alignment generation for the controlled comparison

All arms use the same native tag-count FASTA sequences, abundances and reference.
Native tag-count preprocessing filters minimum abundance but imposes no extra
length filter (the 18–35 nt FASTQ rule is a different input path).

- HISAT2 2.2.1: `--no-softclip --no-spliced-alignment --score-min L,0,0 -k 25 -f`.
  The first two flags and hit cap match normal native mapping. Zero minimum
  score is an explicit exact-alignment restriction for this controlled test;
  native Phasis normally applies default `-mismat 0` after alignment. This
  extra mapper restriction must be disclosed when comparing historical outputs.
- Bowtie 1.3.1: `-f -v 0 -k 25 --best --strata -S`.
- BWA 0.7.17: `aln -n 0 -o 0 -e 0 -l 1024 -N -q 0`, followed by
  `samse -n 25`. The FASTQ representation uses identical sequences/QNAMEs and
  constant dummy qualities; zero mismatches, zero gaps and no trimming make
  quality differences irrelevant to accepted evidence.

BWA-backtrack is selected for these short reads; `-N` requests non-iterative
search, integer `-n 0` requires zero edit distance, and XA reporting is made
large enough to recover every retained placement. Aligners still have different
search implementations. Equal downstream filters do not prove equal sensitivity;
compare observed mapping evidence rather than assuming equivalence.

Primary documentation:
- https://daehwankimlab.github.io/hisat2/manual/
- https://bowtie-bio.sourceforge.net/manual.shtml
- https://bio-bwa.sourceforge.net/bwa.shtml

## Outputs, resource use and tests

Each import emits an indexed, coordinate-sorted `<library>.accepted.bam`, native
`_<phase>.dict` and `.count` parser artifacts, native cluster/scoring artifacts,
`<library>.<phase>-PHAS.candidate.clusters`, `.sum`, alignment metrics and a
success manifest. Normalized BAM uses NM=0, NH=reported multiplicity, XC=abundance,
and MAPQ=0; it is evidence for this adapter, not a claim of comparable aligner
mapping confidence. It is not intended for the old native positional-tag SAM
parser; candidates are the supported interface to `-steps class`.

Alignment input streams by QNAME after disk sorting. Explicit maps, seen names
and the native nested dictionaries still consume memory proportional to tag
counts/retained evidence; set compute resources accordingly. Processing one
library per import limits peak memory. Failed imports intentionally retain their
work directory for diagnosis and do not write an IMPORT_SUCCESS marker.

Unit tests: `python -m unittest discover -s tests -p test_external_bam.py -v`.
The experiment additionally tests actual HISAT2/Bowtie/BWA binaries on a synthetic
genome, and compares native HISAT2 parser dictionaries, candidates and calls
with the importer route for exactly the same alignment evidence. Test reports
and exact commands are part of the experiment handoff.

The synthetic integration fixture includes a known 27-copy exact tag. The tested
HISAT2 invocation reported only three placements for that low-complexity control,
while Bowtie/BWA reached or exceeded the exclusion boundary. This is an explicit
search/reporting difference: the adapter cannot infer unreported genomic hits.
The shared cutoff therefore means **reported** multiplicity, matching native
Phasis; it is not a guarantee that all retained tags have fewer than 25 true
copies. The integration test verifies every unambiguous control and records the
repetitive control separately instead of forcing artificial agreement.

Completeness checks additionally require one native clustering output per
input library/chromosome key, successful native parallel workers, each expected
scored chunk, and exact cluster/row content preservation from scored chunks to
candidate export. These checks allow scientifically filtered empty chunks but
fail missing or partial worker/export work.
