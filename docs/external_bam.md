# Use Bowtie 1 or BWA Alignments With Phasis

Phasis normally maps reads with HISAT2. If you already use Bowtie 1 or
BWA-backtrack, this optional workflow turns their BAM alignments into native
Phasis candidate clusters, then runs the normal classification stage.
You do not need it for a [standard Phasis analysis](../README.md#run-a-real-example).

The supported route is:

**Collapsed sRNA library -> aligner -> BAM -> candidate clusters -> Phasis calls.**

It uses a supporting adapter, not an aligner-selection option on the main
`phasis` command. It currently supports **individual libraries, single-end
reads, and full-length ungapped alignments**. Exact matches remain the default;
opt-in mismatch and reported-placement settings are explained below. Use Bowtie **1**, or BWA
`aln`/`samse`; BWA-MEM/MEM2 are not supported by this adapter.

## 1. Prepare Tools and Inputs

Install Phasis following the [installation instructions](../README.md#installation).
Use a source checkout containing `support_scripts/bam_to_candidates.py`.
This supporting script is kept outside the installed `phasis` package; retain
the checkout to run it. From its root, add the optional BAM dependency and save
the absolute script path before changing to your analysis workspace:

```bash
python -m pip install -e '.[bam]'
bam_importer="$PWD/support_scripts/bam_to_candidates.py"
```

You also need SAMtools and your chosen aligner on `PATH`: `bowtie` and
`bowtie-build`, or `bwa`. Keep HISAT2 installed in the Phasis environment:
the existing classification startup checks still require its executable,
although this workflow does **not** run HISAT2 mapping or require a prior
HISAT2-based Phasis analysis.

The supported interchange format is BAM. Convert SAM to BAM with
`samtools view -b -o sample.bam sample.sam`; CRAM is not a supported input
contract. The BAM can be unsorted and need not be indexed.

Prepare a directory containing:

| File | Requirement |
| --- | --- |
| `genome.fa` | Uncompressed reference FASTA for the correct assembly. Use this same reference for mapping, importing, and calling. |
| `sample.fas` | Collapsed sRNA FASTA: one entry per distinct sequence, with its original name. Retain this file for the final calling step. |
| `sample.qnames.tsv` | Tab-separated read-name, original-sequence, and abundance map, described below. |

Reads must already be adapter/quality trimmed. Collapsing means counting
identical sequences, not discarding their abundances. For example, a sequence
observed 12 times appears once in the FASTA and has abundance 12 in the map:

```text
qname	sequence	abundance
tag_1	ACGTTGCAACGTTGCAACGTTGCA	12
```

Here the corresponding FASTA header is `>tag_1`. Use actual tabs in the map.
`qname` must exactly match the name in the BAM; sequences must be in their
original read orientation, and abundances must be positive integers. Include
all input tags in the map if you need total-input abundance accounting.

Alternatively, if every FASTA header already ends in its true abundance
(for example `>tag_1|12`), you can use `--abundance-in-qname` instead of
`--qname-map` at import. Never add a count of 1 to collapsed sequences just to
make the import succeed: that changes the evidence used for scoring.

Set the absolute input directory, then create a fresh analysis workspace:

```bash
data_dir=/absolute/path/to/inputs
aligner=bowtie1  # Change to bwa-aln for the BWA route.
mkdir "${aligner}_run"
cd "${aligner}_run"
samtools faidx "$data_dir/genome.fa"
```

Keep this shell open for the remaining commands. Use a different workspace for
each aligner. If importing an existing BAM, copy it here as `sample.bam` and
skip step 2 **only if its generation settings meet the requirements below**.

## 2. Generate Alignments

Choose **one** recipe. Both preserve reported alternative placements and request
exact matches without gaps. Do not add a MAPQ filter or keep only primary
alignments: doing so can make repetitive reads appear unique.

### Bowtie 1

```bash
bowtie-build "$data_dir/genome.fa" genome_bt1
bowtie -f -v 0 -k 25 --best --strata -S -p 4 \
  genome_bt1 "$data_dir/sample.fas" sample.sam
samtools view -b -o sample.bam sample.sam
```

`-v 0` requests exact matches. `-k 25` provides enough reported placements for
the importer's default hit cutoff. Keep every alignment in the resulting BAM.

### BWA-backtrack

Use `sample.fastq` containing exactly the same collapsed sequences and read
names as `sample.fas`, not an uncollapsed raw-read library. If converting from
FASTA, retain names and sequences unchanged and assign constant dummy qualities.
With zero mismatches/gaps and trimming disabled below, these qualities do not
change accepted exact-match evidence.

For example, this conversion uses the optional `pysam` dependency installed
above and supports wrapped FASTA sequences:

```bash
python - "$data_dir/sample.fas" "$data_dir/sample.fastq" <<'PY'
import sys
import pysam

with pysam.FastxFile(sys.argv[1]) as reads, open(sys.argv[2], "w") as out:
    for read in reads:
        out.write(f"@{read.name}\n{read.sequence}\n+\n{'I' * len(read.sequence)}\n")
PY
```

```bash
bwa index "$data_dir/genome.fa"
bwa aln -n 0 -o 0 -e 0 -l 1024 -N -q 0 -t 4 \
  "$data_dir/genome.fa" "$data_dir/sample.fastq" > sample.sai
bwa samse -n 25 "$data_dir/genome.fa" sample.sai \
  "$data_dir/sample.fastq" > sample.sam
samtools view -b -o sample.bam sample.sam
```

`aln -n 0` requests zero edit distance, `-o 0` disables gap opens, `-N` requests
non-iterative search, and `-q 0` disables quality trimming. `samse -n 25` allows
reporting alternatives in the `XA` tag. Preserve `XA`, `X0` and `X1` hit-count tags;
the adapter uses them to recover alternative placements and check completeness.

## 3. Generate Phasis Candidate Clusters

From the same workspace:

```bash
python "$bam_importer" \
  --bam sample.bam --aligner "$aligner" --library sample \
  --qname-map "$data_dir/sample.qnames.tsv" \
  --reference "$data_dir/genome.fa" --phase 24 --cores 4 \
  --output-dir candidates/sample
```

If abundances are encoded in the read names, replace the `--qname-map` argument
with `--abundance-in-qname`. The `--library sample` ID must match the FASTA
basename `sample.fas` used in the next step.

The adapter calls Phasis's existing clustering and scoring functions. The main
outputs to check are:

| File under `candidates/sample/` | Purpose |
| --- | --- |
| `sample.24-PHAS.candidate.clusters` | Native candidate file for classification. |
| `IMPORT_SUCCESS.json` | Written only after the import and candidate-completeness checks pass. |
| `alignment_metrics.json` | Accepted/rejected read and placement counts. |
| `import_manifest.json` | Input paths, checksums, and parameters. |
| `resolved_parameters.json` | Resolved Phasis settings. |
| `sample.accepted.bam` and index | Normalized accepted placements for inspection. |

Inspect rejection counts before proceeding. Do not treat a partially written
candidate file without `IMPORT_SUCCESS.json` as a successful import.

## 4. Call PHAS Loci

Use the original library and the candidate file with `-steps class`:

```bash
phasis -steps class -phase 24 -libformat F -cores 4 \
  -reference "$data_dir/genome.fa" --reference_id_mode preserve \
  -libs "$data_dir/sample.fas" \
  -class_cluster_files candidates/sample/sample.24-PHAS.candidate.clusters \
  --outdir results
```

This uses normal calling parameters. Final tables and plots are written to
`results/`; see the [output guide](../README.md#understand-the-outputs).
`--reference_id_mode preserve` keeps the reference names used by the adapter.

For 21-*PHAS*, change `--phase 24` at import, `-phase 24` at calling, and the
candidate filename to 21. Generate candidates in a new output directory.
For multiple libraries, import each separately with its own basename and output
directory, then list all matching original FASTAs and candidate files in the
calling command. Do not add `--pool_libraries` to this workflow.

## Requirements and Why They Matter

| Requirement | Reason |
| --- | --- |
| Original abundance for every collapsed sequence | Setting all counts to 1 changes depth, cluster evidence, and scores. |
| Full-length ungapped alignments | Clipping, gaps, and splicing change read-coordinate/register interpretation and remain unsupported. The adapter checks substitutions against the FASTA and rejects ambiguous sequence bases. |
| Complete reported placements and optional tags | Removing alternative alignments biases hit counts and the interpretation of uniquely mapped evidence. |
| Mapper reporting limit at least `--maxhits` | The native default retains **fewer than 25 reported hits** (1-24), excluding 25 or more. A smaller mapper limit can conceal highly repetitive reads. |
| Matching reference assembly, names, and lengths | Incorrect coordinate systems invalidate clustering and reference comparisons. Renaming chromosomes is not liftover. |
| Separate workspaces and new/empty import directories | Prevents mixing candidates or cached intermediates from different alignment runs. |

Keep alignment commands and tool versions with the BAM. In particular, a Bowtie
BAM alone cannot reliably reveal that alternatives were omitted. Equal filters
also do not guarantee equal aligner search sensitivity: the hit cutoff concerns
**reported placements**, not a known true number of genomic copies.

### Which settings can I change?

You choose the aligner and its generation settings **within this supported
contract**. Phasis still applies additional filters; it does not adopt an
arbitrary user-defined mapping policy unchanged.

- `--maxhits N` is configurable (integer N >= 2). A tag is retained only if its
  reported multiplicity is less than N. Set Bowtie `-k` or BWA `samse -n` to
  at least N, or use exhaustive reporting where supported. For example,
  `--maxhits 2` retains only tags with one reported placement.
- `--multimapping all-reported` is the default: retain all accepted placements
  for tags below the hit cutoff. `--multimapping unique-only` requires exactly
  one reported placement **before** mismatch or other alignment filtering.
  It is equivalent to a hit cutoff of 2 for selection, but records the policy
  explicitly. It does not mean proven genomic uniqueness and must not be used
  with BAMs prefiltered to primary-only alignments.
- `--mindepth` filters original tag abundance (default 1). Native candidate
  scoring also retains its existing filters, including `--unique-ratio`
  (default 0.2). Its native unique-hit criterion counts tags with at most two
  reported hits; the audit metric `unique_sequences` instead means exactly one.
- Abundance is retained in full at each accepted placement, as in native
  Phasis. Fractional weighting, random assignment of multimappers, and MAPQ-based
  selection are not configurable policies in this adapter.
- `--mismatches N` accepts a nonnegative integer (default 0). It is a
  post-alignment substitution limit, analogous to native `-mismat` for ungapped
  evidence, not an instruction to search for additional alignments. Counts are
  recomputed against the supplied FASTA; if present, `NM` must agree. The
  original reported multiplicity is retained even if some placements exceed
  the mismatch limit. Later `-mismat` cannot restore rejected input or refilter
  precomputed candidate files. Clipping, indels, splice junctions, ambiguous
  sequence bases, and `X` CIGAR operations remain unsupported.

Missing alignments cannot be recovered from the BAM. In particular, matching
`NH` or `X0`/`X1` counts verifies consistency of the supplied reporting, not exhaustive
enumeration of every valid genomic placement. Filtering away alignments before
import is not a substitute for configuring `--maxhits`.

### Optional mismatch-tolerant input

Keep the exact-match recipe above unless you intend to change the evidence.
For example, to allow up to one substitution, regenerate the alignments before
importing. Use new output directories and retain commands/tool versions.

```bash
# Bowtie 1: omit --strata so worse mismatch strata are not deliberately hidden.
bowtie -f -v 1 -k 25 --best -S -p 4 \
  genome_bt1 "$data_dir/sample.fas" sample.mm1.sam
samtools view -b -o sample.mm1.bam sample.mm1.sam
aligner=bowtie1
```

Or, using the BWA FASTQ and index from the earlier recipe:

```bash
bwa aln -n 1 -o 0 -e 0 -l 1024 -N -q 0 -t 4 \
  "$data_dir/genome.fa" "$data_dir/sample.fastq" > sample.mm1.sai
bwa samse -n 25 "$data_dir/genome.fa" sample.mm1.sai \
  "$data_dir/sample.fastq" > sample.mm1.sam
samtools view -b -o sample.mm1.bam sample.mm1.sam
aligner=bwa-aln
```

Then import the selected BAM:

```bash
python "$bam_importer" \
  --bam sample.mm1.bam --aligner "$aligner" --library sample \
  --qname-map "$data_dir/sample.qnames.tsv" --reference "$data_dir/genome.fa" \
  --phase 24 --cores 4 --mismatches 1 --multimapping all-reported \
  --output-dir candidates_mm1/sample
```

Use the generated candidate file in the classification command from step 4,
in a new calling workspace. Set `-mismat 1` there to record consistent settings;
filtering has already happened during import. To keep only single-reported-hit
tags, change the import policy to `--multimapping unique-only` and use another
new output directory. Neither policy changes original integer abundances.

BWA mismatch-enabled import requires `X1` as well as `X0` for reads below the
hit cutoff: `X0` counts best hits, while `X1` counts reported suboptimal hits.
The importer uses their sum before mismatch filtering and checks it against
primary plus distinct `XA`/secondary placements. Missing or incomplete evidence
is an error, not permission to call a tag unique. `aln -N` and sufficient
`samse -n` reporting are required for this recipe. Different aligner search
strategies can still produce different reported alignments.

Bowtie 1's `-v` mode supports limits 0-3; the importer's nonnegative limit does
not expand the aligner's capabilities. See the [Bowtie manual](https://bowtie-bio.sourceforge.net/manual.shtml)
and [BWA-backtrack source](https://github.com/lh3/bwa/blob/v0.7.17/bwase.c)
for reporting semantics.

## Troubleshooting and Advanced Details

### Abundance or alignment-completeness errors

Unknown/duplicate read names, sequence conflicts, and multiple accepted names
for the same collapsed sequence fail import. Check the original library and
map rather than changing counts or dropping records to bypass validation.
Below the hit cutoff, BWA's primary plus distinct `XA`/secondary placements
must agree with `X0 + X1` when `X1` is present. Exact-only legacy input may omit
`X1`, in which case the count must agree with `X0`; nonzero mismatch tolerance
requires `X1` below the cutoff. Regenerate an incomplete BAM with sufficient reporting.

The adapter also accepts `--aligner hisat2` when `NH` tags and all reported
placements are retained. Normal Phasis users do not need this import route.
Arbitrary BAM configurations remain outside the supported contract; adjustable
substitution tolerance does not enable gapped or clipped evidence.

### Reference-name differences

If names differ but the underlying assembly and contigs are identical, supply
`--chrom-map mapping.tsv` with `old_id` (BAM name) and `clean_id` (FASTA name)
columns. Empty/duplicate source IDs, many-to-one translations among BAM contigs,
and length mismatches fail. The adapter never guesses name conversions.
Name/length checks are not proof of assembly identity; retain assembly metadata
and reference checksums.

### Memory and temporary disk

Input BAMs need not be sorted or indexed; the adapter sorts by read name using
private temporary files. For node-local sorting scratch, add
`--scratch /path/to/existing/scratch` to the import command. Sorting uses up to
256 MiB per sorting thread, plus memory for read maps and native dictionaries.
Those structures still grow with the dataset; this is not constant-memory
processing. Importing one library at a time limits concurrent memory demand.

### Alignment filtering and audit files

Only single-end flags 0, 16, 256, and 272 contribute evidence. Supplementary,
QC-failed, and duplicate-flagged records are excluded; paired-end mapped input
fails. Only full-length `M` or `=` CIGAR operations are accepted. Conflicting
duplicate tags fail; equivalent placements are deduplicated.
Out-of-contig placements are rejected and counted without clipping or wrapping
coordinates. Valid placements retain the original reported-hit denominator.

The normalized BAM records the reference-checked `NM`, `NH` as reported multiplicity, `XC` as
abundance, and `MAPQ=0`; MAPQ here is not an aligner confidence score. Use the
candidate file, not this BAM, as the interface to classification. Coordinates
are converted to native one-based leftmost positions; Phasis scoring handles
strand-specific phase offsets.

The adapter's tab-separated `.sum` is an import audit, not the legacy native
summary layout. Normalization scripts must explicitly choose the appropriate
denominator from `alignment_metrics.json`. A successful empty candidate file
can mean no candidates survived; verify this before interpreting empty results.

### Further help

```bash
python "$bam_importer" --help
python -m unittest tests.test_external_bam
```

Run tests from the source checkout. The auxiliary tool lives entirely in
`support_scripts/bam_to_candidates.py`; it is not a main Phasis CLI option.
See the official [Bowtie 1 manual](https://bowtie-bio.sourceforge.net/manual.shtml)
and [BWA manual](https://bio-bwa.sourceforge.net/bwa.shtml) for mapper options.

For an optional small end-to-end validation with real installed HISAT2,
Bowtie 1, BWA-backtrack, and SAMtools executables:

```bash
python tests/run_external_bam_integration.py --output-dir /tmp/phasis-bam-check
python tests/run_external_bam_integration.py --output-dir /tmp/phasis-bam-mm1-check --mismatches 1
python tests/run_external_bam_integration.py --output-dir /tmp/phasis-bam-unique-check --mismatches 1 --multimapping unique-only
```

The directory must not already exist. This generates synthetic inputs locally,
runs alignment/import/calling, checks native-versus-imported HISAT2 evidence,
and writes commands, tool versions, and `INTEGRATION_SUCCESS.json`. It performs
no downloads. The repeat control records aligner-reporting differences rather
than assuming all valid genomic placements were enumerated.
