# Phasis - Phased sRNA Cluster Discovery and Annotation

**Version:** v2.8  
**Updated:** 2026-07-03

Phasis is a parallelized tool for large-scale analysis of small RNA (sRNA) libraries. It supports:

- *De novo* discovery of ***PHAS* loci** and precursor transcripts
- Register-resolved interpretation of candidate loci as *PHAS*, *PHAS*-like, or non-*PHAS*
- Summarization, visualization, and annotation of *PHAS* loci across libraries

---

## Installation

### 1) Create an environment

Conda is recommended:
```bash
conda create -n phasis python=3.12 -y
conda activate phasis
conda install "numpy=1.26.4" "scikit-learn=1.3.0" -y
```

### 2) Install external tools

Phasis requires `hisat2` and `samtools` on your `PATH`:
```bash
conda install -c bioconda hisat2 samtools -y
```

### 3) Install Phasis

From the Phasis repository root:
```bash
python -m pip install -U pip
python -m pip install -e .
```

Check that the command is available:
```bash
phasis -h
```

---

## Executable Example: Maize Tag-Count Libraries From GEO

This example downloads four maize tag-count libraries and the B73 RefGen v2 genome, then runs 21-*PHAS* and 24-*PHAS* analyses.

```bash
mkdir -p phasis_example
cd phasis_example

# Retrieve GEO tag-count libraries.
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3466697&format=file&file=GSM3466697%5F7570%5Fchopped%2Etxt%2Egz" -O sTP_dcl5_1_2.0.tag.gz
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4180401&format=file&file=GSM4180401%5FTP%5FW23%5F2%5F0%5F1%5Fchopped%2Etxt%2Egz" -O W23_2.0_1.tag.gz
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4180402&format=file&file=GSM4180402%5FTP%5FW23%5F2%5F0%5F2%5Fchopped%2Etxt%2Egz" -O W23_2.0_2.tag.gz
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3466699&format=file&file=GSM3466699%5F7569%5Fchopped%2Etxt%2Egz" -O sTR_dcl5_1_2.0.tag.gz

# Download the maize B73 RefGen v2 / AGPv2 genome.
wget https://download.maizegdb.org/B73_RefGen_v2/B73_RefGen_v2.fa.gz

# Detect 21-PHAS loci.
phasis -mindepth 1 -phase 21 -libformat T \
  -reference B73_RefGen_v2.fa.gz -cores 12 -maxhits 25 \
  -libs sTR_dcl5_1_2.0.tag.gz W23_2.0_2.tag.gz sTP_dcl5_1_2.0.tag.gz W23_2.0_1.tag.gz

# Detect 24-PHAS loci from the same run directory.
# The HISAT2 index and processed libraries can be reused safely.
phasis -mindepth 1 -phase 24 -libformat T \
  -reference B73_RefGen_v2.fa.gz -cores 12 -maxhits 25 \
  -libs sTR_dcl5_1_2.0.tag.gz W23_2.0_2.tag.gz sTP_dcl5_1_2.0.tag.gz W23_2.0_1.tag.gz
```

---

## Quick Start With Your Own Data

### 21-*PHAS*
```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 21 -cores 0
```

### 24-*PHAS*
```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 24 -cores 0
```

### Pooled-library analysis

By default, Phasis analyzes libraries individually. To pool all input libraries into one virtual library before candidate detection, add `--pool_libraries`:

```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 24 --pool_libraries
```

---

## How Phasis Writes Files

Phasis uses two locations:

1. **Run directory**: the directory where you run `phasis`
   - Intermediate files
   - `index/` with the HISAT2 index
   - `processed_libraries/` with reusable processed libraries
   - `phasis.mem`, the cache used to decide which steps can be reused

2. **Output directory**: selected with `--outdir`
   - Final outputs for the selected phase
   - Default: `{phase}_results`, such as `21_results` or `24_results`

Keeping 21-*PHAS* and 24-*PHAS* runs in the same run directory allows safe reuse of the index and processed libraries.

Completed Phase II text intermediates are gzip-compressed by default to reduce disk usage in large runs. The cache still treats these as the same logical intermediates, so restart behavior is preserved. Add `--no_compress_intermediates` if you need the intermediate TSV/TAB files to remain plain text.

---

## Input Library Formats

Phasis accepts:

- FASTA (`-libformat F`): plain text or `.gz`
- Tag-count (`-libformat T`): plain text or `.gz`
- FASTQ (`-libformat Q`): quality-controlled FASTQ, plain text or `.gz`

The reference FASTA passed to `-reference` can also be plain text or gzip-compressed.

For any supported input format, Phasis stores a processed `.fas.gz` copy under `processed_libraries/` so later runs can reuse it.

### Run directly from FASTQ
```bash
phasis -libs sample.fastq.gz other_sample.fastq.gz -reference genome.fa.gz -libformat Q
```

### Convert FASTA to tag-count
```bash
python support_scripts/fastaToTag.py sample.fasta
```

Then run Phasis:
```bash
phasis -libs *.tag -reference genome.fa -libformat T
```

---

## Core Outputs

For each phase, Phasis writes the main outputs to `--outdir`.

| Output | Description |
| --- | --- |
| `{phase}_calls.tsv` | Main table of high-confidence *PHAS* calls. |
| `{phase}_classification_evidence.tsv` | Register-resolved evidence table for *PHAS*, *PHAS*-like, and non-*PHAS* interpretations. |
| `{phase}_all_clusters.tsv` | Full table of evaluated clusters, including calls that did not pass final reporting. |
| `{phase}_PHAS.gff` | Genome annotation file for detected *PHAS* loci. |
| `{phase}_PHAS.pdf` | Heatmap summarizing *PHAS* classification across libraries. |
| `{phase}_Abundance_PHAS.pdf` | Abundance heatmap for final *PHAS* loci. |
| `{phase}_Abundance_PHAS_and_nonPHAS.pdf` | Combined abundance heatmap for *PHAS* and non-*PHAS* signal. |
| `{phase}_Howell_scores.pdf` | Heatmaps of Peak Howell score and strict Peak Howell score. |
| `{phase}_PHAS_locus_plots/` | Per-locus diagnostic plots showing abundance context and score/register context. |
| `{phase}_phasiRNAs.tsv` | Per-locus phased-register phasiRNA table for final *PHAS* loci. |

### What is *PHAS*-like?

The Register-Resolved Locus Interpretation Layer (RRL) evaluates the register-level evidence after candidate detection. High-confidence *PHAS* calls require sufficient exact-register support. Candidates with a visible phased structure but weaker exact-register support are retained as *PHAS*-like in `{phase}_classification_evidence.tsv`; they are not merged into the high-confidence `{phase}_calls.tsv` table.

### Individual locus plots

Each diagnostic plot contains:

- Top panel: abundance context, with sRNAs colored by size and separated by strand.
- Bottom panel: Howell score/register context across candidate phased windows.
- Phase-colored marks: reads assigned to the called phased register.
- Register anchor: the highest phased-score position used to interpret the locus.

These plots are intended for manual inspection of candidate architecture, opposite-strand partner support, and secondary or overlapping phased windows.

---

## Classifying Existing Candidate Clusters

If you already have `*.candidate.clusters` files from a previous Phasis run or an alternative candidate-generation workflow, you can run only the classification/output stage:

```bash
phasis -mindepth 1 -phase 24 -libformat T \
  -libs *.tag -reference genome.fa -cores 0 \
  -steps class \
  -class_cluster_files previous_run/*.candidate.clusters
```

If `-class_cluster_files` is omitted, Phasis tries to infer the expected cluster files from `-libs`, `-phase`, and the current run directory.

---

## Common Options

| Option | Meaning |
| --- | --- |
| `-libs` | Input libraries to process. |
| `-reference` | Genome or transcriptome reference FASTA. |
| `-libformat` | `F` FASTA, `T` tag-count, or `Q` FASTQ. |
| `-phase` | Phasing length, commonly `21` or `24`. |
| `-cores` | `0` uses most free cores; `>0` sets an exact core count. |
| `-maxhits` | Value passed as `-k` to HISAT2. |
| `-mindepth` | Minimum depth for p-value computation. |
| `-uniqueRatioCut` | Minimum proportion of uniquely mapped reads. |
| `-max_complexity` | Maximum complexity filter. |
| `-mismat` | Post-alignment mismatch filter while parsing alignments. |
| `-min_Howell_score` | Minimum Howell score used during classification/output filtering. |
| `--pool_libraries` | Pool all input libraries into one virtual library before candidate detection. |
| `--no_compress_intermediates` | Keep completed Phase II intermediate TSV/TAB files uncompressed. |
| `--outdir` | Final output directory; supports `{phase}` in the name. |

---

## Advanced Options

### Cleanup

Use cleanup modes from a Phasis run root only:

```bash
phasis -cleanup
phasis -cleanup_all
```

- `-cleanup` deletes intermediates but preserves `index/`, final results, and index-related cache metadata.
- `-cleanup_all` deletes intermediates, `index/`, and `phasis.mem`, while preserving final results directories.

### Plot staging

For large runs or shared filesystems, individual locus plots can be staged through local scratch:

```bash
phasis ... --plot_staging auto
phasis ... --plot_staging local
phasis ... --plot_staging direct
```

The default `auto` mode stages plots when Phasis detects an HPC-style environment or remote/distributed output path.

### Reference ID handling

Phasis can preserve original FASTA IDs or use compact numeric IDs in the indexed `.clean.fa` reference:

```bash
phasis ... --reference_id_mode preserve
phasis ... --reference_id_mode numeric
```

---

## FASTA Headers: Non-Integer Chromosome IDs

Very long or complex FASTA headers can increase memory use. If needed, replace genome headers with compact numeric IDs:

```bash
python support_scripts/replace_genome_headers.py genome.fa new_genome.fa equivalence.tsv
```

This writes:

- `new_genome.fa` with compact chromosome IDs
- `equivalence.tsv` mapping old IDs to new IDs

---

## Comparing *PHAS* Loci Between Runs

Use `phasMatch.py` to compare genomic overlap between two *PHAS* result tables:

```bash
python support_scripts/phasMatch.py <phasis.result.tsv> <alternative_predictions.tsv>
```

The default matching window uses genomic overlap with a +/-300 nt flank.

---

## Troubleshooting

- Phasis currently targets NumPy `1.26.4` and scikit-learn `1.3.0`.
- Avoid NumPy `2.x` with the current Phasis release because scikit-learn `1.3.0` is not compatible with that series.
- If command-line examples produce no plots, check whether the run produced final *PHAS* or *PHAS*-like calls and whether `--plot_staging` copied staged plots back to `--outdir`.
- If reference IDs are unexpected in outputs, use `--reference_id_mode preserve` for the next run.

---

## Authors

- Thales Henrique Cherubino Ribeiro - thalescherubino@gmail.com
- Atul Kakrana - kakrana@gmail.com
- Blake Meyers - bcmeyers@ucdavis.edu
