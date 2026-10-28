# Phasis - Phased sRNA Cluster Discovery and Annotation

**Version:** v2.8.4

Phasis identifies phased small-RNA-producing (*PHAS*) loci from small-RNA
(sRNA) libraries and a reference genome or transcriptome. It reports
high-confidence *PHAS* calls, retains *PHAS*-like candidates for follow-up,
and produces annotation tables, heatmaps, and individual locus plots.

For a first analysis, install Phasis and follow the downloadable example below.
You do not need to change classifiers, choose another aligner, or tune advanced
parameters to get started.

**Start here:** [Install](#installation) | [Run the example](#run-a-real-example) |
[Understand the outputs](#understand-the-outputs) | [Use your own data](#use-your-own-data)

**Go further:** [Manage runs](#manage-runs-and-outputs) |
[Advanced workflows](docs/advanced_usage.md) |
[Bowtie 1 / BWA tutorial](docs/external_bam.md) | [Troubleshooting](#troubleshooting)

## Software History

The original [PHASIS](https://github.com/atulkakrana/PHASIS), described by
Kakrana et al. (2017), remains available for reproducibility. **Phasis 2 is a
complete rewrite** with a redesigned architecture, rather than an incremental
update of the original software.
The maintained implementation is
[blakemeyerslab/Phasis](https://github.com/blakemeyerslab/Phasis).

## Installation

Phasis supports Python 3.10 through 3.12 on Linux and macOS.
The recommended Conda environment includes HISAT2, the default aligner, and
SAMtools. Use Git to clone the source, or `curl` and `unzip` for the ZIP route
below. The example also uses `wget`.

The Phasis 2.8.4 [Bioconda recipe](https://github.com/bioconda/bioconda-recipes/pull/70852)
has been submitted and is awaiting CI and maintainer approval. Direct
installation from the public Bioconda channel is not yet available.
For now, Conda installs the dependencies and pip installs Phasis itself.
On a cluster, see [HPC installation and batch jobs](docs/advanced_usage.md#hpc-installation-and-batch-jobs)
before choosing environment and cache directories.

### 1. Create an environment

```bash
conda create -n phasis --override-channels -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 "numpy=1.26.4" "scikit-learn=1.3.2" \
  hisat2 "samtools>=1.10" wget -y
conda activate phasis
```

### 2. Download and install Phasis

Choose **one** of these download methods, using a new directory so you do not
overwrite an existing checkout or analysis.

**Clone with Git:**

```bash
git clone https://github.com/blakemeyerslab/Phasis.git
cd Phasis
```

**Or download and extract a ZIP:**

```bash
curl -fL https://github.com/blakemeyerslab/Phasis/archive/refs/heads/main.zip -o Phasis-main.zip
unzip Phasis-main.zip
cd Phasis-main
```

Both methods download the maintained `main` branch. Then, from the directory
containing `pyproject.toml`, install Phasis into the activated environment:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Keep this source directory in place: `-e` installs in editable mode, so the
installed command uses these files. The optional supporting scripts also
remain in this directory; they are not installed as Phasis commands.

### 3. Verify the installation

```bash
phasis -version
phasis -h
python -m pip check
hisat2 --version
samtools --version
python -c "from phasis.deps_check import require_dependencies; require_dependencies()"
```

If you already have a source checkout, run the installation commands from its
root instead of cloning again. Keep analysis files in a separate directory.
The commands above check installation without running an analysis. Continue
with the [real example](#run-a-real-example) to test an analysis end to end.

## Run a Real Example

This example downloads four maize [tag-count libraries](#supported-library-formats)
and the maize (*Zea mays*) B73 reference genome, assembly RefGen v2 (AGPv2),
from [MaizeGDB](https://download.maizegdb.org/B73_RefGen_v2/), then runs
21-*PHAS* and 24-*PHAS* analyses.
It uses real sequencing data rather than a small test dataset, so allow time
and disk space for downloading the genome, building its index, and processing
the libraries. Adjust `-cores 12`
to the resources available to your job.

**Run only one Phasis analysis at a time in each working directory.** In this
example, wait for the 21-*PHAS* analysis to finish before starting the
24-*PHAS* analysis. Runs share intermediate files and caches, so different
`--outdir` values do not make simultaneous runs in the same working directory
safe. Use separate working directories for analyses that must run simultaneously.

Run the commands in a new analysis directory outside the source checkout:

```bash
mkdir -p ~/phasis_example
cd ~/phasis_example

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

# Detect 24-PHAS loci, reusing the index and processed libraries.
phasis -mindepth 1 -phase 24 -libformat T \
  -reference B73_RefGen_v2.fa.gz -cores 12 -maxhits 25 \
  -libs sTR_dcl5_1_2.0.tag.gz W23_2.0_2.tag.gz sTP_dcl5_1_2.0.tag.gz W23_2.0_1.tag.gz
```

Results appear in `21_results/` and `24_results/`. Start with the calls table
and the individual locus plots described below. Libraries remain separate by
default; pooling is an optional analysis choice.

## Understand the Outputs

In the filenames below, `{phase}` is the phase length in nucleotides selected
with `-phase`, usually 21 or 24 for 21-nt or 24-nt phased sRNAs. For example,
`-phase 24` produces `24_calls.tsv`.

For each phase, Phasis writes final results to `--outdir` (default:
`21_results/` or `24_results/`).

| Output | What it tells you |
| --- | --- |
| `{phase}_calls.tsv` | Main table of high-confidence *PHAS* calls. Start here. |
| `{phase}_classification_evidence.tsv` | Register-resolved evidence for *PHAS*, *PHAS*-like, and non-*PHAS* interpretations. |
| `{phase}_all_clusters.tsv` | Evaluated clusters, including candidates that did not pass final reporting. |
| `{phase}_PHAS.gff` | Coordinates of detected *PHAS* loci for genome annotation or browsing. |
| `{phase}_PHAS.pdf` | Individual-library classification heatmap for loci called in at least one library. |
| `{phase}_Abundance_PHAS.pdf` | Abundance heatmap for loci called as *PHAS* at least once. |
| `{phase}_Abundance_PHAS_and_nonPHAS.pdf` | Abundance heatmap including non-*PHAS* signal at those loci. |
| `{phase}_Howell_scores.pdf` | Individual-library [Howell-score heatmaps](docs/scoring.md), showing the main and strict scores separately. |
| `{phase}_PHAS_locus_plots/` | Diagnostic plots for inspecting individual loci. |
| `{phase}_phasiRNAs.tsv` | Individual phasiRNA sequences, abundances, and assignments to main or secondary phased units within called *PHAS* loci. See the [phasiRNA table guide](docs/phasirna_output.md). |
| `{phase}_PHAS_like/` | Separate *PHAS*-like calls, evidence, GFF, phasiRNAs, and plots. |

### Individual phasiRNAs and phased units

One *PHAS* locus can contain more than one supported **phased unit**: a group
of phased windows describing a local phasiRNA-producing region. These units
can include overlapping alternative registers or spatially separated regions
within the same locus. `{phase}_phasiRNAs.tsv` records individual sRNA
sequences and abundances together with their library, position, and phased-unit
assignment, rather than treating every locus as one uniform region.

Use this table to inspect which phasiRNAs support each unit and to prepare
sequences for downstream analyses such as target prediction. The
[phasiRNA table guide](docs/phasirna_output.md) explains the columns, sequence
orientation, and how to avoid double-counting repeated assignments.

### What are *PHAS* and *PHAS*-like loci?

A ***PHAS* locus** is a genomic region whose transcripts give rise to phased
small interfering RNAs (phasiRNAs). Successive processing of a double-stranded
RNA precursor by Dicer-like (DCL) enzymes produces sRNAs in a regular register,
usually at 21- or 24-nucleotide intervals. See
[Allen et al. (2005)](https://doi.org/10.1016/j.cell.2005.04.004) for the mechanistic basis of phased siRNA production.

DCL processing produces sRNA duplexes with characteristic **two-nucleotide
3' overhangs**, giving paired products from opposite strands a specific
positional relationship. This geometry provides additional evidence for
DCL-dependent processing; see the primary study on
[RDR6-dependent precursor synthesis and phased duplex production](https://pubmed.ncbi.nlm.nih.gov/21930511/).

After candidate detection, the Register-Resolved Locus Interpretation Layer
(RRL) evaluates exact-register support, competing registers, and supported
phased windows. When evaluating opposite-strand partners, it checks their
compatibility with the expected two-nucleotide overhang geometry. An observed
opposite-strand partner is not required for every high-confidence call.

***PHAS*** calls pass the detection and interpretation criteria, including
sufficient exact-register support. ***PHAS*-like** candidates have evidence of
phased structure but weaker or ambiguous support for a high-confidence
interpretation. They are retained separately for follow-up and are **not**
added to the high-confidence calls table. The evidence table and diagnostic
plots help explain each candidate's interpretation; neither label alone is
experimental confirmation of its biogenesis.

### Read an individual locus plot

![Example individual PHAS locus plot](docs/images/phas_locus_plot_example.png)

- **Top panel:** sRNA abundance across the locus, separated by strand. Diamonds
  represent mapped sRNAs; color indicates length and fill distinguishes
  uniquely and multiply mapped reads. Phase-colored halos identify reads
  assigned to the interpreted register.
- **Bottom panel:** [Howell score and register context](docs/scoring.md#scores-in-the-plots) for candidate phased windows.
  The red point marks the highest phasing score position (HPSP), used as the register
  anchor. Hollow gray points are nearby windows that were not promoted;
  rose-colored windows are promoted secondary phased windows.
- **Blue region:** the main phased unit in this example.
- **Sidebar:** register support, final class, opposite-strand partner status,
  coherent extension, and secondary-window evidence.

Use these plots to inspect phased structure, opposite-strand support, and
secondary or overlapping windows rather than relying on a score alone.

### Understanding the scores

The **Phasis score** summarizes statistical evidence for phasing across candidate
windows. The **Howell score** measures support for regularly spaced sRNA
positions using their abundance and the number of occupied phase cycles.
They are different measures, not interchangeable names for the same score.
See the [scoring guide](docs/scoring.md) for how Phasis calculates them,
the distinction between the main and strict Howell scores, and their use in
filtering and register-resolved interpretation.

## Use Your Own Data

You need a **reference genome or transcriptome in FASTA format**, supplied
with `-reference`, and one or more sRNA libraries supplied with `-libs`.
The reference can be uncompressed (for example, `genome.fa` or `transcripts.fa`)
or gzip-compressed (`genome.fa.gz` or `transcripts.fa.gz`); no manual
decompression is needed.

### Supported library formats

Phasis accepts FASTA, FASTQ, and tag-count libraries, either uncompressed or
gzip-compressed. **All libraries must come from adapter-trimmed,
quality-controlled sRNA reads.** Phasis is not an adapter or quality trimmer.
Select the format with `-libformat`:

| Library format | Option | Contents |
| --- | --- | --- |
| FASTA | `-libformat F` | sRNA sequences. |
| FASTQ | `-libformat Q` | sRNA sequences and quality scores. |
| Tag-count | `-libformat T` | Each distinct sRNA sequence and its read count. |

**Tag-count is recommended for already processed sRNA libraries.** Repeated
reads are stored as one sequence with a count, reducing file size and avoiding
the need for Phasis to count duplicate sequences during input preparation.
The original read abundance is preserved. This is a convenience for storage
and preprocessing, not a different analysis method; you can also provide
preprocessed FASTA or FASTQ directly.

A tag-count file has two tab-separated columns and no header: sequence first,
then read count. For example:

```text
ACGTACGTACGTACGTACGTA	120
TGCATGCATGCATGCATGCAT	8
```

Here, the first sequence was observed 120 times and the second 8 times.
Use whole-number read counts, not normalized abundances. Select this format
with `-libformat T`; files can be uncompressed (for example, `sample.tag`)
or gzip-compressed (`sample.tag.gz`).

To convert quality-controlled libraries to tag-count format, use the supporting
scripts from your downloaded Phasis source directory (replace the input paths
with your own):

```bash
python support_scripts/fastaToTag.py /path/to/sample_fasta.fasta
python support_scripts/fastqToTag.py /path/to/sample_fastq.fastq.gz
```

These commands create `sample_fasta.tag` and `sample_fastq.tag` beside their
input files. The FASTA helper requires uncompressed input with one record per
read; it does not recover abundance counts from collapsed FASTA headers.
The FASTQ helper accepts plain or gzip-compressed input. These helpers retain
18-35 nt sequences and apply ambiguous-base handling; they do not replace
adapter trimming or quality control. See [format conversion](docs/advanced_usage.md#convert-library-formats)
for details and the FASTQ helper's memory-control option.

Direct FASTQ input also expects 18-35 nt sRNAs and checks for invalid,
ambiguous, short, and long reads. Input that appears to contain raw sequencing
reads triggers an error rather than being silently truncated.

### Choose the phase length

From your analysis directory, use `-phase 21` or `-phase 24`:

```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 21 -cores 4
phasis -libs *.tag -libformat T -reference genome.fa -phase 24 -cores 4
```

For preprocessed FASTQ libraries:

```bash
phasis -libs sample.fastq.gz other_sample.fastq.gz \
  -libformat Q -reference genome.fa.gz -phase 24 -cores 4
```

### Choose individual or pooled evidence

By default, Phasis analyzes libraries individually. To combine their evidence
into one virtual library, add `--pool_libraries`:

```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 24 \
  -cores 4 --pool_libraries --outdir 24_results_pooled
```

Pooled mode produces tables, annotations, and individual locus plots, but not
cross-library heatmaps because there is only one virtual library. Neither
evidence mode is guaranteed to recover every locus found by the other.

### Common options

| Option | Meaning |
| --- | --- |
| `-libs` | Input libraries. |
| `-reference` | Reference genome or transcriptome FASTA. |
| `-libformat` | `F` for FASTA, `T` for tag-count, or `Q` for preprocessed FASTQ. |
| `-phase` | Phasing length, commonly 21 or 24. |
| `-cores` | CPU count; `0` uses all available cores within detected allocation limits. |
| `--pool_libraries` | Combine libraries into one virtual library before candidate detection. |
| `--outdir` | Final output directory; supports `{phase}` in its name. |

Run `phasis -h` for all options. Start with the normal filtering settings before
considering [advanced parameter changes](docs/advanced_usage.md#filtering-options).

## Manage Runs and Outputs

| Location | Contents |
| --- | --- |
| **Run directory:** where you execute `phasis` | Intermediates, `index/`, `processed_libraries/`, and the `phasis.mem` cache. |
| **Output directory:** selected by `--outdir` | Final tables, annotations, and plots. |

Keeping 21-*PHAS* and 24-*PHAS* analyses in the same run directory allows reuse
of the index and processed libraries. Phasis stores processed library copies
as `.fas.gz` files and checks its cache to decide which steps can be reused.
Changing `--outdir` changes the location of final results, not the run directory.

Completed Phase II text intermediates are gzip-compressed by default to save
disk space while preserving restart/cache behavior. Use
`--no_compress_intermediates` only when you need plain-text intermediates.

For separate experiments, keep separate run directories. Before deleting
intermediates, read the [cleanup instructions](docs/advanced_usage.md#cleanup):
cleanup preserves final results but removes files that would otherwise be reused.

## Go Further

These workflows are optional; they are not prerequisites for a standard analysis.

| Task | Guide |
| --- | --- |
| Classify existing candidate clusters without repeating discovery | [Reuse candidate files](docs/advanced_usage.md#classify-existing-candidate-clusters); [18-field format and example](docs/candidate_clusters.md) |
| Use Bowtie 1 or BWA-aln (`bwa aln` + `bwa samse`) alignments instead of native HISAT2 mapping | [Alternative-aligner tutorial](docs/external_bam.md) |
| Combine completed runs or evaluate calls against known loci | [Compare results](docs/advanced_usage.md#combine-completed-runs) |
| Convert library formats or manage chromosome names | [Input preparation](docs/advanced_usage.md#input-preparation) |
| Tune memory, temporary disk use, or shared-filesystem performance | [HPC and resource tuning](docs/advanced_usage.md#hpc-and-resource-tuning) |

The alternative-aligner workflow uses `support_scripts/bam_to_candidates.py` for
individual-library, full-length ungapped alignments (exact matches by default).
Optional mismatch tolerance and a single-reported-placement filter are described
in the tutorial. It reuses Phasis clustering,
scoring, and classification; it does not replace the default workflow.
It does not require a prior HISAT2 mapping run. The
[tutorial](docs/external_bam.md#requirements-and-why-they-matter) explains which
settings you can change and which filters Phasis still applies.

## Troubleshooting

- **Command not found:** activate your Phasis environment and check `which phasis`,
  `which hisat2`, and `which samtools`.
- **Installation or dependency errors:** use the environment above. NumPy
  `1.26.4` and scikit-learn `1.3.2` are pinned for classifier reproducibility;
  scikit-learn `1.3.2` supports the Python 3.12 Conda installation. SAMtools
  must be version 1.10 or newer; check `samtools --version`.
- **FASTQ input rejected:** confirm adapter/quality trimming and the expected
  sRNA length range before running Phasis.
- **No plots:** first check whether final *PHAS* or *PHAS*-like calls were
  produced, then check the [plot-staging settings](docs/advanced_usage.md#plot-staging).
- **Plot fonts:** Phasis uses Matplotlib's bundled DejaVu Sans with normal/bold
  weights, including math text, without requiring custom fonts or TeX. Font
  settings are scoped to rendering and restored afterward. If a shared system
  reports an unwritable font cache, set `MPLCONFIGDIR` (and `XDG_CACHE_HOME` for
  fontconfig) to writable directories before launching Phasis.
- **High RAM or temporary disk use:** see [resource tuning](docs/advanced_usage.md#hpc-and-resource-tuning).
  More CPU workers can increase concurrent memory and disk demand.

When reporting a problem, include the command, Phasis version, relevant log
messages, input format, and whether you reused a run directory. Do not share
private sequencing data or credentials in a public issue.

## License

Phasis is distributed under the [Artistic License 2.0](LICENSE.txt).
The software is provided "AS IS", without warranties. Section 14 of the
license disclaims liability for damages arising from its use, subject to
applicable law. See the license for the full terms.

## Authors

- Thales Henrique Cherubino Ribeiro - thalescherubino@gmail.com
- Atul Kakrana - kakrana@gmail.com
- Blake Meyers - bcmeyers@ucdavis.edu
