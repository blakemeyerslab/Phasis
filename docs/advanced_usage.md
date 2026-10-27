# Advanced Phasis Workflows

Start with the [README example](../README.md#run-a-real-example) for a standard
analysis. This guide covers optional workflows and resource settings; most users
do not need to change them. Examples using `support_scripts/` assume you are in
the source checkout, or have replaced that prefix with its absolute path.

## Classify Existing Candidate Clusters

When native-format `*.candidate.clusters` files are available from a previous
or alternative run, use `-steps class` to skip candidate discovery and execute
classification/output generation:

```bash
phasis -mindepth 1 -phase 24 -libformat T \
  -libs *.tag -reference genome.fa -cores 4 \
  -steps class \
  -class_cluster_files previous_run/*.candidate.clusters
```

Supply matching library names, phase, reference assembly, and chromosome IDs.
If `-class_cluster_files` is omitted, Phasis tries to infer expected candidate
paths from `-libs`, `-phase`, and the run directory. Arbitrary interval tables
are not candidate-cluster files.

To generate native candidates from external alignments, follow the
[Bowtie 1 / BWA tutorial](external_bam.md). Use separate calling workspaces for
different aligners so that their intermediates and caches cannot be mixed.

## Combine Completed Runs

`phasis-compare` combines high-confidence *PHAS* calls from two completed result
directories, for example individual-library and pooled-library analyses:

```bash
phasis-compare 21_results_pooled 21_results_nonpooled \
  --outdir 21_results_combined
```

Only loci classified as *PHAS* are included. The default `--overlap-buffer 0`
matches directly overlapping intervals on the same chromosome. The default
`--prefer auto` keeps the non-pooled record for a shared locus when one input is
pooled, while retaining distinct finer non-pooled loci overlapping one broader
pooled call. Override with `--prefer a` or `--prefer b`; use `--label-a` and
`--label-b` to name inputs in reports. Inputs must use compatible coordinates.

The output contains Phasis-style non-plot tables (`{phase}_calls.tsv`,
`{phase}_all_clusters.tsv`, `{phase}_classification_evidence.tsv`,
`{phase}_phasiRNAs.tsv`, and `{phase}_PHAS.gff`) for the deduplicated union.
Companion tables retain rows from each selected representative's source run,
not unrelated rejected candidates. Audit tables report shared and run-specific
loci, source-locus mappings, and overlap pairs. `comparison_summary.txt` and
`comparison_manifest.json` record counts, inputs, and settings.

A nonempty output directory is refused unless `--force` is given; that option
overwrites known comparison outputs. The equivalent source-checkout wrapper is:

```bash
python support_scripts/compare_phasis_results.py DIR_A DIR_B --outdir combined_results
```

## Evaluate Calls Against a Reference

PhasMatch compares a Phasis calls table to reference loci in BED, GFF, or GTF:

```bash
python support_scripts/phasMatch.py 24_results/24_calls.tsv reference_loci.bed
```

The default is one-based Phasis calls and a **250-nt flank** around each predicted
interval for matching. BED references are zero-based, half-open; GFF/GTF
references are one-based. Check assembly, chromosome names, and coordinate
conventions before interpreting matches. The second argument is a reference
annotation, not another arbitrary calls TSV; use `phasis-compare` above for
comparing completed result directories.

PhasMatch reports both prediction-ID and reference-centric metrics. They can
differ when one call matches multiple reference loci. Preserve the reported
denominators and the corresponding precision/recall/F1 definitions when
reporting results; do not mix them into a new metric silently.

## Input Preparation

### Convert library formats

Direct preprocessed FASTQ input (`-libformat Q`) is usually sufficient. To
convert it explicitly to tag-count format:

```bash
python support_scripts/fastqToTag.py sample.fastq.gz --chunk-unique-tags 250000
phasis -libs sample.tag -reference genome.fa.gz -libformat T
```

The helper accepts `.fastq`, `.fq`, and gzip variants and uses the same
disk-backed aggregation as direct FASTQ input. Lower `--chunk-unique-tags`
reduces memory use at the cost of more temporary disk I/O. For direct input,
the corresponding option is `--fastq-chunk-unique-tags`.

To convert FASTA to tag-count:

```bash
python support_scripts/fastaToTag.py sample.fasta
```

### Reference IDs

Phasis can preserve original FASTA IDs or use compact numeric IDs in its
indexed `.clean.fa` reference:

```bash
phasis -libs sample.tag -libformat T -reference genome.fa --reference_id_mode preserve
phasis -libs sample.tag -libformat T -reference genome.fa --reference_id_mode numeric
```

Choose one mode for a workflow and preserve its equivalence table for downstream
reference comparisons. Renaming chromosomes does not convert genome assemblies.
For a separate compact-header FASTA and mapping table:

```bash
python support_scripts/replace_genome_headers.py genome.fa new_genome.fa equivalence.tsv
```

## Filtering Options

Use the normal settings first. Changes can affect calls and should be recorded
when comparing runs. Phasis uses its GMM classifier; no classifier-selection
flag is needed.

| Option | Purpose |
| --- | --- |
| `-mindepth` | Minimum depth for p-value computation; default 1. |
| `-maxhits` | HISAT2 reporting cap and native hit filter. Default 25 retains fewer than 25 reported placements. |
| `-mismat` | Post-alignment mismatch filter. |
| `-uniqueRatioCut` | Minimum proportion of reads satisfying the native unique-hit criterion. |
| `-max_complexity` | Complexity filter. |
| `-phasisScoreCutoff` | Phasis-score cutoff; phase-dependent limits also apply. |
| `-min_Howell_score` | Howell-score threshold during classification/output filtering. |

Consult `phasis -h` for the available options and defaults.

## HPC and Resource Tuning

### HPC installation and batch jobs

Install into an environment you own, using a persistent project filesystem
accessible from compute nodes. Put Conda/Pip caches on a filesystem with
enough quota, following your cluster's policy. On Hive, validation used
Quobyte paths for environments and caches. Node-local temporary storage is
appropriate for disposable scratch files, not an environment needed by later jobs.

Use either the cluster's Conda module or your own Conda installation in a job,
not both. Check `command -v conda`, `command -v python`, `command -v phasis`,
and `python -c "import phasis; print(phasis.__file__)"` after activation.
Avoid inherited `PYTHONPATH` or user-site packages selecting another checkout.

For a module-based setup, initialize modules as instructed by your HPC site
and load its Conda module before activating your environment. Hive validation
used `/etc/profile.d/modules.sh` and `module load conda`. An inherited personal
Miniconda setup conflicted with that module; starting the batch shell without
those inherited Conda variables/functions resolved it. Do not uninstall or
modify your personal environment to repair a job shell.

For example, after selecting one Conda installation, replace this project
path with your writable location before creating the environment:

```bash
export PHASIS_PROJECT=/path/to/writable/project
export CONDA_PKGS_DIRS="$PHASIS_PROJECT/cache/conda"
export PIP_CACHE_DIR="$PHASIS_PROJECT/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$PHASIS_PROJECT/envs"
conda create -p "$PHASIS_PROJECT/envs/phasis" --override-channels \
  -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 numpy=1.26.4 scikit-learn=1.3.2 hisat2 'samtools>=1.10' wget -y
conda activate "$PHASIS_PROJECT/envs/phasis"
```

Then follow the README's source download and pip installation steps. Activate
the same environment explicitly in each batch job. Run analyses inside a
scheduler allocation, using no more CPUs than allocated, with only one Phasis
run active per working directory. For Slurm jobs, the following settings keep
imports and plotting caches separate from unrelated interactive sessions:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/phasis-mpl-${SLURM_JOB_ID:?}"
mkdir -p "$MPLCONFIGDIR"
```

If Conda reports a cache verification error, preserve the error log and retry
with a fresh, private `CONDA_PKGS_DIRS` on suitable storage. Do not delete a
shared cache. Such installation/cache failures are distinct from Phasis
analysis failures. Record the environment specification and command alongside
the analysis outputs.

### Resource allocation

Start by setting `-cores` to your job's CPU allocation. More workers can mean
more concurrent memory and disk use. Keep sufficient temporary storage and
monitor your scheduler's memory report before increasing parallelism.

The environment variables below use the supported `Phasis_` prefix. They tune
resource use rather than biological thresholds. Change one setting at a time
and record it with the run command. Example values below are not universal
recommendations.

### Library preparation

```bash
export Phasis_LIB_WORKER_CAP=1
```

FASTQ conversion begins with one library worker and grows through
1, 2, 4, 6, and 8 after successful batches, up to the available core count and
the default cap of eight. The cap can be lowered or raised according to memory
and temporary-I/O capacity. Failures reduce parallelism and retry work.
This setting controls preparation, not the requested mapping core count.

### Candidate aggregation

Phase II streams candidate records into sorted disk-backed runs and merges them
instead of keeping all libraries and sorted records in memory. Defaults are
100,000 records per active worker chunk and a 64-file merge fan-in.

```bash
export Phasis_CLUSTER_AGGREGATION_CHUNK_ROWS=50000
export Phasis_CLUSTER_AGGREGATION_MERGE_FAN_IN=32
```

Smaller chunks trade RAM for temporary I/O. Lower fan-in limits simultaneously
open files but can add merge passes. Spill directories are removed on normal
completion; cleanup also removes remnants of interrupted runs.

### *PHAS*-cluster construction

After assigning universal IDs, Phase II builds `{phase}_PHAS_to_detect.tab`
in batches within chromosome/library groups. Defaults are 100,000 input rows,
starting with two workers and growing to 70% of allocated cores. Completed
batches are written in stable input order rather than retained as one large
in-memory table.

```bash
export Phasis_PHAS_CLUSTER_BATCH_ROWS=50000
export Phasis_PHAS_CLUSTER_WORKER_CAP=3
```

This splits large groups into consecutive batches without mixing or reordering
groups. Downstream window selection and feature assembly use disk-backed
readers; locus plotting scans rows needed for final *PHAS* or *PHAS*-like calls.

### Feature assembly

Feature assembly packs complete clusters into 100,000-row batches across
chromosomes. It starts with two concurrent batches and can grow to 70% of
allocated cores. Failures reduce concurrency; recovery stays within that cap.

```bash
export Phasis_FEATURE_ASSEMBLY_WORKER_CAP=3
export Phasis_FEATURE_ASSEMBLY_BATCH_ROWS=50000
```

Lower batch sizes reduce per-task memory. A single cluster larger than the
setting remains intact to preserve its features; Phasis reports these cases.
Completed features are streamed to a temporary table.

### Plot staging

Large runs can stage individual locus plots through local scratch instead of
writing each plot directly to a shared filesystem. Add one of these options to
your normal command:

```text
--plot_staging auto
--plot_staging local
--plot_staging direct
```

The default `auto` stages plots when an HPC-style environment or
remote/distributed output path is detected. Confirm that staged plots reach
`--outdir` before removing job-local scratch.

### Logs and executable checks

Progress bars go to standard output; warnings and errors go to standard error.
Capture both in scheduler logs. At startup, Phasis validates SAMtools 1.10 or
newer, prints its resolved path/version, and reuses that executable for mapping
and BAM parsing. If resolution is unexpected, check the active environment with
`which samtools` and `samtools --version`.

## Cleanup

Run cleanup **only from the intended Phasis run root**, after confirming that
you no longer need its intermediate files:

```bash
phasis -cleanup
```

This removes intermediates while preserving `index/`, final results, and
index-related cache metadata. For removal of the index and `phasis.mem` too:

```bash
phasis -cleanup_all
```

Both preserve final results directories, but deleted intermediates must be
regenerated for later analyses. Completed text intermediates are compressed by
default; `--no_compress_intermediates` keeps them uncompressed when needed for
inspection or external tools.
