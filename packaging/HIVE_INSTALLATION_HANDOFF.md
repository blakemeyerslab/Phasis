# Hive installation and usage validation: Phasis 2.8.4

Status update: this original test plan has been executed. See
[accepted Hive results](HIVE_VALIDATION_21.md) for completed checks. The full
21-PHAS example passed; full 24-PHAS completion was unverified at handoff and
is not required to proceed under the user's accepted scope. Instructions
below are retained as the original plan, not a request to repeat passed work.

## Purpose and boundaries

Independently test the staged v2.8.4 package and installation instructions on
Linux/Hive. Do not commit, push, merge, tag, publish, upload Conda packages, or
submit the Bioconda recipe. Do not change scientific algorithms or the frozen
manuscript archive. Use a scheduler allocation for analysis, not a login node.

The latest documentation and packaging changes are local and uncommitted.
Cloning `main` or the development branch does NOT retrieve this exact candidate.
Use the checksummed source snapshot below, and record its hash with all results.

## Files to retrieve from the local Mac

Transfer the contents of:

```text
/Users/triberio/local_bioinfo/local_phasis/Phasis/dist/installation_validation_20261027/
```

The directory contains:

- `Phasis-2.8.4-validation.zip`: exact local source snapshot, not a public release.
- `phasis-2.8.4.tar.gz`: Python source distribution, including supporting scripts.
- `phasis-2.8.4-py3-none-any.whl`: built Python package, without supporting scripts.
- `SHA256SUMS.txt`: checksums for the three archives.
- `BUILD_PROVENANCE.json`: base commit, local modifications, and source-file hashes.
- `LOCAL_VALIDATION.md`: completed local checks and remaining platform limitations.
- `HIVE_INSTALLATION_HANDOFF.md`: these instructions.

Copy from the Mac using rsync or scp to a NEW Hive staging directory. For example,
run this on the Mac, not on Hive (do not use `--delete`):

```bash
rsync -avP -e 'ssh -i ~/.ssh/id_ed25519' \
  /Users/triberio/local_bioinfo/local_phasis/Phasis/dist/installation_validation_20261027/ \
  thales@hive.hpc.ucdavis.edu:~/phasis-v284-install-validation-20261027/
```

No transfer or remote job has been performed by local Codex for this task.

## 1. Verify and unpack

Use a new working directory and preserve existing runs. On Hive:

```bash
cd ~/phasis-v284-install-validation-20261027
sha256sum -c SHA256SUMS.txt
unzip Phasis-2.8.4-validation.zip
export PHASIS_VALIDATION_SOURCE="$PWD/Phasis-2.8.4-validation"
```

Do not infer release status from the version string. This is an unpublished
release candidate with local changes, identified by the archive checksum.

## 2. Clean installation from source

Create a new environment; do not reuse the currently installed Phasis environment.

```bash
conda create -n phasis-v284-install-validation --override-channels \
  -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 numpy=1.26.4 scikit-learn=1.3.2 hisat2 'samtools>=1.10' wget -y
conda activate phasis-v284-install-validation
cd "$PHASIS_VALIDATION_SOURCE"
python -m pip install -U pip
python -m pip install -e .
phasis -version
phasis -h
phasis-compare --help
python -m pip check
python -c 'import phasis; print(phasis.__file__)'
python -c 'from phasis.deps_check import require_dependencies; require_dependencies()'
hisat2 --version
samtools --version
python -m pip freeze > ../hive-pip-freeze.txt
conda list --explicit > ../hive-conda-explicit.txt
```

Confirm the version is 2.8.4 and the import path is the staged source.
The README's Git/ZIP examples intentionally target `main` for the eventual
release. Do not replace the staged candidate with today's older `main`.
For the Git workflow, verify command syntax now and test the actual public
v2.8.4 checkout only after the approved source is published.

## 3. Build and wheel installation

```bash
python -m pip install build
python -m build --outdir ../hive-dist
```

Use another clean environment with HISAT2 and SAMtools to install the wheel
non-editably. Run the verification commands from outside the source directory
with `PYTHONPATH` unset. Confirm `phasis.__file__` is under that environment's
`site-packages`, not an old checkout. Inspect wheel metadata for
`License-Expression: Artistic-2.0`, `License-File: LICENSE.txt`, and the actual
unchanged license file. The source archive should contain `support_scripts/`;
the installed wheel should not create a BAM-import CLI.

## 4. Usage tests

First test the two library converters on tiny adapter-trimmed, valid sRNA
inputs with repeated sequences and known counts. Run them from the staged
source and check the `.tag` sequences and counts. FASTA input to its helper
must be uncompressed and uncollapsed. The FASTQ helper also accepts gzip.

Install optional importer/test dependencies only for the auxiliary workflow:

```bash
conda install --override-channels -c conda-forge -c bioconda bowtie bwa -y
cd "$PHASIS_VALIDATION_SOURCE"
python -m pip install -e '.[bam]'
python -m unittest discover -s tests
```

Inside a compute allocation, use a fresh output path for the existing small
real-aligner integration test:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/phasis-v284-mpl-$USER-$$"
mkdir -p "$MPLCONFIGDIR"
python tests/run_external_bam_integration.py \
  --output-dir "$HOME/phasis-v284-install-validation-20261027/hive-bam-smoke"
```

This exercises HISAT2, Bowtie 1, BWA-aln, the supporting importer, native
candidate functions, and classification. Inspect `INTEGRATION_SUCCESS.json`
and logs, not just command completion. If optional aligner installations fail,
report that separately from the native Phasis installation result.

Then run the README's four-library example under normal parameters in a NEW
analysis directory, reusing downloaded raw inputs by copying or linking them
if needed, but not old indexes, intermediates, or output directories. Run
21-PHAS and 24-PHAS sequentially; use the allocated core count, not more.
Check standard calls, evidence, GFF, plots, and phasiRNA exports. Record counts,
exit status, resource use, and warnings, especially plotting-font warnings.
If the full run cannot finish within the allocation, report exactly which
stages passed; a syntax/startup check is not an end-to-end PASS.

## 5. Local-only Conda recipe test

The source contains `packaging/bioconda/meta.yaml`. Its public tag URL and
checksum are intentionally pending release. In a disposable copy, replace the
ENTIRE `source` block with a local path to the extracted staged source, or a
`file://` source archive URL and its verified hash. Keep the checked-in draft
unchanged. Use a separate `conda-build` environment:

```bash
conda build /path/to/disposable/recipe --no-anaconda-upload \
  --override-channels -c conda-forge -c bioconda
```

Install the resulting local artifact in another clean environment and repeat
CLI/dependency checks. Do not submit the draft or upload artifacts. Record
dependency solver failures verbatim. Bioconda CI and maintainer approval are
still required for publication even if this local test passes.

## Return a concise report

Report source hashes, Python/platform/tool versions, environment logs, source
and wheel installation results, converter results, test counts, integration
results, README example completion/counts, font warnings, and Conda build/test
status. Clearly distinguish PASS, FAIL, and NOT RUN. Include exact reproducing
commands for failures. No claims of Linux validation until these tests run.
