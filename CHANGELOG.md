# Changelog

## 2.8.4 - 2026-10-27

This release has passed macOS and Linux/Hive validation within the scope
described below.

### Added

- Optional external-BAM import through the auxiliary
  `support_scripts/bam_to_candidates.py` tool. Supported HISAT2, Bowtie 1, and
  BWA-aln/samse evidence can feed native candidate generation and subsequent
  `phasis -steps class` classification without native HISAT2 mapping.
- Exact-match import by default, with an opt-in substitution limit, configurable
  reported-hit cutoff, and all-reported or unique-only retention policy.
  These policies act on reported evidence, not unknown genomic placements.
- Import manifests, abundance and reference validation, and regression and
  end-to-end synthetic checks. Clipped, gapped, spliced, paired-end, and
  otherwise incompatible BAMs remain unsupported. See the external-BAM guide.

### Fixed

- Reference preparation now distinguishes sampled fingerprinting from whole
  cleaned-FASTA compression, with elapsed times and byte-based compression
  progress. Compression level is unchanged.
- Replaced semibold legend headings with bold and scoped all Phasis plotting
  entry points to Matplotlib's bundled DejaVu Sans fonts and math text, avoiding
  reliance on optional system fonts or external TeX. Caller settings are
  restored afterward; font warnings are not suppressed.
- Corrected an indexing error message's issue-tracker link to the maintained
  repository.

### Documentation

- Reorganized the README from installation and an executable example to output
  interpretation and advanced workflows; added Git and ZIP installation,
  HPC guidance, and generic Bowtie 1/BWA tutorials.
- Added explanations of Howell scoring, phased units, and the phasiRNA export,
  including sequence-orientation and repeated-assignment cautions.
- Clarified the relationship between the original PHASIS and the completely
  rewritten Phasis 2.

### Packaging

- Declared Artistic-2.0 license metadata and included the unchanged license
  text in built packages.
- Included supporting scripts, tests, and public guides in the source
  distribution. Auxiliary scripts remain source-checkout tools, not installed
  Phasis commands.
- Prepared and locally tested a Bioconda recipe on macOS and Linux. Public
  Bioconda installation remains pending final source/checksum selection,
  submission, CI, and maintainer approval.

### Validation

- Source, wheel, and local Conda installation checks passed on Linux/Hive.
- All 277 regression tests passed, along with real HISAT2/Bowtie 1/BWA-aln
  integration and native/importer parity checks on synthetic data.
- The complete four-library 21-*PHAS* README example passed on Hive:
  349 distinct locus identifiers and 1,084 library-specific call rows, with
  standard outputs and no warnings reported.

These results apply to the checksummed candidate recorded in the
[validation record](packaging/HIVE_VALIDATION_21.md). Documentation revisions
after testing do not change the tested software, but require separately
identified release artifacts. No Phasis source fixes were needed for the
recovered Hive environment and validation-script issues.
