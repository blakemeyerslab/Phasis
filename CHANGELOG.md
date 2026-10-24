# Changelog

## 2.8.4 - Unreleased

Development validation build on `dev/v2.8.4`. A GitHub Release and version tag
will be created only after maintainer testing and approval.

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

- Explicit GMM initialization (`random_from_data`, seed 0, one initialization)
  replaces the environment-dependent k-means/fallback choice. This preserves
  the historical fallback policy validated on the four-library maize example;
  calls can differ from prior normal k-means runs. Broader validation remains
  required before release.
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
  interpretation and advanced workflows; added generic Bowtie 1/BWA tutorials.
- Clarified the relationship between the original PHASIS and the completely
  rewritten Phasis 2, including the intentional capitalization distinction.

Detection, scoring, clustering, feature thresholds, and RRL rules are unchanged.
GMM initialization is now explicit as described above; this can affect calls.
The original manuscript-associated Zenodo archive
is unchanged. Development commits and tests do not constitute a stable release.
