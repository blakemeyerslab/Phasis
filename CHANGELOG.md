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

- Replaced semibold legend headings with bold and scoped all Phasis plotting
  entry points to Matplotlib's bundled DejaVu Sans fonts and math text, avoiding
  reliance on optional system fonts or external TeX. Caller settings are
  restored afterward; font warnings are not suppressed.
- Corrected an indexing error message's issue-tracker link to the maintained
  repository.

### Documentation

- Reorganized the README from installation and an executable example to output
  interpretation and advanced workflows; added generic Bowtie 1/BWA tutorials.
- Clarified the relationship and independent version histories of the original
  PHASIS and the completely rewritten Phasis 2.

Native detection, scoring, clustering, classification, and RRL algorithms are
unchanged by these additions. The original manuscript-associated Zenodo archive
is unchanged. Development commits and tests do not constitute a stable release.
