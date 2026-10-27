# Future improvements

This file tracks deferred ideas. Entries are not implemented features or
commitments for a particular release.

## Export phasiRNA sequences in original read orientation

Status: deferred; documentation added, no software change requested.
Recorded: 2026-10-27.

### Current behavior

`{phase}_phasiRNAs.tsv` copies `tag_seq` from the cluster data. In the current
native and external-BAM paths, this is the alignment-oriented sequence.
Reverse-strand (`c`) rows therefore require reverse complementation to recover
the original 5'-to-3' sRNA sequence. This representation is inherited from
SAM/BAM handling, not a biological requirement for the output table.

For users preparing target-prediction inputs, exporting the original sRNA
sequence directly would be clearer and less error-prone.

### Proposed approach

- Consider adding `srna_sequence_5to3` to both the high-confidence *PHAS* and
  *PHAS*-like exports.
- Preserve the existing meaning of `tag_seq` for compatibility rather than
  silently changing it.
- Derive the new field from the current sequence and strand representation,
  after auditing every supported input path to prevent double complementation.
- Update the output guide and downstream examples if the change is approved.
- Do not change mapping, scoring, classification, RRL, abundance, or unit
  assignment behavior as part of this improvement.

### Checks before implementation

- Trace sequence orientation through native HISAT2 and external-BAM imports,
  including Bowtie 1 and BWA-aln inputs.
- Confirm forward-strand sequences remain unchanged and reverse-strand
  sequences recover the original input read, including mismatch-enabled BAMs.
- Use non-palindromic test sequences, with both strands and multiple placements.
- Test missing sequences and supported strand encodings explicitly.
- Verify existing columns, row counts, abundances, and phased-unit assignments
  remain unchanged in both export bundles.
- Confirm duplicate original sequences can be collapsed for target prediction
  while retaining their locus, library, and unit associations separately.

### Relevant files

- [Current export guide](phasirna_output.md)
- [Export implementation](../phasis/stages/locus_plots.py)
- [Native alignment parsing](../phasis/stages/sam_parsing.py)
- [External BAM importer](../support_scripts/bam_to_candidates.py)

Approval is required before implementing this proposal.
