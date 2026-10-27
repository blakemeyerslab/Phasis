# Individual phasiRNAs and phased units

[Back to the output guide](../README.md#understand-the-outputs)

`{phase}_phasiRNAs.tsv` links individual sRNAs to the interpreted phased
structure of called *PHAS* loci. Here, `{phase}` is the selected phase length,
usually 21 or 24. This is a read-assignment table, not one row per locus or
one row per distinct sequence across the experiment.

## Loci, registers, windows, and units

- A **locus** is the broader candidate region reported by Phasis.
- A **phase register** specifies expected sRNA positions separated by the
  selected phase length.
- A **phased window** is a local interval evaluated for phased support.
- A **phased unit** groups supported windows describing a local phased region,
  potentially including extensions and compatible opposite-strand partners.

These terms follow the distinction between a main phased unit
and additional supported phased windows. A locus need not have a single
register or one continuous phasiRNA-producing region. Phasis can report
supported overlapping alternative registers and other local peaks, including
spatially separated phased regions, as secondary units. These are inferred
structures, not experimental proof of separate precursor transcripts or
independent processing events.

The export includes the main unit and secondary units promoted by the
interpretation procedure. It does not list every possible register, every
scored window, or every sRNA mapped within the locus. Only reads with length
equal to the selected phase are considered for these assignments.

## Column guide

| Column | Meaning |
| --- | --- |
| `identifier` | Reported locus identifier. |
| `cID` | Internal candidate-cluster identifier. |
| `alib` | Library supplying the evidence; pooled analyses use pooled evidence. |
| `phase` | Phase length in nucleotides. |
| `window_unit_id` | `unit_main` or a secondary identifier such as `unit_secondary_1`. IDs are local to the locus/library context, not global identifiers. |
| `window_unit_role` | Role of the unit or member window: see below. |
| `window_unit_rank` | Main unit is 0; secondary units are numbered from 1. This is not a probability or statistical confidence value. |
| `window_unit_shift_nt` | Register shift recorded for the unit/member relative to the main interpretation; may be missing when not defined. It is not a region's length or the gap between regions. |
| `strand` | `w` for forward and `c` for reverse strand. |
| `observed_pos` | Observed read position carried from the native cluster data. |
| `expected_register_pos` | Expected register position associated with this read assignment. |
| `register_class` | Exact, offset, or extension assignment, described below. |
| `abun` | Read abundance carried from the analysis input; not a count of unit assignments. |
| `tag_seq` | sRNA sequence carried from the alignment; see the orientation caution below. |
| `hits` | Mapping-hit count carried from the analysis, where available. It describes reported mapping multiplicity, not necessarily all possible genomic placements. |

Group by `identifier`, `cID`, `alib`, `phase`, and `window_unit_id` to inspect
an individual unit. Do not group all `unit_main` rows across the experiment
as if they belonged to one region.

### Unit roles

The main unit can contain `main_hpsp` (the anchor window), `main_partner`
(compatible opposite-strand support), and `main_extension` members. These
share `unit_main`; they are not automatically separate secondary units.

Secondary units use `overlapping_alternative` or `other_local_peak` roles.
The latter identifies other supported local signal; it should not be assumed
to mean a particular physical gap without examining the positions and plot.

### Register classes

- `core_exact`: a read at an expected position in the core register.
- `core_offset`: when the exact position has no read, a read at a neighboring
  position one nucleotide away. If both neighbors are present, the more
  abundant position is chosen.
- `extended_exact`: a read at an exact register position in a supported
  extension beyond the core window.

The table therefore includes offset support as well as exact-register reads.
Use `register_class` to distinguish these rather than assuming every exported
sequence has identical evidence for phased processing.

## Abundance and repeated assignments

`abun` is copied from the cluster data. In a standard unnormalized analysis it
reflects the input tag abundance; if normalization was enabled, it reflects
the abundance used by that analysis. The export does not independently
normalize abundances or divide them by `hits`.

A read can occur in more than one window/unit assignment, at multiple mapping
positions, or in multiple libraries. The exporter writes those assignments
without a global deduplication step. **Do not sum all rows to estimate total
library abundance or total phasiRNA production.** Choose a deduplication and
multimapping policy appropriate to the downstream question and preserve the
original locus/library/unit associations.

## Preparing sequences for target prediction

The table provides sequences and abundance context for selecting phasiRNAs
for downstream target prediction; it does not itself predict or validate
targets.

1. Select the loci, libraries, units, and register classes relevant to the
   analysis. For an exact-register subset, exclude `core_offset` rows.
2. Restore the original read orientation before exporting sequences. The
   current exporter retains alignment-oriented `tag_seq`: forward-strand
   sequences can be used as written, while reverse-strand (`c`) sequences
   need reverse complementation to recover the original 5'-to-3' sRNA.
3. Remove missing sequences and collapse duplicate original-orientation
   sequences for the target-prediction input, keeping a separate mapping back
   to all source loci, libraries, and phased units.
4. Supply the resulting sequences and appropriate target transcripts to your
   chosen target-prediction tool. Follow that tool's sequence-format
   requirements; candidate targets still require biological validation.

*PHAS*-like sequences are exported separately in the *PHAS*-like output bundle
as `{phase}_PHAS_like_phasiRNAs.tsv`. Keep their evidence category separate
from sequences associated with high-confidence *PHAS* calls.

## Implementation

The schema and assignment logic are in
[`phasis/stages/locus_plots.py`](../phasis/stages/locus_plots.py), including
`PHASIRNA_EXPORT_COLUMNS`, `_collect_export_rows_for_strand`, and
`_write_phasirna_export`. Unit construction is in
[`phasis/stages/feature_assembly.py`](../phasis/stages/feature_assembly.py).
The export shares the interpretation used for the locus diagnostic plots.
