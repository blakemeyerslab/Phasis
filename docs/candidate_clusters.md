# Native Candidate-Cluster Files

Phasis can resume from scored candidate clusters using `-steps class` and
`-class_cluster_files`. This skips mapping and candidate discovery, then runs
window selection/scoring, feature calculation, classification, register-resolved
interpretation, and output generation. It is **not** direct BED/GFF3 input.

## File layout

Use plain-text `<library>.<phase>-PHAS.candidate.clusters` files. `<phase>` is
the discovery phase length, usually 21 or 24 nt. The filename supplies the
library identifier; for example, `W23_2.0_1.21-PHAS.candidate.clusters` belongs
to `W23_2.0_1`. Do not rename it to `example.clusters` or add a library column.
Plain, uncompressed files work in both v2.8.4 and the frozen manuscript build.

Each block begins with `>cluster = <clusterID>`, followed by one **18-field,
tab-separated record per retained sRNA alignment**. There is no column-heading
row. The header supplies the cluster ID, not an interval or a sequence.
Keep IDs unique within the library and each block on one reference sequence.
The native writer uses `<library>-<chromosome>_<cluster-number>_<chromosome>`;
the reader extracts the non-whitespace token after `cluster =`.

## The 18 fields

These names describe field order; they are not written as a table header.
Integers must be integer strings (for example, `21`, not `21.0`). Floating-point
values may use decimal or scientific notation. Requirements below describe
valid native evidence, not comprehensive validation performed by the reader.

| # | Name | Type | Meaning and required values |
|---|---|---|---|
| 1 | `chromosome` | Text | Reference sequence ID in the discovery run's coordinate namespace. Numeric IDs are stored as text. Use the original chromosome-ID map when applicable. |
| 2 | `strand` | Text | Exactly `w` (forward/plus) or `c` (reverse/minus); not `+`/`-`. |
| 3 | `pos` | Integer | **1-based leftmost aligned reference position on either strand**, as in SAM POS; must be positive and in bounds. |
| 4 | `len` | Integer | Aligned sRNA sequence length in nt, positive and equal to the length of `tag_seq`. Not restricted to the selected phase length. |
| 5 | `hits` | Integer | Positive mapping multiplicity recorded for this tag by the discovery workflow. Reported placements need not enumerate all possible genomic placements. |
| 6 | `abun` | Integer | Positive tag abundance assigned to this alignment, not divided by `hits`. Normally the library tag count; native `-norm` runs instead store the rounded normalized abundance, floored at 1. Preserve the discovery run's policy. |
| 7 | `pval_h_f` | Float | Forward-direction hypergeometric enrichment P value. |
| 8 | `N_f` | Integer | Number of occupied position/strand entries considered in the forward scoring window; nonnegative. Not read abundance. |
| 9 | `X_f` | Integer | Those forward-window entries labeled phased; `0 <= X_f <= N_f`. |
| 10 | `pval_r_f` | Float | Forward-direction one-sided Mann-Whitney abundance-test P value. Here `r` names the rank-based test, not the reverse direction. |
| 11 | `pval_corr_f` | Float | Stouffer-combined forward hypergeometric and abundance P value. Despite its name, not a multiple-testing-adjusted P value. |
| 12 | `pval_h_r` | Float | Reverse-direction hypergeometric enrichment P value. |
| 13 | `N_r` | Integer | Number of occupied position/strand entries considered in the reverse scoring window; nonnegative. |
| 14 | `X_r` | Integer | Those reverse-window entries labeled phased; `0 <= X_r <= N_r`. |
| 15 | `pval_r_r` | Float | Reverse-direction one-sided Mann-Whitney abundance-test P value. |
| 16 | `pval_corr_r` | Float | Stouffer-combined reverse hypergeometric and abundance P value. |
| 17 | `tag_id` | Text | Nonempty read/tag identifier. Native IDs have the form `seq_<number>\|<count>`; retain the original ID, including repeated IDs for multiple placements. |
| 18 | `tag_seq` | Text | DNA sequence as stored in SAM alignment orientation. For `c`, this is reverse-complemented relative to the original 5'-to-3' library sRNA. |

For these ungapped alignments, the covered interval is
`[pos, pos + len - 1]`, inclusive. In particular, `pos` is **not** the reverse
strand's 5' coordinate. Coordinates are relative to the reference used for
mapping, which may be a genome or transcriptome. Classification-only mode does
not remap chromosome names or lift coordinates between assemblies.

The `_f` and `_r` suffixes mean scoring toward increasing and decreasing
reference coordinates, respectively; they do not identify the read's strand.
Native candidate scoring uses a 15-cycle template, both strands with the
strand offset, and +/-1-nt offsets. These are candidate-stage statistics, not
the later Howell or final Phasis scores. See [scoring](scoring.md) for the
distinction. Native finite P values lie in `[0, 1]`; rounded zero is valid.
The writer can also emit `nan` for an undefined combined statistic. Preserve
such native values rather than inventing replacements. The parser accepts
floating `nan`, but this is not permission to substitute missing evidence.

The reader is permissive: it skips lines before a recognized cluster header
and records with fewer than 18 fields, ignores extra fields, and raises on
invalid numeric conversions. These behaviors are not a format validator.
Do not insert comments, blank numeric fields, arbitrary P values, or padding
columns to make an interval table look like candidate evidence.

## Real example and runnable command

The [example file](examples/candidate_clusters/W23_2.0_1.21-PHAS.candidate.clusters)
contains **two complete native blocks, 55 alignment records**, copied without
altering IDs, coordinates, sequences, abundances, or statistics from the real
maize W23 library used in the README example. The first two records are:

```text
>cluster = W23_2.0_1-4_1228_4
4	w	2993270	21	1	3	0.001749133	5	4	0.346316392	0.009529134705698	0.1428571429	1	1	0.5	0.225158586306337	seq_226481|3	AGAGAGTGAAGTAGACTACCC
4	c	2993520	21	1	1	0.3645021824	15	3	0.6022229973	0.475376096237774	0.0202134908	2	2	0.3333333333	0.039742150703551	seq_3350519|1	CTAATGAAACGTCACAACACT
```

Use the complete linked file, not this two-record excerpt. With Phasis installed,
start in this repository's root. Set `data` to the directory containing the
matching README inputs; the [example notes](examples/candidate_clusters/README.md)
provide download commands if needed.

```bash
repo="$PWD"
data="/absolute/path/to/phasis_example"
mkdir candidate_class_example
cd candidate_class_example

phasis -mindepth 1 -phase 21 -libformat T \
  -libs "$data/W23_2.0_1.tag.gz" \
  -reference "$data/B73_RefGen_v2.fa.gz" \
  -cores 2 -steps class \
  -class_cluster_files "$repo/docs/examples/candidate_clusters/W23_2.0_1.21-PHAS.candidate.clusters" \
  --outdir results
```

Use a fresh working directory, with only one Phasis run active in it. Both
`-class_cluster_file` and `-class_cluster_files` accept a list of file paths.
Explicit paths avoid automatic filename inference. Retain the matching sRNA
libraries and reference: the CLI checks that they exist, but this mode does not
rerun library preparation/mapping or comprehensively verify their consistency
with imported records. HISAT2 and SAMtools must still be installed for the
startup dependency check, even though mapping is skipped.

The example produced two rows in `results/21_all_clusters.tsv` and
`21_classification_evidence.tsv`, one call in `21_calls.tsv`, a `21_PHAS.gff`,
and `21_phasiRNAs.tsv` in local checks of both implementations. See the
[validation notes](examples/candidate_clusters/README.md#validation).

## Scope and limitations

- Classification consumes the supplied candidate-stage statistics; it does not
  recreate them from BED/GFF3 intervals or the `-libs` input. Mapping options
  such as mismatch or hit limits do not retroactively filter these records.
- Use complete blocks and the discovery phase/settings. Although explicit file
  paths are accepted across phase lengths, acceptance alone does not establish
  that phase-specific P values are appropriate for the new analysis.
- The classifier fits a two-component model to the supplied feature rows.
  At least two usable rows must survive feature extraction. A single interval
  is not a supported general-purpose yes/no assay, and changing the candidate
  set can change fitted classifications. This small example tests the interface,
  not performance or equivalence to a whole-library analysis.
- Independent tools can produce this native representation only if they supply
  compatible mapped-read evidence **and** native scoring statistics. The
  [external-BAM supporting tool](external_bam.md) reuses native clustering and
  scoring; it does not accept user-imposed BED/GFF3 cluster boundaries.
- A restricted custom reference changes the mapping search space, may make
  multimappers appear unique, introduces interval-edge effects, and requires
  mapping local coordinates back to the genome. Prefer genome-wide mapping
  when interpreting targeted results. No direct interval-input interface is
  introduced here.

## Implementation references

The native writer is `clustwrite()` in
[`cluster_scoring.py`](../phasis/stages/cluster_scoring.py); its statistics are
computed by `clustscore()` / `compute_p_vals()` in that file. Alignment positions,
sequence orientation, hit counts, and abundances originate in
[`sam_parsing.py`](../phasis/stages/sam_parsing.py). The field parser is in
[`cluster_aggregation.py`](../phasis/stages/cluster_aggregation.py), and the
classification-only dispatcher is in
[`phase2_pipeline.py`](../phasis/stages/phase2_pipeline.py).

This field layout and command were checked in tagged v2.8.4 and the frozen
[`2.8+zenodo.fix1` manuscript build](https://doi.org/10.5281/zenodo.23718070).
This confirms interface compatibility, not identical behavior for all datasets
or availability of the newer external-BAM supporting tool in the frozen archive.
