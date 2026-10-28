# Real Maize Candidate-Cluster Example

This is a small **real-data excerpt**, not fabricated alignment records. It
contains two complete blocks from `W23_2.0_1.21-PHAS.candidate.clusters` produced
by native Phasis discovery for the maize README example:

| Native cluster ID | Reference | Alignment-start range (1-based) | Records |
|---|---|---|---|
| `W23_2.0_1-4_1228_4` | `4` | 2993270-2994498 | 30 |
| `W23_2.0_1-10_8376_10` | `10` | 24037717-24038223 | 25 |

Records and headers are unchanged. These ranges describe alignment starts,
not BED end coordinates. The matching library is GEO **GSM4180401**, and the
reference is the maize **B73 RefGen v2** genome. Discovery used numeric IDs:
`Chr1`-`Chr10` map to `1`-`10`, `Mt` to `11`, `Cp` to `12`, and `UNMAPPED` to
`13`. This naming map does not change coordinates or genome assembly.

## Matching inputs

Reuse the inputs from the main [README example](../../../README.md#run-a-real-example),
or download them into a separate data directory:

```bash
mkdir candidate_example_data
cd candidate_example_data
wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4180401&format=file&file=GSM4180401%5FTP%5FW23%5F2%5F0%5F1%5Fchopped%2Etxt%2Egz" -O W23_2.0_1.tag.gz
wget https://download.maizegdb.org/B73_RefGen_v2/B73_RefGen_v2.fa.gz
```

The candidate file is small; the original genome and library downloads are not.
Do not replace these with empty placeholder inputs. Use the
[classification-only command](../../candidate_clusters.md#real-example-and-runnable-command)
with `data` set to this data directory. Do not remap the data to run this example.

## Provenance

The source candidate file was retained from the local four-library example
run, not regenerated from hand-entered values. Selecting complete blocks
preserves their discovery context and statistics; it does not make this subset
a representative benchmark or preserve the complete run's fitted classifier.
All 55 sequences and abundances were verified against the original tag-count
library, reverse-complementing `c` records for that comparison. All 55 aligned
sequences also matched the specified intervals of the original B73 RefGen v2
FASTA in SAM orientation.

SHA256 values of the files used for verification:

```text
626fb6a42db6d5d66a966786fdfe3e04f01f27ff2650529fe91c8dfc31650a42  W23_2.0_1.21-PHAS.candidate.clusters (this example)
9bab5646546e458dbc44c4e59d46137da8c3b747268ce22ad338aff81110bb9a  W23_2.0_1.21-PHAS.candidate.clusters (complete source file)
0b9e2c150eb8ef003275f21caebfc0e8b0c00d80c7fe70fa1b514d6ed7e7c5b7  W23_2.0_1.tag.gz
73961e1f1057b8d6ebaab567ce1c25ba746f853c025c77df58d077aa89999c54  B73_RefGen_v2.fa.gz
```

## Validation

On 2026-10-28, the documented classification-only command was run in separate
fresh working directories with the matching original library and reference:

| Implementation | Source identity | Outcome |
|---|---|---|
| Phasis v2.8.4 | Tag `v2.8.4`, commit `4c1cd8f5252bae21e95c39463df89b34e78d26c3` | Exit 0; two feature/evidence rows; one high-confidence call |
| Frozen manuscript build | `2.8+zenodo.fix1`, commit `060b91c72f3901380a7f802e5ea0d61a021f0775` | Exit 0; two feature/evidence rows; one high-confidence call |

The frozen archive SHA256 is
`b2073b8d90f8a29bf9c680ddbe94254df778bf0e8b8995dc60800588eb98d28d`
([version-specific Zenodo record](https://doi.org/10.5281/zenodo.23718070)).
Both checks used macOS, Python 3.11.4, NumPy 1.26.4 and scikit-learn 1.3.2.
This is an interface check, **not** a recreation of the manuscript's original
dependency environment or a guarantee of identical calls on other inputs.
Both parsers returned identical typed records for all 55 alignments. The 27
existing CLI/class-mode and cluster-aggregation unit tests also passed; example
shell syntax and local documentation links were checked.

Both runs wrote `21_all_clusters.tsv`, `21_calls.tsv`,
`21_classification_evidence.tsv`, `21_PHAS.gff`, `21_phasiRNAs.tsv`, and plots.
The call in both checks was `10:24037717..24038223` for `W23_2.0_1`.
Neither run created a mapping index or ran Phase I. The frozen build also
emits older classifier-prefixed output aliases; its output bundle is not
identical to that of v2.8.4. The sandbox required a temporary Matplotlib cache;
this did not prevent plot generation.

Do not interpret the one-call outcome as a validation of calling isolated loci:
the model is fitted across the supplied candidates. Use the full candidate set
for scientific comparisons whenever possible.
