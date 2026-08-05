# Cache invalidation study

Date: 2026-08-05

## Scope and correctness rule

This study maps runtime inputs and parameters to the earliest PHASIS stage whose
result can change. A cached artifact is reusable only when all of the following
match:

1. the artifact fingerprint;
2. the fingerprints of its direct inputs;
3. every semantic parameter used by the stage; and
4. the stage's cache-schema or algorithm version.

Output-only hashes are useful integrity checks, but they are not sufficient
provenance. In particular, a legacy output hash must never override a modern
input-signature mismatch.

## Confirmed `maxhits` defect and repair

The mapping signature already included `maxhits`, but a signature miss was
followed by a legacy `[MAPS]` lookup. Because `[MAPS]` stores only the BAM's own
hash, the stale BAM was accepted and then recorded under the new signature.
SAM parsing and cluster building had the same fallback pattern. Cluster scoring
could also accept verified output hashes after its full signature had changed,
even when a parser nest dictionary changed.

The repair makes centralized signatures authoritative for mapping, SAM parsing,
cluster building, and cluster scoring. Legacy output hashes remain write-only
compatibility records. Cache-schema versions were added to these four stages so
artifacts whose signatures may already have been incorrectly updated are rebuilt
once. After that one-time rebuild, unchanged inputs and parameters remain normal
cache hits.

## Phase I dependency map

| Changed input or parameter | Earliest affected stage | Required invalidation |
|---|---|---|
| Reference contents | Index construction | Index, mapping, parsing, clustering, scoring, and Phase II |
| `reference_id_mode` | Index construction | Same as reference contents |
| Reference-cleaning algorithm/schema | Index construction | Clean FASTA, ID map, index, and all downstream artifacts |
| Index shard contents | Index validation | Index rebuild or a proven-identical complete shard manifest |
| HISAT2-build options or semantic version | Index construction | Same as reference contents |
| Library contents | Library preprocessing | Affected library branch and its downstream artifacts |
| `libformat` or preprocessing mode | Library preprocessing | Affected processed FASTA and downstream artifacts |
| `mindepth` | Library preprocessing | Processed FASTA, mapping, and all downstream artifacts |
| Library-preprocessing algorithm/schema | Library preprocessing | Affected processed FASTA and downstream artifacts |
| `concat_libs` or pooled library membership | Pooled FASTA merge | Pooled mapping branch and all pooled downstream artifacts |
| `maxhits` | Mapping | BAM and downstream parsing; later stages follow changed direct inputs |
| HISAT2 alignment or Samtools sort/view options/versions | Mapping | BAM and all downstream artifacts |
| `mismat` | SAM parsing | Parsed dictionaries/counts and downstream artifacts |
| `norm` or `norm_factor` | SAM parsing | Parsed dictionaries/counts and downstream artifacts |
| `phase` | SAM parsing under the current file design | Phase-specific parsed files and all downstream artifacts |
| Parser algorithm/schema | SAM parsing | Parsed dictionaries/counts and downstream artifacts |
| `clustbuffer` | Cluster building | `.lclust`, cluster scoring, and Phase II |
| Cluster-building algorithm/schema | Cluster building | `.lclust`, cluster scoring, and Phase II |
| `uniqueRatioCut` | Cluster scoring | Candidate-cluster outputs and Phase II |
| Scoring constants or algorithm | Cluster scoring | Candidate-cluster outputs and Phase II |
| Core count, worker caps, batch sizes, or `fastq_chunk_unique_tags` | None when output is deterministic | No semantic invalidation |

Per-library cache keys should remain independent. A change in one non-pooled
library should not force unrelated libraries to remap. The pooled and
non-pooled analyses should remain separate branches rather than deleting one
another. Within the pooled branch, any member-library or input-order change
invalidates the merged `ALL_LIBS` branch.

## Phase II dependency map

| Changed input or parameter | Earliest affected stage |
|---|---|
| Candidate-cluster files or their set | Candidate aggregation |
| `minClusterLength` | Candidate loci table |
| `clustbuffer` | Universal-locus merging |
| `window_len` or sliding interval | Scoring-window selection |
| Window-scoring algorithm | Window scoring |
| Feature columns or feature algorithm | Feature assembly |
| PHAS score, Howell score, complexity, classifier settings | Classification |
| Evidence rules or classification overrides | Evidence assignment |
| Plot mode or plotting code | Plot generation only |
| Compression or staging location | Representation/delivery only |

The Phase II file-signature chain generally propagates changed candidate files
correctly. If a Phase I parameter is changed but a regenerated candidate file is
byte-identical, reusing Phase II is safe because its direct semantic input did
not change. A future explicit stage-dependency manifest can support stricter
"rerun everything downstream" policy when required for auditability.

## Remaining risks

1. Index reuse verifies index-file presence but does not compare the stored
   marker-shard fingerprint. A future index manifest should fingerprint every
   shard; legacy records without `reference_id_mode` also need a conservative
   migration rule.
2. Library preprocessing's legacy adoption path proves input and depth identity,
   but does not fully establish `libformat` and processing-mode provenance.
3. Indexing, library preprocessing/merge, and several later stages still lack
   explicit cache-schema or algorithm versions.
4. Large-file fingerprints are sampled from the beginning, middle, and end.
   Same-size changes elsewhere can theoretically be missed. Complete hashes
   recorded when an artifact is produced would provide stronger provenance.
5. External executable versions, especially HISAT2 and Samtools, are not yet
   semantic mapping-signature inputs.
6. Classification and evidence assignment need clearly separated checkpoints;
   override files should be fingerprinted by contents, not only by pathname.
7. Sidecar files such as reference-cleaning outputs, processed-library/parser
   summaries, and mapping summaries are not consistently covered by cache
   validity rules.

## Recommended staged redesign

1. Extend the authoritative-signature rule to every stage and remove remaining
   hash-only read-side migration paths.
2. Give every stage a cache-schema/algorithm version and a normalized manifest
   containing semantic parameters, ordered input fingerprints, tool versions,
   and output fingerprints.
3. Represent dependencies as an explicit stage DAG. Mark only affected
   per-library branches dirty, then propagate dirtiness downstream; treat the
   pooled branch as depending on every member library.
4. Compute complete content hashes while outputs are written. Sampled hashes can
   remain an optional fast precheck, but should not be the final provenance key.
5. Add regression tests that change one parameter at a time, assert the earliest
   required regeneration point, assert downstream behavior, and then verify that
   a third identical run is a true cache hit.
