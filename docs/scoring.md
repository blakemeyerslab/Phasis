# Understanding Phasis scores

[Back to the output guide](../README.md#understand-the-outputs)

Phasis reports two different types of score. The Phasis score summarizes
statistical evidence for phasing; Howell scores describe the abundance and
spacing of sRNAs within a candidate window. A high score alone does not
establish a high-confidence *PHAS* call: classification, filtering, and
register-resolved interpretation also contribute to the final result.

## Phasis score

For each candidate cluster, Phasis combines its available window P values
using Fisher's method and reports `-log10(combined P value)`, capped at 300.
Higher values indicate stronger statistical evidence. Values at 300 reach
the reporting cap and should not be interpreted as identical underlying
evidence. The `-phasisScoreCutoff` option controls the corresponding score
filter; other phase-dependent checks also apply.

## Howell scores

A phase register is a series of expected positions separated by the chosen
phase length, usually 21 or 24 nt. Phasis evaluates strands separately and
uses only reads whose length equals that phase length for Howell scoring.
Windows span ten phase cycles: 210 nt for 21-*PHAS* and 240 nt for 24-*PHAS*.

For a register, the implemented score has the form:

```text
H = (min(k, C) - 2) * ln(1 + 10 * I / (1 + O))
```

Here, `k` is the number of occupied cycles, `C` is the number of cycles in the
window, `I` is in-phase abundance, and `O` is out-of-phase abundance.
`ln` is the natural logarithm. A register with fewer than four occupied
cycles, or no in-phase abundance, receives a score of zero. The distinction
between the following formulations lies in how positions are assigned and
registers are selected, not just in the formula.

### Main Howell score: positional tolerance

`Peak_Howell_score` is the peak of the positional-tolerance formulation used
by Phasis. Internal code also calls this the relaxed or wobble-tolerant Howell
score; this describes positional tolerance, not a separate analysis mode.

At each expected register position, the exact position is preferred. If it
has no read, a position one nucleotide to either side can be used; when both
neighbors are present, the more abundant one is selected. Only one position
is counted per cycle, and a position is not counted twice. Neighboring
positions excluded by this choice are also excluded from the effective
out-of-phase abundance, rather than penalizing the score.

Registers are compared by their resulting score. The peak is selected across
the evaluated windows and strands. This tolerance can capture slightly offset
signal, but can also produce a high score when exact-register support is weak.

### Strict Howell score: exact positions only

`Peak_Howell_score_strict` is computed in a separate scan with no positional
tolerance. Only reads at exact register positions count as in-phase; other
phase-length reads in the window contribute to out-of-phase abundance.

In this implementation, the strict scan selects the register with the largest
in-phase abundance within each window, computes its score, and then selects
the maximum across windows and strands. This differs from selecting the
highest-scoring register in the positional-tolerance calculation. The two
reported peaks therefore need not describe the same window or register.

### Exact support and RRL

`Howell_exact_support_score` is another distinct quantity: it summarizes
exact-register support in the selected positional-tolerance peak window.
It is not an alias for the independently scanned `Peak_Howell_score_strict`.

The Register-Resolved Locus Interpretation Layer (RRL) considers exact support
alongside competing registers, extensions, and other structural evidence.
Consequently, a high positional-tolerance Howell score does not by itself
guarantee a high-confidence *PHAS* call. Candidates can instead be interpreted
as *PHAS*-like or non-*PHAS*.

The `-min_Howell_score` post-filter applies to `Peak_Howell_score` (default
12.5), not directly to `Peak_Howell_score_strict`. Strict and exact-support
measurements also inform interpretation; the strict score is not simply a
replacement threshold for the main score.

## Scores in the plots

- `{phase}_Howell_scores.pdf` shows the main peak Howell score and the strict
  peak Howell score in separate heatmap panels.
- The bottom panel of an individual locus plot shows Howell window scores
  and register context, not the cluster-level Phasis score. Its red HPSP marker
  identifies the highest phasing score position used as the register anchor.
  Additional markers distinguish nearby and secondary windows. The default
  clean display hides some off-register trace points; it is not a separate
  plot of the strict peak statistic.

## Implementation references

These descriptions refer to the maintained source implementation:

- [Window scoring](../phasis/stages/window_scoring.py):
  `compute_scores_for_group` calculates the Phasis score.
- [Feature assembly](../phasis/stages/feature_assembly.py):
  `compute_phasing_score_Howell`, `compute_phasing_score_Howell_strict`, and
  `_summarize_peak_howell_ambiguity` calculate Howell features.
- [Classification](../phasis/stages/classify.py): `_apply_post_filters`
  applies the main score and Howell cutoffs.
- [Locus plots](../phasis/stages/locus_plots.py) and
  [output generation](../phasis/stages/output.py) draw the diagnostic plots
  and heatmaps.
