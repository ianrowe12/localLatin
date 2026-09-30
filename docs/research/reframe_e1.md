# Reframe experiment E1: coordinate ablation (issue #246)

Results for James's experiment E1 of the analysis reframe
(`reframe_handoff_20260927.md`), and the paper edits that put them into
`overleaf_drafts/acl_latex.tex`. Every number below is in
`runs/active/reframe/e1/facts_e1.md` (generated) or in a CSV beside it.

Short version:

- **The main prediction fails.** Zeroing the top k residual coordinates restores
  0 of the 26 collapsed T5 layers to AUROC >= 0.90, under either ranking, for
  k <= 5 and for k = 10. The few-massive-coordinates account of the collapse
  does not hold as stated.
- **Standardization partly works.** Per-coordinate standardization restores 9
  of 26 (LaTa 1 of 10, PhilTa 5 of 9, mT5-base 3 of 7) and recovers a median 79
  percent of the ABTT D=10 gain. Centering alone restores 0 of 26, ABTT D=1
  1 of 26, ABTT D=3 all 26.
- **The dominant direction is concentrated on about ten coordinates but not
  confined to them** (descriptive, added after the ablation results were seen).
  The ten highest-variance coordinates hold a median 92 percent of PC1's
  squared loading at the collapsed layers, yet zeroing them leaves one
  direction dominant and AUROC below 0.80. Why is not tested.
- **The embedding-trained models behave as predicted**, and the ratio r
  separates collapsed from healthy layers less cleanly than predicted (18 of
  26 collapsed layers, all 8 misses in PhilTa).

## Provenance

| Item | Value |
|---|---|
| Script | `scripts/paper/reframe/e1_coordinate_ablation.py` (subcommands `compute`, `check`, `render`) |
| Batch file | `slurm/reframe/reframe_e1.sbatch` (CPU, 16 cores, `--time=00:20:00`) |
| Jobs | 22572008 (five cached models, 76 model-layers, 2 min 46 s), 22572072 (all six, 100 model-layers, 3 min 50 s), 22572402 (final run with the concentration measure, 3 min 35 s, exit 0), all on `cpu-interactive`. The first two exited with the gate code because two D=0 cells missed 1e-6 (below); their ablation, coordinate and cosine-share CSVs equal those of the final run |
| Inputs | `runs/active/resubmit/data/phase_resubmit_split.csv`; mean-pooled hidden states `runs/active/resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/`, re-extracted on Delta in James's checkout (Ian's cache is not readable from it) |
| Outputs | `runs/active/reframe/e1/e1_coordinate_ablation.csv` (1,400 rows: base and 13 interventions at 100 model-layers), `e1_top_coordinates.csv`, `e1_cosine_shares.csv`, `e1_concentration.csv`, `e1_gate_check.csv`, `facts_e1.md`; `overleaf_drafts/tables/e1_coordinate_ablation.tex` |
| Tests | `tests/test_e1_coordinate_ablation.py` (25 tests: synthetic guards, gate behavior, byte-identical regeneration of the table from the committed CSV) |

Regenerate the table and the facts file from the committed CSVs with
`python scripts/paper/reframe/e1_coordinate_ablation.py render` (seconds, no
embeddings needed).

### Protocol

- Six models, every layer: LaTa, PhilTa, mT5-base and LaBSE 12 layers each,
  Qwen3-0.6B 28, KaLM-mini 24 (100 model-layers). 847 train and 858 test
  passages.
- Every statistic (coordinate ranking, mean, SD, principal components) is fit
  on the training passages and applied to both splits. Interventions act on
  the raw mean-pooled vectors; nothing is centered before zeroing.
- Rankings: mean absolute value of a coordinate over training passages, and its
  variance across them. k in {1, 3, 5, 10}.
- Thresholds are those of the Sec. 5 paragraph and its `\pending` marker, fixed
  before the results: restored = AUROC >= 0.90, k <= 5, top-PC share < 0.2,
  |change| <= 0.03, collapsed = baseline AUROC < 0.70 (26 layers: LaTa 2 to
  11, PhilTa 3 to 11, mT5-base 5 to 11).
- r = SD / |mean| of a pooled coordinate over training passages (population SD).
- Cosine share: Timkey and van Schijndel's summed per-coordinate contribution
  of the top k coordinates over all distinct pairs of a split, divided by the
  summed pairwise cosine, on L2-normalized raw vectors.

## Reproduction gates

All pass at their default tolerance (`e1_gate_check.csv`, facts section 0).

| Gate | Reference | Cells | Tolerance | Largest difference |
|---|---|---|---|---|
| 1 base AUROC | `phase_resubmit_results.csv` (published baseline) | 100 | 1e-6 | 2.31e-7 (Qwen3-0.6B) |
| 2a D=0 (centering) AUROC | H1 CSV | 100 | 1e-5 | 3.60e-6 (mT5-base layer 5) |
| 2b ABTT D=1, 3, 10 AUROC | H1 CSV | 300 | 1e-6 | 5.37e-7 (mT5-base) |
| 3 base top-PC share | `geometry_per_layer.csv` (train, raw) | 100 | 1e-4 | 2.42e-7 (KaLM-mini) |

**Why centering has its own tolerance.** Two mT5-base D=0 cells differ from H1
by more than 1e-6: layer 5 by 3.6e-6 and layer 9 by 1.05e-6. No other D=0 cell
exceeds 1e-6, and outside mT5-base the largest difference is 8.2e-8. At the collapsed mT5-base layers the centered
vectors are near rank one (top-PC share 0.999 or more), and these cells are
not stable at 1e-6: on one cache they move by 1e-6 to 1e-5 between float32 and
float64 arithmetic or when every cached value moves by one float32 ulp, and by
up to 3e-7 with the BLAS thread count (diagnostic job 22572037). The
independent recomputation (next section) gives 0.67010057 in float64 against
0.67009702 in float32 at layer 5. Doing either the training mean or the cosine
in float64 removes the gap (0.67010099 and 0.67010054), so it is float32
rounding at a near-rank-one layer and needs both steps in float32. The cells
agree with H1 to five decimals. The tolerance is `GATE_TOL_CENTER` in the
script, with this reason beside it.

## Independent replication

`/projects/bimc/swong2/setup/e1_replicate.py` (job 22572386, 34 s) recomputes
15 named cells from the cached vectors and the split alone, without importing
the repository's reframe code, with its own filename join. All agree with the
facts file to the reported precision:

| Cells | Quantity |
|---|---|
| 1 to 4, 8 | LaTa layer 6: base AUROC 0.496; zero by variance k=5 0.516, k=10 0.574; standardization 0.822; top-PC share 0.934 base, 0.712 after zeroing k=10 |
| 5 | PhilTa layer 10: standardization 0.907 |
| 6, 7 | mT5-base layer 5: zero by variance k=10 0.716; D=0 0.670 (float64 and float32 above) |
| 9, 11, 12 | LaTa layer 7: cosine share 0.218 of a mean pairwise cosine of 0.228; top three coordinates #665, #764, #172 with mean \|x\| 805, 780, 753 against a median of 9.47; r 1.83, 1.15, 2.06 |
| 10, 15 | mT5-base layer 1: cosine share 0.858 of a mean pairwise cosine of 0.951; r 0.052, 0.109, 0.122 |
| 13 | PhilTa layer 7: r 1.27, 0.94, 1.31 |
| 14 | Qwen3-0.6B layer 1: r 0.05, 0.04, 0.12 |

A follow-up (`e1_replicate_philta7.py`, job 22572391) prints the top ten
PhilTa coordinates at layers 6 to 8 and confirms that one of the top three by
mean |x| (coordinate #239) has r just below 1 while the top three by variance
all exceed 1.

## KaLM-mini re-extraction

The handoff lists five models for E1. KaLM-mini was added so that the whole
panel is covered. Its vectors were not in James's cache:
`/projects/bimc/swong2/setup/extract_kalm.sbatch` (job 22571953,
`gpuA40x4-interactive`, 14 min 52 s) extracts layers 1 to 24 with the
repository CLI and the arguments of `slurm/resubmit/resubmit_extract_kalm.sbatch`,
then rescores them. The 24 baseline AUROC cells match the published ones
within 8.7e-8 (`reproduce_baseline_kalm.csv`); E1's own gate 1 for KaLM-mini,
a separate computation, gives 9.83e-8.

## Findings

Verdicts are those of facts section 8.

### Zeroing: FAIL

| At the 26 collapsed T5 layers | By mean \|x\| | By variance |
|---|---|---|
| Restored to AUROC >= 0.90, some k <= 5 | 0 of 26 | 0 of 26 |
| Restored, k = 10 | 0 of 26 | 0 of 26 |
| Best AUROC, k <= 5 | 0.736 | 0.736 |
| Best AUROC, k = 10 | 0.794 | 0.775 |
| Top-PC share < 0.2, some k <= 10 | 0 of 26 | 0 of 26 |
| Top-PC share at k = 5 | 0.568 to 0.998 | 0.568 to 0.998 |
| Top-PC share at k = 10 | 0.356 to 0.995 | 0.324 to 0.995 |
| D=10 gain recovered at k = 10, median (range) | 16% (4% to 43%) | 16% (3% to 41%) |

All four best cells are mT5-base layer 11. Base top-PC share at these layers is
0.764 to 1.000.

### Variance ranking against magnitude ranking: NO VERDICT

The two rankings are nearly the same ranking at the collapsed layers: on
average 2.4 of the top 3 and 9.1 of the top 10 coordinates are shared. The
AUROC difference has median 0.000 at every k; the variance ranking is higher
in 29, lower in 25 and equal in 50 of the 104 (layer, k) cells. The paragraph
gives no size for "should help less", so no verdict is attached.

### Standardization, centering, projections

| At the 26 collapsed layers | Restored to >= 0.90 | AUROC range | D=10 gain recovered, median (range) | Top-PC share < 0.2 |
|---|---|---|---|---|
| Standardization | 9 (LaTa 1 of 10, PhilTa 5 of 9, mT5-base 3 of 7) | 0.798 to 0.946 | 79% (47% to 93%) | 14 |
| Centering (D=0) | 0 | 0.482 to 0.681 | -4% (-10% to 7%) | (unchanged by definition) |
| ABTT D=1 | 1 (LaTa layer 2) | 0.598 to 0.925 | 45% (13% to 89%) | 0 |
| ABTT D=3 | 26 | 0.914 to 0.974 | 96% (80% to 102%) | 26 |
| ABTT D=10 | 26 | 0.962 to 0.984 | | 26 |

Centering changes AUROC by -0.044 to +0.022.

### The paragraph's decision rule

- Zeroing k <= 5 repairs: 0 of 26.
- Standardization repairs, zeroing does not ("a few coordinates, but more than
  five"): 9 of 26 (LaTa 2; PhilTa 3, 4, 8, 10, 11; mT5-base 9, 10, 11).
- Neither repairs, a projection does ("spread over many coordinates of similar
  variance"): 17 of 26.

Zeroing ten coordinates under either ranking repairs none of the 9 either.

### Concentration (descriptive, post hoc)

Added after the ablation results were seen (facts section 9). It carried no
prediction and changes no verdict. At the 26 collapsed layers, median (range):

| Measure | All 26 | LaTa | PhilTa | mT5-base |
|---|---|---|---|---|
| Training variance on the 10 highest-variance coordinates | 0.891 (0.611 to 0.996) | 0.893 | 0.773 | 0.996 |
| PC1 squared loading on those 10 | 0.918 (0.759 to 0.997) | 0.919 | 0.835 | 0.996 |
| Coordinates holding 90% of PC1's squared loading | 10 (4 to 22) | 10 | 12 | 4 |
| Participation ratio of PC1 | 7.4 (3.7 to 15.3) | 7.3 | 11.7 | 3.7 |

So the gloss of the rule's last branch does not fit: the dominant direction is
not spread over many coordinates of similar variance. It is concentrated on
about ten coordinates but not confined to them. After those ten are zeroed,
one direction still dominates what remains (top-PC share 0.324 to 0.995) and
AUROC stays at 0.775 or less. Removing the coordinates is not the same as
removing the direction. Why the remainder is still dominated by one direction
was not tested.

### Embedding-trained models: PASS

| Model | Train-selected layer | Largest \|AUROC change\| over the eight zeroings | Standardization | Weakest layer | Zeroing, share of D=10 gain | Centering, share of D=10 gain |
|---|---|---|---|---|---|---|
| LaBSE | 12 | 0.0044 | +0.0064 | 1 | -18% to -1% | 36% |
| Qwen3-0.6B | 26 | 0.0071 | +0.0121 | 1 | 6% to 18% | 80% |
| KaLM-mini | 23 | 0.0024 | +0.0088 | 5 | -4% to 6% | 46% |

All inside the predicted 0.03. The "small part of the ABTT gain" prediction
has no threshold in the paragraph and gets no verdict; the numbers are above.

### Ratio r: FAIL at collapsed layers, PASS at the healthy ones

- Top 3 by mean |x|: all three have r >= 1 at 18 of 26 collapsed layers (LaTa
  10 of 10, mT5-base 7 of 7, PhilTa 1 of 9). PhilTa layers 5 to 10: one of the
  three lies between 0.81 and 0.96, the other two exceed 1 (layer 7: 1.27,
  0.94, 1.31). PhilTa layer 3: 0.21, 0.51, 1.52; layer 4: 1.04, 0.61, 1.38.
  At least one of the three has r >= 1 at all 26.
- Top 3 by variance: all three have r >= 1 at 26 of 26.
- r < 1 for the top 3 by mean |x| at mT5-base layer 1 (0.052, 0.109, 0.122)
  and at every layer of Qwen3-0.6B (at most 0.20), LaBSE (at most 0.39) and
  KaLM-mini (at most 0.39).

### Cosine shares

Top 3 coordinates by mean |x|, pairs of training passages (test in
parentheses): LaTa layer 7 0.218 (0.224), mean pairwise cosine 0.228;
mT5-base layer 1 0.858 (0.856), mean pairwise cosine 0.951. By variance the
mT5-base layer 1 share is 0.460.

What the share measures: the mean over distinct pairs of a coordinate's
contribution equals the square of that coordinate's mean over L2-normalized
passages minus its variance divided by n - 1. It therefore credits a
coordinate for shifting all passages together, not for varying between them.
A low share at a collapsed layer and a high one at a healthy layer is what the
offset-against-variation distinction predicts. It is not evidence that three
coordinates dominate similarity at LaTa layer 7.

## Deviations from the handoff

1. **KaLM-mini added.** The handoff's E1 row names five models; the run covers
   all six.
2. **Concentration measure added post hoc.** Not in the handoff or the
   paragraph. The paper says it is descriptive and carried no prediction.
3. **A quoted number does not reproduce.** The draft said that at LaTa layer 7
   "three coordinates have a mean magnitude of about 5,000 against a median of
   36". On the mean-pooled vectors the top three have mean |x| 805, 780 and 753
   against a median of 9.47. Diagnostic job 22572045 tried other readings of
   the same vectors (|mean|, maximum, 99th percentile, SD, RMS, range); the
   nearest are a per-coordinate range of 5,071 and a median maximum of 44.5,
   and no single statistic gives both numbers. No source for the quote is in
   the repository. "AUROC is 0.82 and top-PC share is 0.23" at mT5-base layer 1
   does reproduce (0.8216, 0.2289).
4. **D=3 reference column.** The handoff row names ABTT D=1 only. The D=3
   column follows item 8 of the integration list in `reframe_e3_h1_whiten.md`.
   The table's second block carries the top-PC share the handoff asks for.
5. **Table column spacing.** With numbers in every cell the generated table was
   13.7pt wider than the text block; `\tabcolsep` in the render code went from
   4pt to 3pt. No value changed.
6. **Handoff claim row "Massive-coordinate mechanism".** Its surviving wording
   ("confirmation in a new setting: in T5 encoders the known mechanism drives
   mean-pooled passage retrieval to chance") does not survive E1. What E1
   supports: zeroing outlier coordinates, which prior work reports as helpful
   in its settings, does not repair the collapse here; a diagonal rescaling
   recovers most of the gain at some layers; a three-component projection
   repairs all.

## Paper edits

All in `overleaf_drafts/acl_latex.tex`; line numbers are those before the
edit. James's region: the E1 paragraph and `tab:e1_coordinate_ablation`. The
other edits are outside it and were made only where an E1 result decides the
sentence.

| # | Where | Region | Change |
|---|---|---|---|
| 1 | Abstract, line 72 | outside | The sentence with `\pending{E1, E2, P2x2: ...}` becomes three: the test, the E1 outcome (zeroing up to ten restores none of 26, standardization 9), and the pretraining-or-objective question with `\pending{E2, P2x2: one-sentence result}` |
| 2 | Contribution (3), line 129 | outside | `\pending{E1, E2: ...}` replaced by the E1 outcome plus `\pending{E2: token audit result}` |
| 3 | Sec. 5 preamble, lines 381 to 385 | outside | "Preliminary evidence fits this account." removed. LaTa layer 7 magnitudes 5,000 and 36 replaced by 805, 780, 753 and 9.5. mT5-base layer 1 magnitudes added. Both `\pendingnum{E1: ...}` shares filled (0.22, 0.86, with the mean pairwise cosines), followed by what the share measures. The hypothesis and the r prediction are unchanged |
| 4 | E1 paragraph, lines 397 and 404 | James | KaLM-mini added to the model list and to the embedding-trained prediction. "0.007 to 0.031" kept: the headline table gives ABTT gains of 0.031 (LaBSE), 0.007 (Qwen3-0.6B) and 0.009 (KaLM-mini) |
| 5 | E1 paragraph, lines 409 and 410 | James | Table sentence names both blocks. `\pending{E1: ...}` replaced by the findings |
| 6 | Placeholder table, lines 412 to 432 | James | Replaced by `\input{tables/e1_coordinate_ablation}` |
| 7 | Scope, after line 508 | outside | Five sentences added after the unchanged falsifiers: the first occurs at 17 of 26 (at the other 9 standardization repairs, zeroing does not), the second does not occur (Qwen3-0.6B r at most 0.20), the claim about the cause reduces to the geometric description and the repair, with the concentration qualification |
| 8 | Discussion, line 673 | outside | `\pending{E1, E2: ...}` replaced by the E1 answer plus `\pending{E2: whether specific tokens carry these coordinates}` |
| 9 | Discussion, line 679 | outside | "a few massive coordinates carry much of the pooled cosine" becomes "three massive coordinates carry 0.86 of the mean pairwise cosine", with r 0.05 to 0.12 |

Build: `/projects/bimc/swong2/setup/build_paper.sh` gives 58 pages (57
before), no undefined reference or citation, no duplicate label, no overfull
box.

## For the first author: sentences not edited

1. **Sec. 5 Scope, last sentence**: "whether T5 pretraining or the missing
   embedding objective produces them". "Them" is the coordinates. After E1 the
   question of Sec. 6 is what produces the collapse, as the abstract says.
2. **Introduction, line 114**: "We test whether mean pooling turns these
   coordinates into values that vary from passage to passage and so drown out
   content". Still true as a statement of the test; the outcome could be added.
3. **Discussion, line 678**: "the E2 audit tests whether their coordinates
   shift passages together instead of varying across them". E1 already
   measures this on the pooled vectors: r is below 1 for the top three
   coordinates at every layer of Qwen3-0.6B and KaLM-mini.
4. **E2 paragraph** (James's region, left for the E2 pass): its `\pending`
   still lists "ratio r of the pooled coordinates" and "Qwen3-0.6B coordinates
   have r < 1", both now reported under E1. Its premise "some tokens hold the
   massive values" should be read against the failed zeroing.
5. **Sec. 5 preamble**: "The testable content is that this direction is
   aligned with a few coordinates and has token carriers." Unchanged; E1 finds
   the direction concentrated on about ten coordinates but not removable by
   zeroing them.
6. **Related work**: nothing presupposes that the account holds here. The
   sentence on outlier dimensions describes prior work and stands.
7. **Page budget**: the E1 findings take 13 sentences where the brief asked for 8
   to 12, and the paper grew from 57 to 58 pages.
