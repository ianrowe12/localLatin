# Attribution re-sample on benchmark v1

Issue #141 (G6), part of epic #109. Companion to
`docs/research/attribution_metrics_decision.md`, whose Part A and Part B this
re-derives on the corrected corpus.

**Status: complete. One paper sentence changed, and it is a weaker claim than
the one it replaces.** `rho_LOO` is 5/6, not 6/6. The sixth cell is a
statistical tie, not a loss.

## Why this run exists

Issue #113 found that `runs/active/ig_examples_200pos_run3_operational` sampled
its 600 pairs from the legacy phase-9 split over `data/canon` (1,278 files), not
from the paper's split over `data/canon_labelled` (1,705 files, benchmark v1).
At least five of the 600 rows are negatives under the corrected labels; by
filename identity the count is 28 (5 involving `BN2123.89r.5`, 23 from the
`CNIC.325` renumbering of BN2123 items). The two counts differ because the old
sample records `data/canon` paths and `file_id`s from a different split, so
mapping a run-3 row onto benchmark v1 requires an identity rule; filename is the
one used here. Every attribution
number in the paper therefore rested on a sample drawn from a different corpus
and a different split than every retrieval number in the same paper. The layer
contract was never in doubt: LaTa 7, PhilTa 1, mT5-base 1 at `D=10` are selected
by a train-only retrieval rule and are unchanged here.

## Provenance

| Item | Value |
|---|---|
| Run directory | `runs/active/ig_examples_200pos_v1/` |
| Corpus | `data/canon_labelled/` (benchmark v1) |
| Split | `runs/active/resubmit/data/phase_resubmit_split.csv` |
| Sampler | `scripts/ig/sample_positive_test_pairs.py`, seed `20260426` (unchanged) |
| Pool | 585 winnable test files across 241 folders |
| Sample | 200 positive pairs per model, 600 total, 510 distinct files, 191 folders |
| Layers | LaTa 7, PhilTa 1, mT5-base 1, `D=10` |
| Cleaners | `runs/active/ig_examples_200pos_v1/pcs/`, refit train-only |
| Cleaner source | `runs/active/resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/` |
| `pair_manifest.tsv` sha256 | `4fcdaa2be99ebcb279c16d47ce2af158411727df27c5b86414404780a43591c8` |
| `positive200_examples.csv` sha256 | `4a3399e5b6d1e234b444febc75f7b5a6dfa44da39142adbe0b70ee076364e54a` |
| New summary | `runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2.csv` |
| Old summary (kept, untouched) | `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv` |

`pair_manifest_sha256` digests only `(model, query filename, candidate filename,
folder)`, so it identifies the sample itself and is stable against the
`methods_available` column that the merge step rewrites in place. It is **not**
`sha256sum pair_manifest.tsv` (that is `bf064cf2...`, and covers the file's
trailing newline). `scripts/ig/pair_manifest_digest.py` is the definition and
verifies it:

```bash
python scripts/ig/pair_manifest_digest.py \
  --examples_csv runs/active/ig_examples_200pos_v1/positive200_examples.csv \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --expect 4fcdaa2be99ebcb279c16d47ce2af158411727df27c5b86414404780a43591c8
```

**Overlap with run 3**: 48 of the 600 (model, pair) rows reuse a run-3 pair, 35
distinct pairs, and 152 of the 510 files are shared. The two samples are drawn
from overlapping corpora, so they are not independent.

### Sample verification

All four checks pass, and are re-runnable from the manifest:

* 600 of 600 pairs are positives under the corrected labels.
* 1200 of 1200 files are `split == test`.
* 0 pairs reference a `data/canon`-only path; every path is under
  `data/canon_labelled/`.
* no self-pairs, and every referenced file exists on disk.

### Alignment

`scripts/ig/refit_pcs_for_attribution.py` selected its fit rows positionally
(`np.where(split["split"] == "train")`), which is the hazard PR #139 was written
to remove: the cache is frozen in corpus-walk order while the split is re-sorted
by `(folder_id, filename)`, so the benchmark v1 correction moves seventeen rows
and a positional read silently fits on seventeen wrong vectors with every row
count still equal. It now loads through `AlignmentResolver` and says so:

```
[alignment] .../bowphs_LaTa/hidden_mean_tokempty:
verified-permuted (1705 rows, 17 moved, manifest .../meta.csv)
[bowphs_LaTa] fitting D=10 on layer 7 train ((847, 768))
row alignment: 3 cache dir(s) [3 verified-permuted]; rows moved per cache: [17]
```

Seventeen, matching the permutation table in `benchmark_v1.md` exactly.

## Compute

| Stage | Job | Partition | Elapsed | Reserved |
|---|---|---|---|---|
| IG + MaRC + 7-method persistence | `22070707` | `gpuA100x4` | **00:58:45** | **01:25:00** |
| Metrics + operator spot check | `22072507` | `cpu` | 00:37:50 | 03:00:00 |

```
22070707  attr_200pos_v1  COMPLETED  00:58:45  01:25:00  0:0
```

**GPU cost: 1.42 GPU-hours** (SLURM charges the reservation, not the elapsed
time). One GPU job, as briefed.

Run 3 took 4 h 17 m for the same work in one stream. Two changes bought the
reduction: the three models run as concurrent streams on the one A100 (these are
T5-base encoders at batch 1, so a single stream leaves the card mostly idle),
and the metrics stage moved to the CPU partition, where it belongs. The metrics
stage alone was 2 h 32 m of run 3's GPU time.

## The pooling defect this run exposed

`canon_labelled` is only cached at `hidden_mean_tokempty`, so the cleaners had to
be fit through the `tokenizer_empty` pooling and the artifacts generated with
`--token_filter tokenizer_empty`. Run 3 used `--token_filter all` throughout.
That change is what surfaced a latent bug in the metrics stage.

**The first metrics pass produced nonsense, and the pipeline's own self-check
caught it.** Mean pair cosine under ABTT rose from 0.923 to 0.983 for LaTa, the
signature of cosine inflation documented in
`docs/analyses/cosine_inflation_investigation.md`. The `full_cos_drift` column,
which compares the backend's recomputed unmasked cosine against the value stored
at artifact-build time, read **0.04 to 1.14** where run 3 reads 2.5e-07.

### Diagnosis

`scripts/_archive/run_phase12e_pair_explanations.py` applies its token filter
inside `pool_hidden`, as a pooling mask, and stores the **unfiltered** per-token
hidden states in the NPZ. `run_attribution_metrics.py` then meaned over every
attended row, so it reconstructed a different document vector than the one the
generator pooled and than the one the PCs were fit on, and applied the cleaner to
it. Under `--token_filter all` the two are identical, which is why no published
run ever saw this and why every run 3 number is unaffected.

The decisive measurement, artifact pooling against the cache the PCs came from:

| Model / layer | filtered pooling vs cache | unfiltered pooling vs cache |
|---|--:|--:|
| LaTa L7 | **+0.999960** (min +0.998381) | -0.058228 |
| mT5-base L1 | **+0.999997** (min +0.999899) | +0.950327 |

So the GPU artifacts were correct all along: they live in exactly the space
their PCs were fit on. The fault was entirely in the CPU metrics stage, and the
fix is CPU-only. `run_attribution_metrics.py` gains a `--token_filter` flag
(default `all`, which reproduces every published run bit for bit) and applies
the keep mask to the pooled rows, the pair matrices and the stored IG scores in
both backends. In the model backend the filter is a *pooling weight*, never an
attention mask, matching the generator: a filtered token still contextualises
its neighbours inside the encoder, it just does not enter the mean.

LaTa layer 7 is where the two poolings diverge most, and that is not a
coincidence. It is the anisotropy dip: unfiltered, its pooled vectors are so
dominated by one common direction that unrelated documents sit at cosine 0.93,
and that direction is carried by the tokens the filter drops.

### A second defect, not fixed here

`src/retrieval_mask.py` computes `cos_orig_*` as
cos(**unfiltered** query mean, **filtered** candidate partner): the query side is
a plain `q_hidden.mean(axis=0)` that never consults the token filter, while the
partner vector comes through the filtered pooling. That mixture is what the
stored values actually are, and it reproduces them exactly (examples 001-003:
-0.6840, -0.2184, -0.3067, where pooling both sides unfiltered gives +0.99,
+0.99, +0.87). Under any filter other than `all` it therefore describes a
different vector than the generator pooled, so it cannot serve as the drift
reference. `full_cos_drift` is now
emitted as `NaN` under a token filter, with a printed note, rather than
reporting a disagreement between two stored quantities as an error in the
backend. **Consequence: this run has no independent full-cosine reproduction
check.** That check is worth restoring and is the natural follow-up issue; it
needs `cos_orig_*` to be written through the same pooling the generator uses.

## Old versus new

Full cell-by-cell output, regenerable:

```bash
python scripts/ig/compare_attribution_runs.py \
  --out runs/active/ig_examples_200pos_v1/attribution_metrics/old_vs_new.md
```

### Wins per metric

| Metric | Dir | old ABTT wins | new ABTT wins | cells that flipped |
|---|---|---|---|---|
| **rho_LOO** | higher | **6/6** | **5/6** | PhilTa/MaRC A->b |
| **DelAUC gap** | higher | **3/6** | **4/6** | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| InsAUC gap | higher | 5/6 | 5/6 | - |
| tau_LOO | higher | 6/6 | 5/6 | PhilTa/MaRC A->b |
| AOPC-Suff | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| AOPC-Comp | higher | 3/6 | 4/6 | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| DelAUC | lower | 3/6 | 4/6 | LaTa/MaRC A->b, PhilTa/IG b->A, PhilTa/MaRC b->A |
| InsAUC | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| Suff@25% | higher | 2/6 | 4/6 | PhilTa/IG b->A, mT5-base/MaRC b->A |
| Comp@25% | higher | 2/6 | 4/6 | PhilTa/IG b->A, PhilTa/MaRC b->A |
| MinFrac@0.80 | lower | 1/6 | 2/6 | PhilTa/IG b->A |

### The two main-table columns, cell by cell

Cells read `baseline -> ABTT`. **A** marks an ABTT win.

| Cell | rho_LOO old | rho_LOO new | DelAUC gap old | DelAUC gap new |
|---|---|---|---|---|
| LaTa/IG | 0.013 -> 0.298 **A** | 0.003 -> 0.370 **A** | 0.504 -> 0.178 b | 0.842 -> 0.180 b |
| LaTa/MaRC | 0.070 -> 0.397 **A** | 0.083 -> 0.538 **A** | 0.244 -> 0.287 **A** | 0.506 -> 0.327 b |
| PhilTa/IG | 0.144 -> 0.577 **A** | 0.329 -> 0.614 **A** | 0.806 -> 0.420 b | 0.112 -> 0.400 **A** |
| PhilTa/MaRC | 0.179 -> 0.367 **A** | 0.469 -> 0.445 b | 0.758 -> 0.353 b | 0.200 -> 0.310 **A** |
| mT5-base/IG | 0.138 -> 0.626 **A** | 0.143 -> 0.686 **A** | 0.394 -> 0.548 **A** | 0.061 -> 0.561 **A** |
| mT5-base/MaRC | 0.272 -> 0.415 **A** | 0.257 -> 0.504 **A** | 0.075 -> 0.431 **A** | 0.042 -> 0.464 **A** |

### Effect sizes on the new sample

**Paired** ABTT-minus-baseline differences, over the pairs valid under both
variants: the two variants score the same pairs, so the difference is paired and
its standard error is smaller than the unpaired combination
`sqrt(se_base^2 + se_abtt^2)` that the summary's per-variant columns would give.
These are the numbers `paired_cell_stats` computes, which is the same statistic
the table caption uses to decide which cells are ties, and the same statistic the
paper prose quotes. Read from the per-pair JSON cache, not from `summary_v2.csv`:

```python
import build_main_attribution_artifacts as b
b.paired_cell_stats(Path(".../attribution_metrics/v2_hidden"), b.RHO_KEY)
```

| Cell | rho_LOO | DelAUC gap |
|---|--:|--:|
| LaTa/IG | +0.367 +/- 0.023 (16.0 SE) | -0.634 +/- 0.078 (-8.1 SE) |
| LaTa/MaRC | +0.455 +/- 0.027 (17.0 SE) | **-0.159 +/- 0.089 (-1.8 SE)** |
| PhilTa/IG | +0.285 +/- 0.016 (17.5 SE) | +0.287 +/- 0.010 (27.7 SE) |
| PhilTa/MaRC | **-0.024 +/- 0.016 (-1.5 SE)** | +0.110 +/- 0.014 (7.7 SE) |
| mT5-base/IG | +0.543 +/- 0.024 (22.5 SE) | +0.500 +/- 0.021 (23.3 SE) |
| mT5-base/MaRC | +0.247 +/- 0.024 (10.3 SE) | +0.422 +/- 0.023 (18.3 SE) |

**The one `rho_LOO` cell ABTT loses is a tie at 1.5 standard errors, not a
defeat.** The five wins run from 10.3 to 22.5 SE. That distinction is what the
changed paper sentence has to carry. Both ties sit inside the generator's 2 SE
threshold, so the caption names both and no verdict depends on the choice of
paired over unpaired.

### Shuffled-attribution control (criterion 5)

| Metric | old positive cells | new positive cells | new failures |
|---|---|---|---|
| rho_LOO | 12/12 | **12/12** | none |
| tau_LOO | 12/12 | **12/12** | none |
| DelAUC gap | 12/12 | **12/12** | none |
| AOPC-Comp | 12/12 | **12/12** | none |
| InsAUC gap | 10/12 | **11/12** | mT5-base/IG baseline (-0.024) |
| AOPC-Suff | 10/12 | **11/12** | mT5-base/IG baseline (-0.024) |

**The appendix sufficiency metrics still fail the shuffle control.** One cell
instead of two, and it is still a baseline cell, so the Part B reasoning is
unchanged: criterion 5 is per-cell and "in every cell", so `InsAUC gap` and its
identical twin `AOPC-Suff` stay out of the main table. The three columns with a
calibrated zero that the main table and its rank companion rest on are 12/12 on
both samples.

### Operator spot check (memo A8)

Does the baseline-versus-ABTT sign on the AUC columns survive a change of
erasure operator? Bounded exactly as in the memo: LaTa, IG only, the first 20
pairs, both variants, run under both operators on the identical subset, CPU
only, no GPU hours. Means over the 19 pairs valid under both variants.

| Operator | InsAUC gap | DelAUC gap |
|---|---|---|
| input-level (`--backend model`) | 0.418 -> 0.269, baseline | **1.110 -> 0.372, baseline** |
| representation-level (`--backend hidden`) | 0.580 -> 0.077, baseline | **1.102 -> 0.175, baseline** |

**`DelAUC gap`, the main-table column, agrees across operators**, and agrees with
the 200-pair verdict for this cell (LaTa/IG is a baseline win there too). That is
the same outcome A8 reports on the canon sample, so criterion 6 still passes for
`DelAUC gap` on the one cell we can afford to test. `InsAUC gap` also reads
baseline under both operators here, where on the canon sample its nominal winner
flipped between them; either way it is in the appendix because it fails
criterion 5, not because of this check.

The random-order floors again move far more between variants under the
input-level operator than the representation-level one, so the two operators'
absolute numbers still must never share a table.

### Pair counts

The ratio metrics are undefined below a full-query cosine of 0.05. On the canon
sample the baseline cleared that floor for all 200 pairs in every cell. It does
not here: LaTa loses one baseline pair and five ABTT pairs, because under the
filtered pooling its layer-7 baseline cosines sit near zero rather than near
0.93. DelAUC now averages 195 to 199 ABTT pairs against 199 to 200 baseline
pairs. The table generator printed a single baseline count and raised on
disagreement; it now prints a range.

## Verdict: what must change in the paper

**One sentence, and it weakens a claim.** `rho_LOO` 6/6 becomes 5/6.

Changed, in `overleaf_drafts/acl_latex.tex` (hand-written prose only; every
`.tex` table and figure comes from its generator). All five sentences are re-applied on top of the wording that #182 and #186 merged, not on the wording
that preceded them:

* **Section 5.4, the headline attribution sentence.** Was "improves rank
  faithfulness in all six model-view cells ($\rho_{\mathrm{LOO}}$ 6/6, with
  tie-corrected Kendall $\tau_b$ agreeing in all six) ... deletion faithfulness
  improves in only three of six, and one of those three wins is within 1.2
  standard errors of zero". Now five of six by 10.3 to 22.5 paired standard
  errors, the sixth a tie at 1.5, tau agreeing in the same five, and deletion
  faithfulness four of six with one of the two remaining cells a tie.
* **The abstract**, which said ABTT "improves leave-one-out rank faithfulness
  in every LaTa, PhilTa, and mT5-base model-method cell". That sentence was
  false against the regenerated attribution table (Table 4).
* **The Introduction contributions list**, which said ABTT "improves
  leave-one-out rank faithfulness" without qualification.
* **The Discussion**, which said "only the leave-one-out rank gain is stable
  across all six cells".
* **The Limitations paragraph**, same phrase.

`docs/research/attribution_metrics_decision.md` B5 is superseded for this sample
and now points here.

Checked and **not** changed:

* The operational layers. LaTa 7, PhilTa 1, mT5-base 1 are selected by the
  train-only retrieval rule and do not depend on the attribution sample.
* The Task A and Task B numbers in Section 5.4's first paragraph, which come
  from the retrieval tables, not from this run.
* The appendix reasoning for keeping the sufficiency metrics out of the main
  table. The control still fails, in one cell instead of two.
* The erasure-operator scope caveat, the two ranking conventions, and the
  shuffled-attribution wording.
* Every claim about geometry, anisotropy and layer selection.

### The honest caveat on this comparison

**The old and new runs differ in two ways at once, not one.** The sample changed
(different corpus, different split, different pairs) and the pooling changed
(`tokenizer_empty` instead of `all`, forced by which caches exist for
`canon_labelled`). This memo cannot attribute any individual flip to the
re-sample alone. What it does establish is that the *new* numbers are
internally consistent and correctly aligned, and those are the numbers that now
match the corpus every retrieval number in the paper is computed on, which is
the whole point of issue #141.

A run that isolated the sample change would need an unfiltered `hidden_mean`
cache over `canon_labelled` for the three T5 encoders, then IG and MaRC at
`--token_filter all`. That is an extraction pass plus another approximately
1.5 GPU-hours, and it would answer a methodological question rather than fix a
paper number. It is not proposed here.

## Reproduce

```bash
# 1. Sample (CPU, seconds)
python scripts/ig/sample_positive_test_pairs.py \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --repo_root /projects/beto/irowerojas/localLatin \
  --out_csv runs/active/ig_examples_200pos_v1/positive200_examples.csv \
  --n_per_model 200 --seed 20260426 \
  --models bowphs/LaTa bowphs/PhilTa google/mt5-base

# 2. Cleaners (CPU, seconds; prints the alignment status per cache)
python scripts/ig/refit_pcs_for_attribution.py \
  --slugs bowphs_LaTa bowphs_PhilTa google_mt5-base \
  --bases_root runs/active/resubmit_bases/phase9_bases \
  --pooling hidden_mean_tokempty \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --pc_root runs/active/ig_examples_200pos_v1/pcs --d 10

# 3. IG + MaRC + persistence (GPU, one job)
sbatch slurm/ig/run_attribution_200pos_v1.sbatch

# 4. Metrics + operator spot check (CPU)
sbatch slurm/ig/attribution_metrics_200pos_v1.sbatch

# 5. Paper artifacts
python scripts/ig/build_main_attribution_artifacts.py \
  --summary_csv runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2.csv
python scripts/ig/package_attribution_sweep_appendix.py \
  --summary_csv runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2.csv \
  --long_out runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2_sweep_long_appendix.csv \
  --missing_report_out runs/active/ig_examples_200pos_v1/attribution_metrics/appendix_sweep_v2_completeness.json
```

**Regenerating the main table needs the per-pair cache.** The caption's tie
clause is computed from paired per-pair differences under
`attribution_metrics/v2_hidden/`, which is gitignored and rebuilt by step 4. The
generator now fails with a clear message if that directory is absent rather than
silently dropping the clause; `--no_tie_clause` is the explicit opt-out.

The generators take the new run through their existing `--summary_csv` flag.
Their module-level `DEFAULT_SUMMARY` still points at the run 3 directory, which
is deliberate: run 3 stays reproducible byte for byte, and the new path is
passed explicitly by the sbatch and by the commands above.

`runs/active/ig_examples_200pos_run3_operational/` is untouched.
