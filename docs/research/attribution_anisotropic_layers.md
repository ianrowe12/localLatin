# Attribution at the most anisotropic layers

Issue #227. Companion to `docs/research/attribution_v1_resample.md` (the run of
record) and `docs/research/attribution_metrics_decision.md` (the metric
selection, reused unchanged here).

**Status: complete. Reported whatever it showed; this was one predeclared run,
not a search.** Rank faithfulness (`rho_LOO`) favours ABTT in all six cells at
the most anisotropic layers, against five wins and a tie at the operational
layers. Deletion faithfulness (`DelAUC gap`) gets weaker, not stronger: two
wins, two ties, two losses, against four wins and two losses. At these layers
baseline integrated gradients for LaTa and mT5-base do no better than a shuffle
of their own scores on `rho_LOO`, so this layer set fails the shuffled-attribution
control in two baseline cells.

## Why this run exists

James Wong, 22 September 2026: the predeclared attribution rule (earliest layer
within 0.5 points of the best training-set DirAcc@1 under ABTT) selects LaTa 7,
PhilTa 1 and mT5-base 1. On PhilTa and mT5-base layer 1 the baseline already
separates equivalent pairs (test AUROC 0.939 and 0.822), so only LaTa's
attribution is measured inside the collapse, and the modest attribution gains
might be a property of the layer choice. Of the two options he offered, Ian
approved the first on 2026-09-22: a second attribution row at the most
anisotropic layers, a label-free layer set the paper already names as mechanism
checks, reported whatever it shows. The second option (flip the tie-break to
"latest layer within x points") was not run: it would change a predeclared rule
after its results are known.

Since the reframe of 2026-09-27 (`docs/research/reframe_handoff_20260927.md`)
attribution is an appendix (`app:attribution`) and no longer a contribution, so
this result belongs in that appendix and nowhere in the main text.

## The layer set

"Most anisotropic" is the argmax of the top-PC (PC1) variance share of the
baseline mean-pooled vectors, `diagnostic_layer_by_max_pc1_dominance` in
`runs/active/resubmit/layer_diagnostics/layer_rule_candidates.csv`, the same
layers as `tab:layer_diagnostics_main` and the Setup paragraph of
`app:attribution` ("the most anisotropic layers on the test split, LaTa 8,
PhilTa 6, and mT5-base 5"). Confirmed from `geometry_per_layer.csv`:

| Model | Operational layer | Most anisotropic (test split) | PC1 share there | Argmax on the train split |
|---|---|---|---|---|
| LaTa | 7 | **8** | 0.956 (layer 7: 0.945) | 4 (0.952; layer 8: 0.950) |
| PhilTa | 1 | **6** | 0.862 | 6 |
| mT5-base | 1 | **5** | 1.000 | 5 |

Two things a reader should know, both already acknowledged in the paper's
layer-diagnostics caption. The rule reads the test split, which is legitimate
for a label-free statistic but differs from the train-only convention elsewhere.
And the argmax sits on a plateau: LaTa's PC1 share is 0.93 to 0.96 over layers 3
to 11 (so layers 7 and 8 are geometrically near-identical), and mT5-base reads
1.000 at every layer from 5 to 11 (5 is the first). The layer set was fixed
before any attribution was computed at it.

## Provenance

| Item | Value |
|---|---|
| Run directory | `runs/active/ig_examples_200pos_v1_aniso/` (gitignored, not committed) |
| Reference run (unchanged) | `runs/active/ig_examples_200pos_v1/`, metrics `attribution_metrics_draws20/` (`RUN_OF_RECORD`, `METRICS_DIR_OF_RECORD`) |
| Layers | LaTa 8, PhilTa 6, mT5-base 5; `D=10` |
| Pairs | the v1 sample: `sample_positive_test_pairs.py`, seed `20260426`, rerun with `--layer_overrides`; the examples CSV differs from v1's in the `layer` column only |
| `pair_manifest_sha256` | `4fcdaa2be99ebcb279c16d47ce2af158411727df27c5b86414404780a43591c8` (identical to v1; `pair_manifest.tsv` byte-identical) |
| `positive200_examples.csv` sha256 | `7007ae9a5e59f7946bf8af31d9414037245edd217a43bb816990be9ca38c56d7` |
| Cleaners | `pcs/<slug>/layer{8,6,5}_pcs.npz`, refit train-only by `refit_pcs_for_attribution.py` from `resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/` through `AlignmentResolver` (3 caches verified-permuted, 17 rows moved, as for v1) |
| Code | worktree at `858f013` (`CODE_ROOT`), pipeline scripts identical to `main` apart from the new `--layer_overrides` flag of the manifest check |

### Compute

| Stage | Job | Partition | Elapsed | Reserved | State |
|---|---|---|---|---|---|
| CPU smoke, 3 pairs (my argument error in the smoke script, not the pipeline) | `22321796` | `cpu` | 00:02:55 | 00:45:00 | FAILED |
| CPU smoke, 3 pairs, whole chain incl. metrics | `22321840` | `cpu` | 00:06:25 | 00:45:00 | COMPLETED |
| IG + MaRC + persistence + manifest | `22321913` | `gpuA100x4` | **01:08:45** | **01:25:00** | COMPLETED |
| Metrics, hidden backend, 20 draws | `22321915` | `cpu` | 00:06:02 | 00:30:00 | COMPLETED |

**GPU cost: 1.42 GPU-hours reserved (1.15 elapsed), one GPU job, as approved.**
The repo's budget rule treats the reservation as the charge. The 01:25:00 cap
is the v1 reservation, kept because MaRC's early stopping is layer-dependent:
at layer 7 LaTa used 69% of its MaRC step budget, and at 100% its stream would
have run about 77 minutes. At the new layers the step-budget use was 69%
(LaTa), 75% (PhilTa) and 88% (mT5-base); the mT5-base stream was the slowest
(IG 13 min, MaRC 54 min).

### Completeness checks (all pass)

* Pair digest checked inside the job before any GPU work (`--expect`).
* 200/200 IG NPZs and 200/200 MaRC sidecars per model, 0 failures in each stream;
  `verify_attribution_artifacts.py`: "600 pairs, every artifact carries ig,
  retrieval_mark"; manifest `errors: []` with the layer overrides recorded.
* Metrics: 600 per-pair JSONs, 54 summary rows, no errors in the log.
* Stored hidden states are at the new layer: on the smoke pairs, filtered
  pooling of each NPZ's `query_hidden` against the cache row has cosine
  1.0000000 at the new layer and 0.997 / 0.196 / -0.258 at the old one (LaTa /
  PhilTa / mT5-base).
* The refit procedure reproduces v1's layer-7/1/1 PC files bit for bit, so the
  only change in the cleaners is the layer.
* The v1 run of record was not written to: its tracked summaries and examples
  CSV compare byte-identical, and no file under it is newer than this run's
  start.

## What changed and what did not

Changed: the layer (IG target and stored hidden states, MaRC optimiser, PCs and
mean vector, hence every hidden-backend metric). Nothing else. Same 600 pairs;
IG 40 steps; MaRC 200 steps, lr 0.1, lambda 0.01, gamma 0.001, init logit
2.197, early stop 0.01 after 50 steps, seed 0; token filter `tokenizer_empty`;
`max_length` 256; `D=10` fit on the 847 training passages; hidden backend with
20 random-order draws and the same seed stream; the same 2-SE tie rule on
paired per-pair differences. The unused `tau`/`baseline_tau`/`abtt_tau`
columns of the examples CSV keep v1's values (nothing downstream reads them).

## The retrieval score being explained

Test-split retrieval from `runs/active/resubmit/results/phase_resubmit_results.csv`
(mean pooling over `hidden_mean_tokempty`, `abtt_fixed` is `D=10`), and geometry
from `geometry_per_layer.csv` (test split):

| Model, layer | Set | AUROC base -> ABTT | DirAcc@1 base -> ABTT | PC1 share base -> ABTT | Eff. rank base -> ABTT |
|---|---|---|---|---|---|
| LaTa 7 | op. | 0.498 -> 0.962 | 30.2 -> 86.8 | 0.945 -> 0.036 | 1.40 -> 165.1 |
| LaTa 8 | anis. | 0.502 -> 0.965 | 30.2 -> 86.1 | 0.956 -> 0.036 | 1.34 -> 168.4 |
| PhilTa 1 | op. | 0.939 -> 0.977 | 69.3 -> 88.3 | 0.062 -> 0.027 | 136.0 -> 202.5 |
| PhilTa 6 | anis. | 0.541 -> 0.981 | 31.0 -> 84.5 | 0.862 -> 0.039 | 1.83 -> 155.0 |
| mT5-base 1 | op. | 0.822 -> 0.979 | 45.1 -> 88.5 | 0.237 -> 0.025 | 57.0 -> 210.4 |
| mT5-base 5 | anis. | 0.654 -> 0.978 | 35.2 -> 75.6 | 1.000 -> 0.044 | 1.00 -> 151.7 |

All three new layers are inside the collapse (baseline AUROC 0.50 to 0.65, top
PC share 0.86 to 1.00) and ABTT repairs each to 0.965 to 0.981. mT5-base layer 5
is the paper's mT5-base AUROC minimum (0.654). Mean full-pair cosine over the
200 attribution pairs: baseline 0.071 / 0.634 / 0.473, ABTT 0.585 / 0.585 /
0.504 (LaTa / PhilTa / mT5-base).

## Results

### Main-table metrics, paired ABTT minus baseline

Mean +/- paired SE (ratio in parentheses) over pairs valid under both variants,
from each run's per-pair cache (`paired_cell_stats`). Verdict by the main
table's rule: a tie if |ratio| < 2.

| Cell | rho_LOO, op. layer | rho_LOO, anis. layer | DelAUC gap, op. layer | DelAUC gap, anis. layer |
|---|--:|--:|--:|--:|
| LaTa/IG | +0.367 +/- 0.023 (16.0) | +0.375 +/- 0.022 (17.0) | -0.651 +/- 0.077 (-8.5) loss | -0.711 +/- 0.065 (-10.9) loss |
| LaTa/MaRC | +0.455 +/- 0.027 (17.0) | +0.427 +/- 0.026 (16.5) | -0.177 +/- 0.084 (-2.1) loss | -0.212 +/- 0.065 (-3.2) loss |
| PhilTa/IG | +0.285 +/- 0.016 (17.5) | +0.213 +/- 0.020 (10.8) | +0.290 +/- 0.010 (28.4) | +0.089 +/- 0.106 (0.8) **tie** |
| PhilTa/MaRC | -0.024 +/- 0.016 (-1.5) **tie** | +0.377 +/- 0.025 (15.3) | +0.112 +/- 0.014 (8.0) | +0.241 +/- 0.068 (3.6) |
| mT5-base/IG | +0.543 +/- 0.024 (22.5) | +0.298 +/- 0.016 (18.3) | +0.499 +/- 0.021 (24.0) | -0.037 +/- 0.061 (-0.6) **tie** |
| mT5-base/MaRC | +0.247 +/- 0.024 (10.3) | +0.372 +/- 0.021 (17.8) | +0.421 +/- 0.022 (19.4) | +0.341 +/- 0.046 (7.5) |
| **win / tie / loss** | **5 / 1 / 0** | **6 / 0 / 0** | **4 / 0 / 2** | **2 / 2 / 2** |

Per-variant means (the numbers the table prints) at the anisotropic layers,
baseline -> ABTT: `rho_LOO` LaTa -0.018 -> 0.357 (IG), 0.108 -> 0.535 (MaRC);
PhilTa 0.065 -> 0.278, 0.107 -> 0.484; mT5-base -0.026 -> 0.272, 0.041 -> 0.413.
`DelAUC gap` LaTa 0.913 -> 0.182, 0.567 -> 0.336; PhilTa 0.052 -> 0.140, 0.044
-> 0.285; mT5-base 0.262 -> 0.230, 0.028 -> 0.370. By sign of the per-variant
means (the table's boldface), ABTT wins 6/6 and 3/6; the paired rule turns two
of those sign verdicts into ties (PhilTa/IG, mT5-base/IG).

### Secondary metrics

Wins by the sign of the per-variant means, as the secondary table counts them
(`scripts/ig/compare_attribution_runs.py` against the draws20 summaries):

| Metric | Dir | op. layers | anis. layers | cells that flipped |
|---|---|---|---|---|
| tau_LOO | higher | 5/6 | 6/6 | PhilTa/MaRC b->A |
| InsAUC gap | higher | 5/6 | 3/6 | PhilTa/IG A->b, mT5-base/IG A->b |
| AOPC-Suff | higher | 4/6 | 2/6 | PhilTa/IG A->b, mT5-base/IG A->b |
| AOPC-Comp | higher | 4/6 | 3/6 | mT5-base/IG A->b |
| Suff@25% | higher | 4/6 | 3/6 | PhilTa/IG A->b, PhilTa/MaRC b->A, mT5-base/IG A->b |
| Comp@25% | higher | 4/6 | 2/6 | PhilTa/IG A->b, mT5-base/IG A->b |
| MinFrac@0.80 | lower | 2/6 | 1/6 | LaTa/MaRC A->b, PhilTa/IG A->b, PhilTa/MaRC b->A |

Paired, under the 2-SE rule: `tau_LOO` 5/1/0 -> 6/0/0 (ratios 10.6 to 18.3 at
the new layers); `InsAUC gap` 4/1/1 -> 1/4/1; `Comp@25%` 4/0/2 -> 1/2/3. Every
threshold-free rank metric improves; every erasure-curve and ERASER metric
weakens, mostly through the IG cells of PhilTa and mT5-base turning into ties or
losses.

### Shuffled-attribution control (six metrics, twelve cells)

Real-minus-shuffled gap per cell (`rand_<metric>_gap_mean`); a cell passes if
the gap is positive, as criterion 5 of the decision memo defines it.

| Metric | op. layers positive | anis. layers positive | anis. failures |
|---|---|---|---|
| rho_LOO | 12/12 | **10/12** | LaTa/IG baseline (-0.014, -0.8 SE), mT5-base/IG baseline (-0.026, -1.9 SE) |
| tau_LOO | 12/12 | **10/12** | LaTa/IG baseline (-0.005, -0.4 SE), mT5-base/IG baseline (-0.019, -2.1 SE) |
| DelAUC gap | 12/12 | 12/12 | none |
| AOPC-Comp | 12/12 | 12/12 | none |
| InsAUC gap | 11/12 | 12/12 | none |
| AOPC-Suff | 11/12 | 12/12 | none |

(`AOPC-Comp` equals `DelAUC gap` and `AOPC-Suff` equals `InsAUC gap` per cell,
as on v1.) All six ABTT cells pass every metric, by 6 to 40 standard errors. The
failures are baseline IG cells at the two layers where the baseline retrieval
score is nearest chance: there baseline IG ranks tokens no better than a
permutation of its own scores. Several passing baseline cells are also within
two standard errors of zero (DelAUC gap PhilTa baseline IG +0.4 SE and MaRC +0.7
SE, mT5-base baseline MaRC +0.9 SE). Read literally, criterion 5 ("in every
cell") would disqualify `rho_LOO` as a headline column at these layers; the
cells that fail are baseline cells, which is the claim under test rather than a
fault of the measurement, but the paper must not say the control passes here.

### Pair validity (full-query cosine floor 0.05)

`rho_LOO` is defined for all 200 pairs in every cell at both layer sets.
DelAUC columns, pairs above the floor, baseline / ABTT:

| Model | op. layers | anis. layers |
|---|---|---|
| LaTa | 199 / 195 | 197 / 195 |
| PhilTa | 200 / 199 | **191** / 200 |
| mT5-base | 200 / 198 | 198 / 196 |

PhilTa layer 6 loses nine baseline pairs to the floor, the largest loss in
either run.

## Honest reading

**What it supports.** Inside the collapse, ABTT's rank-faithfulness gain is
unanimous: 6/6 on `rho_LOO` and on `tau_LOO`, every cell at 10.8 standard errors
or more. The one operational-layer tie, PhilTa/MaRC at layer 1 where the
baseline already retrieves (AUROC 0.939), becomes a +0.377 win (15.3 SE) at
layer 6 where it does not. That is what the account in the issue predicts for
the rank metric: when the baseline score is healthy there is little for ABTT to
fix, and when it has collapsed the baseline explanation collapses with it
(baseline `rho_LOO` -0.026 to 0.108 at the new layers, two cells no better than
shuffled). LaTa, whose layers 7 and 8 are geometrically near-identical, gives
the same answer at both to within about one standard error in every main-table
cell, a replication check the design did not plan but that it passes.

**What it does not support.**

1. *Larger gains inside the collapse.* The mean `rho_LOO` gain is 0.344 at the
   anisotropic layers against 0.312 at the operational ones, and it is smaller
   in two cells (PhilTa/IG +0.213 vs +0.285, mT5-base/IG +0.298 vs +0.543). The
   gains hold their size because the baseline falls, not because ABTT
   explanations get better: post-ABTT `rho_LOO` is lower at the anisotropic
   layers for PhilTa/IG (0.278 vs 0.614) and mT5-base/IG (0.272 vs 0.686).
2. *Deletion faithfulness.* It gets worse: 2/2/2 against 4/0/2. Both LaTa cells
   still favour the baseline, as they do at layer 7, and PhilTa/IG and
   mT5-base/IG become ties. The erasure-curve and ERASER metrics all weaken the
   same way.
3. *So the claim "attribution gains are modest because layer 1 sits outside the
   collapse" does not survive as a general statement.* It survives for the rank
   metric in a narrow form (the one tie disappears and the gain is unanimous),
   and it fails for deletion faithfulness, where moving into the collapse
   removes two wins. The modest, mixed picture of the appendix is not an
   artefact of the operational layer choice.

**Limits.** One sample of 200 pairs per model, one erasure operator (the
representation-level one; masking acts on cached layer-L states), and a layer
set that is the argmax of a flat plateau read on the test split. The layer set
was fixed before this run and is not tuned, but a neighbouring layer on the same
plateau could move individual cells.

## Recommendation for the paper

Keep the run of record at the operational layers as the attribution result in
`app:attribution`. Add this run as a second appendix table,
`tables/attribution_metrics_aniso.tex` (`tab:attribution_metrics_aniso`),
`\input` next to `tab:attribution_metrics_main`, and one or two sentences in the
Results subsection of `app:attribution`. Nothing goes in the main text, since
attribution is no longer a contribution.

Proposed sentences (for Ian, who is editing `acl_latex.tex`):

> As a mechanism check we repeated the analysis on the same pairs at each
> model's most anisotropic layer (LaTa 8, PhilTa 6, mT5-base 5;
> Table~\ref{tab:attribution_metrics_aniso}), where baseline AUROC is 0.502,
> 0.541 and 0.654 and ABTT restores 0.965, 0.981 and 0.978. ABTT then improves
> $\rho_{\mathrm{LOO}}$ in all six cells (10.8 to 18.3 standard errors), mainly
> because the baseline explanation collapses with the score (baseline integrated
> gradients for LaTa and mT5-base no longer beat a shuffle of their own
> scores), while chance-corrected deletion faithfulness improves in only two
> cells, ties in two, and still favours the baseline in both LaTa cells.

If one sentence is preferred, drop the parenthetical on the shuffle control
but keep the deletion clause: the deletion result is what stops the check from
reading as a stronger claim than the evidence carries.

Two follow-ups for whoever inserts it: the Setup paragraph already names these
layers as mechanism checks, so no new definition is needed; and the sentence
"a metric qualifies for the headline table only if the real attribution beats
the permutation mean in every cell" is about the operational layers, where it
still holds.

## Reproduce

```bash
# 1. Pairs: v1's sample at the new layers (CPU, seconds)
PYTHONPATH=scripts/ig python scripts/ig/sample_positive_test_pairs.py \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --repo_root /projects/beto/irowerojas/localLatin \
  --out_csv runs/active/ig_examples_200pos_v1_aniso/positive200_examples.csv \
  --n_per_model 200 --seed 20260426 \
  --models bowphs/LaTa bowphs/PhilTa google/mt5-base \
  --layer_overrides bowphs/LaTa=8 bowphs/PhilTa=6 google/mt5-base=5
python scripts/ig/pair_manifest_digest.py \
  --examples_csv runs/active/ig_examples_200pos_v1_aniso/positive200_examples.csv \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --expect 4fcdaa2be99ebcb279c16d47ce2af158411727df27c5b86414404780a43591c8

# 2. Cleaners at the new layers (CPU, seconds)
python scripts/ig/refit_pcs_for_attribution.py \
  --slugs bowphs_LaTa bowphs_PhilTa google_mt5-base \
  --bases_root runs/active/resubmit_bases/phase9_bases \
  --pooling hidden_mean_tokempty \
  --split_csv runs/active/resubmit/data/phase_resubmit_split.csv \
  --pc_root runs/active/ig_examples_200pos_v1_aniso/pcs --d 10 \
  --layer_overrides bowphs_LaTa=8 bowphs_PhilTa=6 google_mt5-base=5

# 3. IG + MaRC + persistence (GPU, one job), then metrics (CPU)
jid=$(sbatch --parsable slurm/ig/run_attribution_200pos_v1_aniso.sbatch)
sbatch --dependency=afterok:$jid slurm/ig/attribution_metrics_200pos_v1_aniso_draws20.sbatch

# 4. Table (needs both runs' gitignored v2_hidden caches)
python scripts/ig/build_aniso_attribution_table.py
python scripts/ig/compare_attribution_runs.py \
  --old_summary runs/active/ig_examples_200pos_v1/attribution_metrics_draws20/summary_v2.csv \
  --new_summary runs/active/ig_examples_200pos_v1_aniso/attribution_metrics_draws20/summary_v2.csv
```

`tests/test_aniso_attribution_table.py` regenerates the committed table byte for
byte when both runs' summaries and per-pair caches are present, and skips
otherwise (CI).
