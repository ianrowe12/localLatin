# Cosine inflation in the 200-pair attribution: root cause and fix

**Investigated**: 2026-04-26
**Branch**: `run-1/agent-1-cosine-investigation`
**Trigger**: Professor flagged that the 200-random-pair attribution PDF shows
even gold-NEGATIVE pairs at cosine 0.93–0.99 at the attribution layer, while
the live retrieval demo shows gold-POSITIVES at 0.84–0.92.

## TL;DR

**There is a real, fixable bug.** The PCs file at
`runs/phase12_release/pcs/bowphs_LaTa/layer4_pcs.npz` was extracted from old
phase12c artifacts that were fit on a fundamentally different embedding
distribution (token-keep-filtered) than what `run_attribution_metrics.py`
actually pools at metrics time (no token filter). Using those stale PCs to
"clean" freshly-pooled vectors *inflates* cosines instead of cleaning them:
`abtt full_cos_mean = 0.985` is HIGHER than `baseline = 0.926` for LaTa.
Empirically, the old PC1 has cosine ~0.10 with a freshly-fit PC1 on the same
dataset, and the old mean_vec differs by L2 norm ~5700.

**Fix shipped on this branch**:
1. `scripts/ig/sample_random_test_pairs.py`: `FEATURED_MODELS` D values set
   to **D=10 universally** (was LaTa=2, PhilTa=10, LaBSE=1, Qwen3-0.6B=3).
   Per the 2026-03-31 meeting decision, D=10 universal matches `abtt_optimal`
   across all models.
2. `scripts/ig/refit_pcs_for_attribution.py`: NEW helper that refits
   `mean_vec` and top-D PCs from canon train embeddings using the same
   mean-pool path that `run_attribution_metrics.forward_pooled` produces.
   Run this once before the next 200-pair experiment.

**Empirical validation on the same canon test split the 200-pair pipeline
samples from** (100 same-folder + 100 different-folder pairs at LaTa L4):

| variant                                                     | full_cos_mean | pos mean ± std    | neg mean ± std    |
|-------------------------------------------------------------|---------------|-------------------|-------------------|
| baseline (raw mean-pool, anisotropic)                       | 0.917         | 0.940 ± 0.084     | 0.894 ± 0.253     |
| OLD ABTT D=2 PCs from `runs/phase12_release/...`           | **0.984**     | **0.999 ± 0.002** | **0.968 ± 0.199** |
| NEW ABTT D=2 PCs refit on canon train                       | 0.198         | 0.395 ± 0.333     | 0.001 ± 0.301     |
| NEW ABTT D=10 PCs refit on canon train                      | 0.295         | **0.583 ± 0.232** | **0.008 ± 0.098** |

The OLD-PCs row matches the broken numbers in `summary.csv` (LaTa abtt
0.985, baseline 0.926). The NEW-PCs D=10 row shows the fix recovers proper
positive/negative separation: positives at 0.58, negatives essentially zero.

**Open methodology questions for Monday** that I am NOT silently deciding
(see "What this branch does NOT change" below): SIF on/off in attribution,
attribution layer choice, dataset choice (canon vs canon_labelled), and
token-keep-lookup consistency between the IG NPZ generator and the metrics
script.

## 1. What the professor saw

`runs/active/ig_examples_200pair/attribution_metrics/summary.csv` reports
one `full_cos_mean` per (model × variant), shared across all 9 attribution
methods (since attribution algorithm choice doesn't change the cosine
target):

| Model        | variant              | full_cos_mean |
|--------------|----------------------|---------------|
| LaTa  L4     | baseline             | **0.9258**    |
| LaTa  L4     | abtt (current D=2)   | **0.9851**    |
| PhilTa L6    | baseline             | 0.9502        |
| PhilTa L6    | abtt (current D=10)  | 0.9380        |

The means aggregate over 100 positives + 100 negatives. The professor's
"negatives at 0.93–0.99" observation is faithful to the data; per-pair JSONs
show concrete pairs with `full_cos = 0.998` despite being gold-negatives
(verified: `runs/active/ig_examples_200pair/attribution_metrics/bowphs_LaTa/
example001_pair_example.json` has baseline 0.999, abtt 0.999 for a
different-folder pair).

For LaTa, **abtt is *higher* than baseline** (0.985 > 0.926). ABTT is meant
to deisotropize, pulling random-pair cosines down. Something is wrong.

## 2. What is NOT the bug

The investigation ruled out the original hypothesis list:

| Hypothesis                                                         | Verdict | Evidence                                                                                                      |
|--------------------------------------------------------------------|---------|---------------------------------------------------------------------------------------------------------------|
| Negative labels are not actually negatives                         | NO      | All 5 sampled negative pairs verified as different `folder_id` directories under `data/canon/`.              |
| Pooling mismatch (model side mean-pool, partner cached SIF-pool)   | NO      | `run_attribution_metrics.py:99-110` re-runs the model and mean-pools both query and candidate identically.    |
| ABTT applied asymmetrically                                        | NO      | `run_attribution_metrics.py:178-206` applies `abtt_clean` to BOTH `c_pooled` and `q_pooled` with identical PCs/mean_vec. |
| Layer mismatch between query and candidate                         | NO      | Same `layer` arg is read from each NPZ row and used for both forward passes.                                  |
| Cosine math wrong                                                  | NO      | Direct re-computation from the cached normalized embedding store reproduces summary.csv values to 6 decimals (e.g., 0.998707). |

The cosine-similarity computation itself is correct. The data flow is
internally consistent. So why are the numbers wrong?

## 3. Root cause: the pre-baked PCs are fit on a different distribution

`run_attribution_metrics.py:174-175` loads the PCs from
`pcs = torch.from_numpy(data["pcs"])` (each NPZ stores its own PCs), and the
NPZ generator `scripts/_archive/run_phase12e_pair_explanations.py:222-229`
reads the PCs from `runs/phase12_release/pcs/<slug>/layer{L}_pcs.npz` and
slices `pc_data["pcs"][:d_value]`.

Those PC files were created by `scripts/ig/extract_pcs_from_npzs.py:21-26`,
which **copies PCs out of old phase12c NPZs**. Those phase12c NPZs were
generated by `run_phase12c_retrieval_attribution.py` with token-keep-lookup
filtering (the `pool_hidden` function masks out punctuation / special
tokens before averaging — see
`scripts/_archive/run_phase12c_retrieval_attribution.py:127-134`).

But `run_attribution_metrics.py` does NOT use a token filter. Its
`forward_pooled` (line 99-110) is plain attention-mask-weighted mean-pool:

```python
hidden = out.hidden_states[layer].float()  # (1, seq, hidden)
mask = attention_mask.float()              # (1, seq)
pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
```

So:
- The mean_vec stored in the PC file represents the centroid of
  *token-keep-filtered* embeddings.
- The PCs represent variance directions in *token-keep-filtered* space.
- The vectors being cleaned (`forward_pooled` output) live in *no-filter*
  space — a totally different distribution.

`abtt_clean(vec, mean_vec, pcs)` does `(vec - mean_vec) - (vec - mean_vec) @ pcs.T @ pcs`.
When `mean_vec` is wildly off-distribution, the centering step pushes the
vector AWAY from the actual data centroid rather than toward zero, and the
projection step removes directions that aren't the dominant variance axes
of the actual data. The result lives even closer to the anisotropic cone
than the raw input.

### Numerical proof

Refit fresh PCs from canon train embeddings (same data path as the live
metrics computation: `runs/active/encoder_bases/bowphs_LaTa/hidden_mean/
hidden_layer4_embeddings.npy`, filtered to phase9 train), and compare to
the in-tree `runs/phase12_release/pcs/bowphs_LaTa/layer4_pcs.npz`:

```
Old vs new mean_vec L2 diff:  5728.6
Old PC1 vs new PC1 cos:       0.099
```

A mean_vec L2 diff of ~5700 in a 768-dim space and a PC1 alignment of
0.10 (essentially orthogonal) confirm the old PCs are unrelated to the
distribution they're being used to clean.

### Why D=2 made it especially bad for LaTa

PhilTa already had D=10 in `FEATURED_MODELS`, so the slice
`pc_data["pcs"][:10]` retained 10 PCs (and the off-distribution PC effect
partly self-cancelled). LaTa had D=2: only the top 2 (wrong) PCs were
removed, leaving the dominant anisotropic axis untouched while perturbing
the centering. Result: cosines went UP after cleaning.

## 4. The fix on this branch

### 4.1 D=10 universal in `FEATURED_MODELS`

`scripts/ig/sample_random_test_pairs.py` updated so all four featured
models use D=10:

```diff
 "bowphs/LaTa":   { ..., "D": 10 (was 2) }
 "bowphs/PhilTa": { ..., "D": 10 (unchanged) }
 "sentence-transformers/LaBSE":   { ..., "D": 10 (was 1) }
 "Qwen/Qwen3-Embedding-0.6B":     { ..., "D": 10 (was 3) }
```

This alone is necessary but **not sufficient**, because the existing PC
files have only 2 rows for LaTa (and 1 for LaBSE, 3 for Qwen3-0.6B).
Slicing `[:10]` against a (2, hidden) array silently returns (2, hidden),
so without refitting the underlying PCs, D=10 in config has no effect.

### 4.2 New script: `scripts/ig/refit_pcs_for_attribution.py`

Refits `mean_vec` and the top 10 PCs from canon train embeddings using the
same mean-pool path (`hidden_mean`, no token filter) that `forward_pooled`
produces. Defaults match the 200-pair pipeline (canon dataset, phase9
split, encoder_bases cache). Validated on dry-run:

```
[bowphs_LaTa]  fitting D=10 on layer 4 train ((639, 768))
[bowphs_LaTa]  wrote .../bowphs_LaTa/layer4_pcs.npz  pcs.shape=(10, 768)
[bowphs_PhilTa] fitting D=10 on layer 6 train ((639, 768))
[bowphs_PhilTa] wrote .../bowphs_PhilTa/layer6_pcs.npz  pcs.shape=(10, 768)
```

**Run this once before the next 200-pair experiment**:
```
python scripts/ig/refit_pcs_for_attribution.py
```

This overwrites `runs/phase12_release/pcs/<slug>/layer{L}_pcs.npz`. The
existing files would get stomped — they were broken anyway. Backup or git-stash
if reproduction of the broken numbers is needed for comparison.

### 4.3 Validated end-to-end

Same 100 pos + 100 neg sampling logic the pipeline uses, on canon test:

| variant                            | full_cos | pos mean | neg mean |
|------------------------------------|----------|----------|----------|
| baseline (raw)                     | 0.917    | 0.940    | 0.894    |
| OLD pre-baked PCs (current bug)    | 0.984    | 0.999    | 0.968    |
| NEW refit, D=10                    | 0.295    | 0.583    | 0.008    |

Post-fix the negatives drop to ~0 (from 0.97), positives drop to 0.58
(from 0.999), and the variant separation is real for the first time.

## 5. What this branch does NOT change

Per the user's standing instruction ("If the investigation reveals a
fundamental design problem ... STOP and flag it loudly in the report — do
not silently push a half-fix"), the following remain open and should be
discussed Monday rather than patched unilaterally:

### 5.1 Token-keep-lookup inconsistency between IG NPZ generator and metrics

`scripts/_archive/run_phase12e_pair_explanations.py:127-134` (the IG NPZ
generator) uses `pool_hidden(hidden, enc, token_keep_lookup)` — applies a
token filter masking out punctuation / special tokens.

`scripts/ig/run_attribution_metrics.py:99-110` (the metrics computation)
uses plain `attention_mask` weighting only — no token filter.

The IG attribution masks are computed against one cosine target; the
sufficiency / comprehensiveness / compactness metrics are computed
against a different cosine target. This is an open inconsistency. Agent
1.2's pipeline audit may have additional findings here. Refitting PCs at
D=10 *for the metrics computation* (what this branch does) makes the
metrics-side cleaning correct, but doesn't reconcile the two sides.

### 5.2 Dataset choice: canon (1278) vs canon_labelled (1705)

The 200-pair sampler reads from `runs/phase9/phase9_split.csv` (canon,
1278 short fragments under `data/canon/`). The live retrieval pipeline
(`scripts/resubmit/evaluate_vectors.py`) operates on canon_labelled
(1705 files under `data/canon_labelled/`) via
`runs/active/resubmit/data/phase_resubmit_split.csv`.

Whether the attribution should be on the same dataset as the headline
retrieval claim is a design decision. If yes, the sampler, embedding
cache path (`bases_root`), PC refit dataset, and split CSV all need to
move to canon_labelled. The refit script supports this via
`--bases_root runs/active/resubmit_bases/phase9_bases --pooling
hidden_mean_tokempty --split_csv runs/active/resubmit/data/phase_resubmit_split.csv`.

### 5.3 Attribution layer for LaTa

LaTa L4 is the anisotropy-dip layer. Per
`runs/active/resubmit/results/perlayer_tables_v2/taskA_main.csv`:
LaTa L1 AUCROC=0.969, L11=0.973, L4=0.971 — all close on AUCROC but the
ranking under Assignment Accuracy (the canonical resubmit metric) may
differ. Whether to attribute at L4 (illustrate-the-dip story) or at the
canonical retrieval layer (illustrate-what-actually-works story) is a
narrative choice for the Monday meeting.

### 5.4 SIF on/off

Per the user's confirmation, the professor's current direction is
**ABTT only, no SIF**. This branch respects that constraint. If that
changes, the attribution forward path needs a `pool_sif` variant (the
retrieval pipeline already has one at `src/sif_abtt.py:185-210`).

## 6. ⚠️ Post-merge action item: Agent 1.3's positives sampler duplicates FEATURED_MODELS

Agent 1.3 (`run-1/agent-3-positives-sampler` branch) created a NEW file
`scripts/ig/sample_positive_test_pairs.py` for the 200-positives experiment
and DUPLICATED the `FEATURED_MODELS` dict inline rather than importing it.
Their copy was made before this investigation and therefore still has
**LaTa D=2, LaBSE D=1, Qwen3-0.6B D=3**.

After Run 1 merges all three branches, the merge agent or Agent 2.1 must
update `scripts/ig/sample_positive_test_pairs.py` so its inline
`FEATURED_MODELS` dict matches the canonical D=10 universal values from
`scripts/ig/sample_random_test_pairs.py`. Otherwise the new positives-only
GPU run will still slice `pcs[:2]` for LaTa and reproduce the inflation,
even though the underlying PC files have 10 PCs available after the refit.

Concrete diff to apply post-merge to `scripts/ig/sample_positive_test_pairs.py`:

```diff
   "bowphs/LaTa":   { ..., "D": 2,  ... }      →  "D": 10
   "sentence-transformers/LaBSE": { ..., "D": 1,  ... }  →  "D": 10
   "Qwen/Qwen3-Embedding-0.6B":   { ..., "D": 3,  ... }  →  "D": 10
```

A cleaner long-term fix is to have `sample_positive_test_pairs.py` import
from `sample_random_test_pairs.py` rather than duplicate. Out of scope for
this branch (touches Agent 1.3's file).

## 7. What to run next

1. **Refit PCs** (CPU, seconds):
   ```
   cd /projects/beto/irowerojas/localLatin
   conda run -n localLatin python scripts/ig/refit_pcs_for_attribution.py
   ```
   This produces fresh D=10 PCs at the same target paths the IG NPZ
   generator and metrics script read from.

2. **Sanity-check** (optional, CPU): run the comparison block from §4.3
   above to confirm post-fix `full_cos_mean ≈ 0.30` for D=10 on canon test
   pairs.

3. **GPU rerun** of the 200-pair pipeline against the new PCs. Per the
   orchestra plan, Run 2 (`Agent 2.1`) handles this. The new
   `slurm/ig/run_attribution_200pos.sbatch` (Agent 1.3's deliverable)
   is the entrypoint. Expected behavior post-fix: abtt `full_cos_mean`
   should drop from 0.985 to ~0.30 (LaTa) with positives ≈ 0.58 and
   negatives ≈ 0.

4. **Audit cross-check**: Read `docs/analyses/attribution_pipeline_audit.md`
   (Agent 1.2's deliverable) for any additional findings, especially around
   the token-keep-lookup inconsistency in §5.1.

## 8. Files referenced

Code:
- `scripts/ig/sample_random_test_pairs.py:27-65` — `FEATURED_MODELS` dict
  (the patched D values land here)
- `scripts/ig/run_attribution_metrics.py:99-129` — `forward_pooled`,
  `abtt_clean`, `cos`
- `scripts/ig/run_attribution_metrics.py:174-206` — main per-variant cosine
  computation
- `scripts/ig/extract_pcs_from_npzs.py:21-26` — **DEPRECATED**: legacy PC
  extraction that reads from old phase12c NPZs and is the source of the
  stale PC files. Replaced by
  `scripts/ig/refit_pcs_for_attribution.py`. Do not call
  `extract_pcs_from_npzs.py` in any new sbatch chain.
- `scripts/_archive/run_phase12e_pair_explanations.py:127-134` —
  token-keep-lookup pool_hidden (cause of distribution mismatch)
- `scripts/_archive/run_phase12e_pair_explanations.py:222-229` — IG NPZ
  generator's PC loading + slice
- `src/sif_abtt.py:125-139` — `remove_top_components` (used by the new
  refit script)
- `scripts/resubmit/evaluate_vectors.py:263-329` — canonical retrieval
  path (for comparison)

Data:
- `runs/active/ig_examples_200pair/attribution_metrics/summary.csv` — the
  broken aggregate numbers
- `runs/active/ig_examples_200pair/random200_examples.csv` — the source
  pair list (200 pairs, canon paths, phase9 split)
- `runs/active/encoder_bases/bowphs_LaTa/hidden_mean/hidden_layer4_embeddings.npy`
  — canonical canon mean-pool cache
- `runs/phase9/phase9_split.csv` — canon train/test split
- `runs/phase12_release/pcs/<slug>/layer{L}_pcs.npz` — the broken PC files
  this branch's refit script overwrites
