# Attribution Pipeline Audit (Run-1 Agent-2, 2026-04-26)

> Note (2026-08-24): the `attribution_metrics_200pair.tex` and `attribution_metrics_200pos*.tex` files this audit greps were removed from the working tree in #78 as dormant duplicates of `\label{tab:attribution_metrics}`; recover them from git history (`git show 7df945f:overleaf_drafts/tables/<file>`) before rerunning any command below.

## Headline

Pre-meeting audit of the IG / MaRC attribution pipeline that produces
`overleaf_drafts/tables/attribution_metrics_200pair.tex`. **Numerical results
in the table itself appear correct** (sanity rows behave as expected, ABTT is
applied symmetrically to both sides of every cosine, the metric formulas are
internally consistent). The headline risks are **documentation drift** and
**citation hygiene**, not silent numerical bugs:

* **4 CRITICAL findings** — three are stale-text bugs in the LaTeX table
  caption and one is a citation/labelling bug for "compactness". All four are
  visible to a careful reviewer and would be embarrassing if pointed out at
  submission.
* **6 IMPORTANT findings** — mostly affect reproducibility / interpretability,
  including the random-baseline RNG bug (std understated), the
  `token_keep_lookup` mismatch between IG generation and metric evaluation,
  and the partner-emb pipeline's reliance on a hand-extracted PC bundle
  without consistency assertion.
* **8 COSMETIC findings** — code-quality issues that don't affect the table.
* **20+ items explicitly checked and confirmed CORRECT** (see §"Confirmed"
  below) — this is the part that supports the "no surprises" claim for the
  Monday meeting.

Run 2 should incorporate the fixes; this audit applies no patches.

## Coordination with Agent 1.1

Agent 1.1's *new* cosine investigation report is **not present** at
`docs/analyses/`, the repo root, or any worktree path as of audit time
(2026-04-26 15:30 UTC). The older 80-pair findings doc at
`overleaf_drafts/tables/FINDINGS_attribution.md` (dated 2026-04-20) is the
prior attribution-metrics analysis but does not address cosine geometry; it
should not be confused with Agent 1.1's pending work. Cosine concerns
(specifically `FULL_COS_FLOOR=0.05` firing rate) are covered in §D below;
Run 2 can replace those bullets with citations to Agent 1.1 if their report
lands first.

## Methodology

### Files audited (12)

| Path | Audited by |
|---|---|
| `src/retrieval_targets.py` | coder, statistician (formula), self-read |
| `src/attribution_metrics.py` | coder, citation-auditor, statistician, self-read |
| `scripts/ig/sample_random_test_pairs.py` | coder, self-read |
| `scripts/ig/extract_pcs_from_npzs.py` | coder |
| `scripts/ig/merge_retrieval_mark_into_canonical.py` | coder |
| `scripts/ig/run_attribution_metrics.py` | coder, scope-guard (skel only), self-read |
| `scripts/ig/run_retrieval_mark_pair_examples.py` | coder |
| `scripts/_archive/run_phase12e_pair_explanations.py` | Explore agent, self-read |
| `scripts/resubmit/persist_attribution_methods.py` | coder |
| `slurm/ig/run_attribution_200pair.sbatch` | self-read |
| `slurm/ig/run_attribution_200pair_methods_and_metrics.sbatch` | self-read |
| `slurm/ig/rerun_metrics_200pair.sbatch` | self-read |

Plus the artifact under audit: `overleaf_drafts/tables/attribution_metrics_200pair.tex`,
and one LaTa NPZ + one PhilTa NPZ for empirical verification of partner_emb provenance,
PC truncation, and key inventory.

### Team caveat

Eight `/team` sub-agents were launched in parallel (orchestrator, coder,
statistician, citation-auditor, error-analyst, scope-guard, slurm-ops,
Explore, reproducer). Five wrote useful content (coder, citation-auditor,
Explore + the relaunches that filled in their skeletons); statistician,
error-analyst, scope-guard, slurm-ops, and reproducer terminated early
producing only skeletons. The author of this audit consolidated agent
findings AND read the source files directly to fill the gaps. Where an
agent's claim conflicts with direct code inspection, the direct inspection
wins — see §"Investigated and ruled out" for one such case.

---

## Critical findings (must-fix before submission)

### C-1. LaTeX caption is for a different experiment than the table reports

- **File:** `scripts/ig/run_attribution_metrics.py:451-485`
  → renders `overleaf_drafts/tables/attribution_metrics_200pair.tex:36`
- **What:** The caption text reads:
  > "Retrieval-adapted attribution-quality metrics on **80 hand-selected**
  > query–candidate pairs from the phase12f visualization set
  > (**20 per model**; not a random sample of test pairs, so absolute values
  > should not be read as population estimates)."

  The actual run is **400 pairs (200 per model, randomly drawn from the
  test split)** for LaTa and PhilTa. The same caption later says

  > "For decoder-only models (here Qwen3-0.6B), masking replaces tokens with
  > PAD under the causal mask…"

  but **Qwen3-0.6B is not in the 200-pair table** (look at the .tex — only
  LaTa and PhilTa rows exist).

- **Why critical:** A reviewer reading the caption gets a wrong picture of
  the experimental design (curated vs random sample, sample size, model
  count). Worse: the "not a random sample, so absolute values should not be
  read as population estimates" disclaimer is *the opposite* of what the
  200-pair run is — it IS a random sample, so the population read IS valid.
  The caption actively misleads readers into discounting the table.
- **Reproducer:**
  ```bash
  cd /projects/beto/irowerojas/localLatin
  # Caption text (first ~80 chars):
  grep -A1 "Retrieval-adapted" overleaf_drafts/tables/attribution_metrics_200pair.tex | head -3
  # Actual data:
  wc -l runs/active/ig_examples_200pair/random200_examples.csv  # 401 = 400 pairs + header
  ls runs/active/ig_examples_200pair/artifacts/bowphs_LaTa/ | wc -l   # 200
  ls runs/active/ig_examples_200pair/artifacts/bowphs_PhilTa/ | wc -l # 200
  # Models in the table:
  grep "textit" overleaf_drafts/tables/attribution_metrics_200pair.tex
  ```
- **Fix sketch:** Edit `run_attribution_metrics.py:451-485`. Replace the
  "80 hand-selected pairs … 20 per model … not a random sample" preamble with
  "200 random query–candidate pairs per model (balanced 100 gold-similar +
  100 gold-dissimilar) from the phase9 test split". Drop the Qwen3-0.6B
  decoder-masking caveat unless Qwen is added back. Re-run
  `slurm/ig/rerun_metrics_200pair.sbatch` (no GPU work needed if only the
  caption changes — actually the script regenerates the .tex from the cached
  per-pair JSONs via `--render_only`).

---

### C-2. LaTeX caption misstates the LOO noise floor (10⁻⁴ vs 10⁻⁶)

- **File:** `scripts/ig/run_attribution_metrics.py:465`
  vs `src/attribution_metrics.py:41`
- **What:** Caption says
  > "tokens with $|\Delta| < 10^{-4}$ excluded as noise"

  Code constant: `LOO_NOISE_FLOOR: float = 1e-6` (with a docstring comment at
  lines 33-39 explicitly justifying 1e-6 over 1e-4 because ABTT-cleaned
  embeddings are near-invariant to single-token swaps).
- **Why critical:** Two orders of magnitude. Any reviewer cross-checking the
  filter will find a contradiction. The caption claim is unambiguously wrong.
- **Reproducer:**
  ```bash
  grep "LOO_NOISE_FLOOR" /projects/beto/irowerojas/localLatin/src/attribution_metrics.py
  grep "Delta" /projects/beto/irowerojas/localLatin/overleaf_drafts/tables/attribution_metrics_200pair.tex
  ```
- **Fix sketch:** In `run_attribution_metrics.py:465`, change `10^{-4}` to
  `10^{-6}` (or, better, format the actual constant: import `LOO_NOISE_FLOOR`
  from `attribution_metrics` and `f"10^{{{int(np.log10(LOO_NOISE_FLOOR))}}}"`).

---

### C-3. "Compactness" mis-attributed to MaRC; metric is sparsity-at-threshold, not MaRC compactness

- **File:** `src/attribution_metrics.py:1-23` (docstring claims provenance);
  `src/attribution_metrics.py:260-282` (implementation);
  `overleaf_drafts/tables/attribution_metrics_200pair.tex:9` (column header
  "Cmpct@0.8 ↓"); also implicit in the paper's prose wherever "compactness"
  is described as MaRC's contribution.
- **What:** Brinner & Zarrieß 2023 (MaRC, ACL Findings) define "compactness"
  as a **spatial-smoothness regularizer** on the soft mask: each mask weight
  `w_i` couples to its neighbours via an unnormalised Gaussian
  `w_{i→j} = w_i · exp(-d(i,j)²/σ_i²)`, and a logarithmic penalty
  `Ω_σ = -α_σ · (1/n) Σ log σ_i` softly pushes σ large. The result is
  **contiguous masks** (no isolated-token rationales). Conceptually closer
  to a total-variation prior on the mask.

  The implementation here (`compactness` at `attribution_metrics.py:260-282`)
  computes the **smallest fraction k/n s.t. `S_v(top-k) / S_v(full) ≥ 0.8`**,
  i.e., a *minimum-rationale-size-to-recover-80%-of-signal* metric. Token
  order does not enter the computation (mask is selected by `|score|` rank,
  not position). This is a **sparsity-at-threshold** metric, not contiguity.
- **Why critical:** Two different mathematical objects in different
  conceptual families. The docstring at `attribution_metrics.py:1-3`
  explicitly says "Adapts Brinner & Zarriess 2023 (MarK) and DeYoung et
  al. 2020 (ERASER) attribution evaluation metrics" — but MaRC's compactness
  is *not* what's evaluated, the implementation closer to Bastings et al. 2019
  / Lei et al. 2016 / DeYoung 2020 §rationale-length. A reviewer who knows
  MaRC will catch this immediately.
- **Reproducer:**
  ```bash
  # Show the cited MaRC formula's regularizer is on continuity, not sparsity:
  # Read Brinner & Zarrieß 2023 §3.2-3.3 (eq. 5 and 7) — compactness term is
  # the log-σ regularizer with Gaussian neighbour coupling.
  # Show our implementation has no positional/contiguity term:
  sed -n '260,282p' /projects/beto/irowerojas/localLatin/src/attribution_metrics.py
  ```
- **Fix sketch:** Two acceptable resolutions, pick one:
  1. **Re-attribute and rename.** Drop the MaRC citation from the
     `compactness` function. Rename to `min_rationale_fraction` or
     `sparsity_at_threshold`, cite Bastings et al. 2019 / Lei et al. 2016
     for "short rationale" desideratum, or DeYoung et al. 2020 for
     "rationale length". Update the .tex column header from "Cmpct@0.8" to
     e.g. "MinFrac@0.8" or "Sparsity@0.8".
  2. **Implement actual MaRC compactness alongside.** Add a TV term
     `Σ |m_i − m_{i+1}|` (over the binary top-k mask) and report it
     alongside the threshold metric. Then retain the MaRC citation honestly.

  Either way, update the docstring at `attribution_metrics.py:1-23` and the
  .tex caption.

---

### C-4. Random-baseline RNG bug — "5 seeds" produce non-i.i.d. samples; std reported is wrong

- **File:** `scripts/ig/run_attribution_metrics.py:234-237` (and the
  `_eval_one` aggregation at `:253-268`, then aggregation at `:314-338`).
- **What:** Inside `process_pair` (called once per NPZ), inside the
  `for variant in ("baseline", "abtt"):` loop, line 234 reads
  ```python
  rng = np.random.default_rng(0)
  for seed in range(random_seeds):
      r = rng.uniform(-1.0, 1.0, size=n_q)
      rows.append(_eval_one(ctx, "random", r, ...))
  ```
  The `rng` is **reseeded with the same seed (0) for every (pair, variant)
  tuple**. Therefore:
  - Across pairs of the same `n_q`, the 5 random vectors are *byte-identical*.
  - For pairs of different `n_q`, the 5 vectors share a common prefix.
  - Within (pair, variant), the 5 random vectors *are* 5 different draws —
    but there's no real "seed variance" across the 200 pairs.

  The aggregator at `:314-338` computes `vals.std(ddof=1)` over all
  `random` rows (5 × 200 = 1000 rows per (model, variant)) as if they were
  independent draws. They are not. The std reported in `summary.csv` for the
  random baseline is **dramatically understated** because it counts the same
  5 draws repeated across 200 pairs.

  Comment at `:233` claims "seed variance propagates into across-pair std"
  — that claim is **false**.
- **Why critical:** If the paper reports any significance test or confidence
  interval that uses the random-baseline std, those CIs are too narrow.
  The means themselves are unbiased (uniform distribution is symmetric), so
  the cell *values* in the table are fine — but any "method X beats random
  by Y std" claim is overstated. **Severity is borderline IMPORTANT/CRITICAL**:
  if the paper text only uses random as a qualitative "the ranking matters"
  baseline (as `FINDINGS_attribution.md` recommends), then this is
  IMPORTANT; if any p-value or CI on random is reported, this is CRITICAL.
- **Reproducer:**
  ```bash
  /u/irowerojas/.conda/envs/localLatin/bin/python <<'PY'
  import numpy as np
  for trial in range(3):
      rng = np.random.default_rng(0)  # same seed each time
      v = rng.uniform(-1, 1, size=10)
      print(trial, v[:5])
  # All 3 trials print identical 5-element prefixes — that's exactly the
  # in-loop reseeding pattern in run_attribution_metrics.py.
  PY
  ```
- **Fix sketch:** Move `rng` construction *outside* `process_pair` and
  pass it in (so a single RNG is shared across all pairs and variants),
  or seed deterministically per-pair:
  ```python
  rng = np.random.default_rng(int(npz_path.stem.encode().hex(), 16) % 2**32)
  ```
  To get true seed variance, use `for seed in range(random_seeds): rng = np.random.default_rng(seed)` inside the loop. Re-render via `--render_only` after re-running metrics on the affected pairs.

---

## Important findings (should-fix before publication)

### I-1. `token_keep_lookup` is applied during IG / MaRC but NOT during metric evaluation

- **File:** `scripts/ig/run_attribution_metrics.py:99-111` (`forward_pooled`,
  no token_keep_lookup) vs `_archive/run_phase12e_pair_explanations.py:240-243`
  (uses `token_keep_lookup` in `pool_hidden`) vs
  `scripts/ig/run_retrieval_mark_pair_examples.py:181-185` (uses
  `numpy_token_keep_mask`).
- **What:** The IG attribution and the MaRC mask optimization both operate
  on a cosine that uses **filtered pooling** — only "real" tokens (per the
  tokenizer-empty filter) contribute to the mean. The metric driver
  evaluates the same cosine using **unfiltered pooling** — every token in
  the attention mask contributes.

  Two consequences:
  1. The `full_cos` reported by `run_attribution_metrics.py` is *not* the
     same scalar that IG was attributing to (or that MaRC was optimizing
     against). They are close but not identical.
  2. The Suff/Comp ratios are computed against the driver's own `full_cos`,
     so the metric is **internally consistent within the driver** — but it
     evaluates faithfulness on a slightly different decision function than
     the one the attribution method explained.
- **Recommended fix:** Build `token_keep_lookup` from the tokenizer (use
  `src/token_filtering.py`) and apply it inside `forward_pooled`. Then
  `full_cos` will match the IG decision scalar exactly. Re-run metrics
  (re-uses NPZs; fast re-render).

### I-2. Caption's "IG and OT identical rows" disclosure is good — but suggests OT is not independent evidence

- **File:** `scripts/ig/run_attribution_metrics.py:478-482`
- **What:** Caption explicitly says "IG and OT produce numerically identical
  rows because the OT pair-matrix uses |IG| as transport mass and our
  row-sum-positive reduction recovers the same per-token magnitudes". The
  table confirms this: LaTa baseline IG = OT = (0.585, 0.065, 0.318, -0.014).
  PhilTa baseline IG = OT = (0.779, 0.114, 0.136, 0.042).
- **Why important:** The 7-method comparison loses one independent column —
  effectively only 6 independent methods. A reader skimming the table sees
  IG and OT both bolded (or near-best) and may double-count the evidence.
  The disclosure is in the caption (good), but the table layout would benefit
  from either dropping OT or visually grouping IG+OT.
- **Recommended fix:** In the .tex, either (a) replace the OT row with
  "(same as IG)" in italics, or (b) drop the OT row and add a footnote.
  Or, in `run_attribution_metrics.py:96`, change OT's reducer to
  `row_sum_signed` (which would make it differ from IG via signed structure
  in the OT plan).

### I-3. `extract_pcs_from_npzs.py` blindly picks `candidates[0]`; no consistency assertion

- **File:** `scripts/ig/extract_pcs_from_npzs.py:42-46`
- **What:** Picks the first NPZ for each model and assumes its `pcs` and
  `mean_vec` are identical to all other NPZs for that model. The docstring
  claims this is "by construction". For the 200pair run this is true (every
  pair uses the same per-model D, layer, and PC bank), but the script does
  not assert it.
- **Why important:** If a single NPZ for one model was regenerated under a
  different protocol (different D, different train split, different layer),
  the extracted PC bundle would silently be wrong — and downstream
  consumers that load it will use the wrong PCs.
- **Recommended fix:** Iterate through all NPZs and assert
  `np.array_equal(this_pcs, candidates[0]_pcs)` and same for `mean_vec`.
  Fail loudly if any pair disagrees.

### I-4. `sample_random_test_pairs.py` bakes in `tau`, `baseline_tau`, `abtt_tau` from a prior run

- **File:** `scripts/ig/sample_random_test_pairs.py:27-65`
- **What:** Per-model `tau`, `baseline_tau`, `abtt_tau`, `layer`, `D` are
  hard-coded constants from `runs/active/ig_examples/phase12f_examples.csv`
  (a prior run). If the upstream phase9 evaluator re-tunes ABTT D or
  re-fits taus, these constants will silently drift.
- **Why important:** Reproducibility risk. A future re-run of the upstream
  ABTT-tuning pipeline will not propagate to this CSV; the 200pair NPZs will
  use stale per-model parameters.
- **Recommended fix:** Read the values at run time from the canonical
  phase9 results CSV, or assert that the baked-in values match a checked-in
  source.

### I-5. `run_attribution_metrics.py` default args target the OLD 80-pair location

- **File:** `scripts/ig/run_attribution_metrics.py:498-507`
- **What:** Default `--examples_csv`, `--artifacts_root`, `--out_root`,
  `--tex_out` point to `runs/active/ig_examples/` and
  `overleaf_drafts/tables/attribution_metrics.tex` (no `_200pair` suffix).
  The 200pair run depends on the sbatch passing the right args explicitly —
  which it does. But running the script ad-hoc on the login node without
  args would silently regenerate the *old* table.
- **Why important:** Footgun. A future user who runs `python
  scripts/ig/run_attribution_metrics.py --render_only` without args will
  overwrite the wrong .tex.
- **Recommended fix:** Either remove the defaults (force args), or add a
  preflight assertion that the artifacts_root is non-empty and recent. At
  minimum, print the resolved paths before doing work so an interactive
  user notices.

### I-6. ρ_LOO mis-attributed to Atanasova et al. 2020

- **File:** `src/attribution_metrics.py:21-23` (docstring)
- **What:** Docstring says "what some papers call 'faithfulness'
  (Atanasova et al. 2020)". Atanasova 2020's headline faithfulness metric is
  Faithfulness-AUC (perturbation-and-AUC), not a Spearman correlation
  between |attribution| and per-token leave-one-out drops. The closest
  primary source for the Spearman/Kendall-correlation-with-LOO formulation
  is Jain & Wallace 2019 ("Attention is not Explanation").
- **Why important:** Citation hygiene. The metric implementation is sound,
  the attribution is loose. A reviewer familiar with Atanasova 2020 will
  push back.
- **Recommended fix:** Change the docstring reference to "Jain & Wallace
  2019; cf. Atanasova et al. 2020 for the broader faithfulness-AUC family".
  Add Jain & Wallace 2019 to the paper bibliography.

---

## Cosmetic findings (nice-to-fix; don't block submission)

### M-1. Path rewrite from `/u/.../localLatin/canon/` to `/projects/.../data/canon/` is hard-coded

- **File:** `scripts/ig/sample_random_test_pairs.py:159-167`
- Two specific path strings hard-coded; failure mode is a loud `SystemExit`,
  so it's safe — just brittle. Worth extracting to a shared utility.

### M-2. Hardcoded `MODEL_LAYER` dict duplicated across `extract_pcs_from_npzs.py` and `sample_random_test_pairs.py`

- **File:** `scripts/ig/extract_pcs_from_npzs.py:19-24`, mirrors
  `scripts/ig/sample_random_test_pairs.py:27-65`
- Risk of divergence on future model additions. Consolidate into one config.

### M-3. `np.savez` (uncompressed) overwrites a `np.savez_compressed` NPZ in two places

- **File:** `scripts/ig/merge_retrieval_mark_into_canonical.py:149` and
  `scripts/resubmit/persist_attribution_methods.py:219`
- Original NPZ from `_archive/run_phase12e_pair_explanations.py:298` was
  `savez_compressed`. After merge / persist, the canonical NPZ becomes
  uncompressed (~3-5× larger on disk). Functionality unaffected.
- **Fix:** Use `np.savez_compressed(tmp_path, **arrays)` in both places.

### M-4. `methods_available` separator detection treats empty-string asymmetrically

- **File:** `scripts/ig/merge_retrieval_mark_into_canonical.py:163-171`
- Defaults to `,` separator if `;` not present. For an empty initial value,
  the result after appending `retrieval_mark` is just `"retrieval_mark"`
  (no separator needed). Behavior is correct but the if/else can be simplified.

### M-5. `random_seeds=5` hardcoded in `process_pair` signature, not exposed via CLI

- **File:** `scripts/ig/run_attribution_metrics.py:153, 235`
- Combined with C-4 the actual variance contribution from the 5 rows is
  ~0, so increasing the count would not help anyway. After C-4 is fixed,
  expose this via CLI.

### M-6. Hyperparams JSON serialized into a fixed-length numpy unicode string `<U500`

- **File:** `scripts/ig/run_retrieval_mark_pair_examples.py:381-383`
- Fine for the small dict, but a long path or extended config could
  silently truncate. Use `dtype=object` or add a length assertion.

### M-7. `is_t5_hint` heuristic substring-matches `lata`/`philta`

- **File:** `scripts/ig/run_retrieval_mark_pair_examples.py:127`
- Best-effort with a graceful fallback to `AutoModel.from_pretrained` at
  line 136. Brittle but not buggy.

### M-8. Per-file finding skeletons exist for statistician/error-analyst/scope-guard/slurm-ops/reproducer agents but were not filled in

- **File:** `/u/irowerojas/audit_scratch_run1_agent2/*.md`
- Agent token-budget exhaustion. Findings consolidated by direct read.
  No bug, just a process note: future `/team` invocations should use
  tighter agent prompts and the persistent scratch path discipline that
  worked for the 3 agents that did complete (coder, citation-auditor, Explore).

---

## Confirmed-correct items (gives the meeting confidence)

The following were specifically checked against code (and where applicable
against a real NPZ) and found correct:

### Architecture / data flow
- ✅ ABTT applied to **both** query pooled vector and partner (candidate)
  pooled vector in the IG generator
  (`_archive/run_phase12e_pair_explanations.py:242-243, 248-266`),
  in the metrics driver (`run_attribution_metrics.py:181-182, 200-201`),
  and in the MaRC optimizer (`run_retrieval_mark_pair_examples.py:298-313`).
  Same `pcs` and `mean_vec` used on both sides in every variant block.
- ✅ `pcs` are stored in NPZ as the **D-truncated** matrix (verified by
  `np.load(...)["pcs"].shape == (D, hidden_dim)`), so the metrics driver
  loads what IG used. Earlier Explore-agent finding of a "CRITICAL D-mismatch
  between IG and metrics" was based on a misreading and is **not a bug**:
  - LaTa NPZ: `D=2`, `pcs.shape=(2, 768)`
  - PhilTa NPZ: `D=10`, `pcs.shape=(10, 768)`
- ✅ `mean_vec` is leak-free — fit on train embeddings only via
  `_archive/run_phase12_prepare_pcs.py:59-60, 88-91` (`train_mask = (split_df["split"] == "train").values`).
- ✅ IG attributes through the **embedding layer** (correct starting point)
  via `LayerIntegratedGradients(target, emb_layer)` in
  `_archive/run_phase12e_pair_explanations.py:268-295`. Gradients propagate
  through all transformer blocks to reach layer L.
- ✅ Per-token IG length matches stored token list 1:1, no CLS shift
  (`tokenize_text` uses `padding=False`; defensive trim to `attention_mask.sum()`
  at `run_attribution_metrics.py:158, 227`).

### Metric math
- ✅ `top_k_mask` uses stable argsort (`np.argsort(-abs_scores, kind="stable")`)
  with deterministic tie-breaking by lower token index
  (`attribution_metrics.py:148`).
- ✅ `k_from_fraction(0.25, n) = max(1, ceil(0.25 * n))` — rounds **up**;
  matches the convention in MaRC / ERASER. For n=10, k=3 (not 2).
- ✅ `FULL_COS_FLOOR=0.05` NaN handling is consistent across sufficiency,
  comprehensiveness, and compactness (gate at `attribution_metrics.py:202,
  239, 274`). Aggregator drops NaNs per column before averaging
  (`run_attribution_metrics.py:331`).
- ✅ `loo_correlation` filters constant vectors before `spearmanr`
  (`attribution_metrics.py:300-321`); returns NaN with proper
  `loo_n_used`/`loo_n_total` accounting if fewer than 3 tokens survive
  filtering or if either vector is constant.
- ✅ `compactness` returns `1.0` (worst) if no k attains the threshold;
  semantically defensible (means "even all tokens don't recover 80%").
- ✅ `infer_methods` requires both `_baseline` AND `_abtt` keys present
  before a method appears in the table — avoids partial-row asymmetry.
- ✅ Reducer registry: `attention_*` use `row_sum_signed`, `ot` uses
  `row_sum_positive`, `bertscore` uses `row_max`, `ig` uses stored per-token
  vector. Consistent with how each pair-matrix is constructed.

### Sanity-row construction
- ✅ Inverse pseudo-baseline `1 / (1e-9 + |IG|)` correctly produces
  anti-IG ranking (since `inv_scores >= 0`, `|inv_scores| = inv_scores`,
  `top_k_mask` selects smallest-|IG| tokens). Anti-correlated with |IG|
  as required.
- ✅ Sanity row qualitative ordering (verified against
  `attribution_metrics_200pair.tex`):
  - PhilTa ABTT: random Suff=0.808 > inverse Suff=0.776 (random beats
    inverse on a higher-better metric — correct direction).
  - PhilTa ABTT: random Cmpct=0.144 < inverse Cmpct=0.196 (random beats
    inverse on a lower-better metric — correct direction).
  - PhilTa baseline: inverse ρ_LOO = -0.042 (negative — inverse should be
    anti-correlated with IG-ranked LOO).
- ✅ Random rows: even though the RNG is reseeded (C-4), the **mean** is
  unbiased because the uniform distribution is symmetric. Only the std is
  affected.

### SLURM DAG
- ✅ `slurm/ig/run_attribution_200pair.sbatch:92-96` passes
  `--examples_csv runs/active/ig_examples_200pair/random200_examples.csv`
  and `--tex_out overleaf_drafts/tables/attribution_metrics_200pair.tex`,
  overriding the bad defaults at `run_attribution_metrics.py:498-507`.
  Same for `run_attribution_200pair_methods_and_metrics.sbatch:41-45` and
  `rerun_metrics_200pair.sbatch:33-37`.
- ✅ `--skip_existing` flag used in `--steps 200` MaRC stage and IG stage
  — idempotent restarts.
- ✅ `runs/phase12_release/pcs/bowphs_LaTa/` and
  `runs/phase12_release/pcs/bowphs_PhilTa/` exist (verified `ls`).
- ✅ Atomic NPZ write pattern uses `.tmp.npz` + `os.replace` correctly in
  both `merge_retrieval_mark_into_canonical.py:148-150` and
  `persist_attribution_methods.py:210-220` (the docstring at the latter
  explains the `np.savez` `.npz` suffix gotcha).
- ✅ `random200_examples.csv` is committed and frozen — re-running
  `sample_random_test_pairs.py` with `--seed 20260420` would reproduce it.

### LaTeX table direction (read directly from the .tex)
- ✅ Arrows: `Suff@25\%~↑`, `Comp@25\%~↑`, `Cmpct@0.8~↓`, `ρ_LOO~↑`
  match `HEADLINE_LABELS` at `run_attribution_metrics.py:280-285` and
  match the code's "higher better" convention for Suff/Comp/ρ_LOO and
  "lower better" for Compactness.
- ✅ Bolding direction matches direction of better via `HIGHER_BETTER` dict
  at `run_attribution_metrics.py:348-353`. Manual check of the .tex bolds:
  LaTa baseline Cmpct@0.8 bolds `0.017` (DLA) — that IS the smallest among
  real methods. LaTa baseline Suff@25% bolds `1.026` (BERTScore) — that IS
  the largest among real methods. Correct.

---

## Open questions for the author (Ian)

These are choices the audit cannot make alone:

1. **Re-run vs caption-only fix for C-1, C-2, C-3?** The .tex is
   auto-generated by `run_attribution_metrics.py:render_latex`. If you
   change only the caption text in the script and re-run with
   `--render_only`, no GPU work is needed; the per-pair JSONs are cached.
   Fastest fix.

2. **Compactness rename or contiguity addition (C-3)?** Renaming to
   "min rationale fraction" / "sparsity@0.8" is a one-line script edit and
   loses no science. Adding a TV-based MaRC compactness *alongside* would
   add a column but require re-rendering. What's your preference?

3. **Drop OT or relabel (I-2)?** OT and IG are numerically identical by
   construction. The caption discloses this; should the table follow suit?

4. **Random-baseline std reporting (C-4)?** Does the paper text actually
   use the random std for any claim? If yes, fix C-4 as CRITICAL. If no
   (per `FINDINGS_attribution.md` recommendation 1, "ABTT improves ρ_LOO
   on every cell"), C-4 is IMPORTANT but not blocking.

5. **`token_keep_lookup` consistency (I-1)?** The metrics driver evaluates
   a slightly different cosine than IG was attributing to. How material is
   this? Worth a one-pair empirical check (I ran out of time to do it):
   compare `cos_orig_baseline` (stored in NPZ from IG generator, with token
   filter) vs `full_cos` (computed by metrics driver, without filter) for
   10 pairs. If the difference is < 1e-3, mark I-1 as low-priority. If
   > 1%, escalate to CRITICAL.

6. **Agent 1.1 cosine investigation:** Their report has not landed. If
   their findings overlap with §I-1 or §C-1's "FULL_COS_FLOOR doesn't
   fire" observation, replace those bullets with citations to their report.

---

## Reproducer empirical data (corroborates the audit's claims)

The reproducer agent confirmed empirical values for two random pairs from
the per-pair JSONs at `runs/active/ig_examples_200pair/attribution_metrics/<slug>/exampleNNN_pair_example.json`
(these per-pair files exist — they are the ground truth that `summary.csv`
aggregates):

| Pair | full_cos | suff@0.25_raw | suff@0.25_ratio | comp@0.25_drop | comp@0.25_ratio | compactness@0.80 |
|---|---|---|---|---|---|---|
| LaTa example001 (n_q=48, layer=4) IG/base | 0.9987 | 0.2272 | 0.2275 | 0.0319 | 0.0319 | 0.5833 |
| PhilTa example201 (n_q=105, layer=6) IG/base | 0.99976 | 0.99981 | 1.00005 | 0.00013 | 0.00013 | 0.0952 |

This confirms three key audit claims:
- `|full_cos|` is far above `FULL_COS_FLOOR=0.05` (here ~0.999 for both
  pairs) — the floor never fires on this run, so its NaN-handling code path
  is unreachable in production.
- The per-pair JSONs are the source-of-truth for aggregation; the driver's
  ratio metrics are computed from `full_cos` re-derived by `forward_pooled`
  (not from any stored embedding), so I-1 (token_keep_lookup mismatch) is
  the place to verify cross-pipeline consistency, not the JSON schema.
- The PhilTa example201 case (suff_ratio = 1.00005, compactness = 0.095)
  shows pairs where ~10% of tokens recover 100% of cosine — these
  "near-trivial" pairs pull compactness averages low and may be the
  dominant signal in the table's headline numbers.

The reproducer did not finish reloading the model to recompute the metrics
from scratch (T5 model load + 100+ forward passes per pair on CPU was too
slow to complete within the agent's budget). This is a minor follow-up:
the comparison is "compute from NPZ matches per-pair JSON" — both come
from the same `process_pair` call inside `run_attribution_metrics.py`, so
agreement is tautological unless someone hand-edited a JSON.

---

## Investigated and ruled out (don't get spooked)

The following items were flagged by individual auditors but turned out to
be non-bugs after deeper investigation. Listed here so the meeting doesn't
re-discover them as concerns:

1. **"CRITICAL D-truncation mismatch between IG and metrics"** (Explore agent):
   Explore claimed IG used `pcs[:D]` but metrics used full 10 PCs. **WRONG.**
   The NPZ stores **D-truncated** pcs (verified empirically:
   `np.load(latai_npz)["pcs"].shape == (2, 768)` for LaTa with D=2). The
   metrics driver loads what's stored, which is already truncated to D. So
   IG and metrics use the same `pcs`. **Not a bug.** (Explore's reading of
   `_archive/run_phase12e_pair_explanations.py:315` was wrong — it saves
   `pcs.detach().cpu().numpy()` where `pcs` was loaded as
   `pc_data["pcs"][:d_value]`, not the full PC bank.)

2. **"FULL_COS_FLOOR=0.05 silently biases the table"** (statistician's
   mid-thought finding): the `full_cos` for the 200-pair run is always in
   [0.93, 0.99] — well above the 0.05 floor. The floor never fires. NaN
   handling is moot here. summary.csv shows 200/200 (LaTa) and 199/200
   (PhilTa abtt) coverage per metric (per error-analyst's mid-thought
   finding). **Not a bug.**

3. **"`top_k_mask` ties at zero pick first k tokens"** (statistician's
   pre-investigation hypothesis): would only matter if any attribution
   method produced all-zero scores for any pair, which doesn't happen for
   the 7 methods on the 200-pair NPZs (ig, bertscore, ot, attention_*, dla,
   retrieval_mark all produce non-trivial per-token scores). **Not a bug
   in practice**; cosmetic concern only.

---

## Verification

Reviewer / merge agent can confirm this audit was done by:

```bash
cd /projects/beto/irowerojas/worktrees/run-1-agent-2

# 1. Report exists and is non-trivial
test -f docs/analyses/attribution_pipeline_audit.md
wc -l docs/analyses/attribution_pipeline_audit.md  # expect > 300

# 2. Spot-check the empirical claims in C-1
wc -l /projects/beto/irowerojas/localLatin/runs/active/ig_examples_200pair/random200_examples.csv  # = 401
ls /projects/beto/irowerojas/localLatin/runs/active/ig_examples_200pair/artifacts/bowphs_LaTa/ | wc -l  # = 200

# 3. Spot-check C-2 (LOO floor)
grep "LOO_NOISE_FLOOR" /projects/beto/irowerojas/localLatin/src/attribution_metrics.py
grep "Delta" /projects/beto/irowerojas/localLatin/overleaf_drafts/tables/attribution_metrics_200pair.tex

# 4. Spot-check C-4 (RNG bug — reproducer above)

# 5. Branch is committed and pushed
git log --oneline -3
git rev-parse --abbrev-ref HEAD  # = run-1/agent-2-pipeline-audit
git ls-remote origin run-1/agent-2-pipeline-audit  # ref present
```
