# Orchestra: Monday Meeting Prep — Attribution Experiment v3

Generated: 2026-04-26

## Context

Friday's meeting (2026-04-25) flagged two distinct concerns about the 200-random-pair attribution experiment that we are presenting on Monday:

1. **Negative pairs are ill-defined for these metrics.** Suff/Comp/Compactness assume there is a real cosine signal to localize. For negatives, the model can trivially "win" by masking everything (cos→0). Fix: rerun on **200 positives** per model.
2. **Cosine inflation.** `full_cos` lands at 0.93–0.99 across all cells regardless of label, but the live demo shows even gold-positives at 0.84–0.92. Either the attribution layer's anisotropy is the culprit (in which case ABTT-variant cosines should be lower and they aren't), or there is a pooling / preprocessing / labeling bug. **Must root-cause before re-running** — wrong-but-confident numbers are worse than no numbers.

The deliverable for Monday: a refreshed bundle in `professor_share/` (PDF table + charts + comprehensive HTML walkthrough + Q&A prep), backed by a clean diagnostic of the cosine issue and a full pipeline audit, all re-validated on 200 positives.

## Overview
- **Total runs**: 4
- **Total agents**: 8 working + 2 merge
- **Estimated parallelism**: Run 1 = 3 parallel, Run 2 = 1 sequential, Run 3 = 3 parallel, Run 4 = 1 sequential
- **Key dependencies**: Run 1 must finish before Run 2 because the GPU experiment incorporates Run 1's fixes. Run 3 can only start once Run 2's `summary.csv` exists. Run 4 is a final cross-check before Ian sends.

## Dependency Graph

```
Run 1 (parallel diagnostic + prep, no GPU)
  ├── 1.1 Cosine inflation root cause + fix
  ├── 1.2 Pipeline code audit (use /team)
  └── 1.3 Positives-only 200-pair sampler + new sbatch
       │
       └── Merge 1.M
            │
            ▼
Run 2 (sequential, GPU)
  └── 2.1 Submit + cron-poll the 200-positives experiment
       │
       │ (job completes, summary.csv exists)
       ▼
Run 3 (parallel bundle build)
  ├── 3.1 PDF table + charts from new summary.csv
  ├── 3.2 Comprehensive HTML walkthrough (use /team)
  └── 3.3 Q&A prep doc
       │
       └── Merge 3.M
            │
            ▼
Run 4 (sequential verify)
  └── 4.1 Cross-check bundle, write READY.md
```

---

## Run 1: Diagnose & prep
> **Prerequisite**: None
> **Agents**: 3 in parallel
> **Worktrees**: Yes
> **Why parallel**: Each agent produces an independent artifact (one fix patch + report, one audit report, one new sampler + sbatch). They touch disjoint files.

### Agent 1.1: Cosine inflation root cause + fix
> **Area**: `src/retrieval_targets.py`, `src/attribution_metrics.py`, `scripts/ig/sample_random_test_pairs.py`, `runs/active/ig_examples_200pair/`
> **Worktree branch**: `run-1/agent-1-cosine-investigation`
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-1 -b run-1/agent-1-cosine-investigation
cd ../worktrees/run-1-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

The professor noticed in our 200-random-pair attribution PDF that even gold-NEGATIVE pairs show cosine 0.93-0.99 at the attribution layer — but the live retrieval demo shows gold-POSITIVES landing at 0.84-0.92. The attribution pipeline is producing inflated cosines that don't match the rest of the system. Find out why.

Hypotheses worth chasing (don't stop at the obvious — verify each with code):
- Mid-layer anisotropy puts everything in a tight cone. But ABTT-variant cosines should NOT be inflated and they apparently are.
- Pooling mismatch between the model-side path (BaselineCosSimTarget mean-pools) and the partner_emb (possibly SIF-weighted or differently constructed from the cached embedding store).
- "Negatives" in the sampler aren't actually negatives — verify directory labels.
- ABTT cleaning applied to one side but not the other, or with different PCs / mean_vec.
- Layer mismatch between partner_emb (cached at one layer) and model output (extracted at another).

Check the source CSV at runs/active/ig_examples_200pair/attribution_metrics/summary.csv and the per-pair detail under that directory to see exactly what the negative-pair full_cos values look like. Read CLAUDE.md for retrieval pipeline conventions.

Write up the root cause and the fix at docs/analyses/cosine_inflation_investigation.md. If the bug is real, also include a code patch in this branch. If the investigation reveals a fundamental design problem that changes what we should run on Monday (e.g., we need a different attribution layer), STOP and flag it loudly in the report — do not silently push a half-fix.

Commit the report and any code change. Push the branch.
````

</details>

### Agent 1.2: Full pipeline code audit
> **Area**: `src/retrieval_targets.py`, `src/attribution_metrics.py`, `scripts/ig/*.py`, `slurm/ig/run_attribution_200pair.sbatch`
> **Worktree branch**: `run-1/agent-2-pipeline-audit`
> **Use /team**: Yes — split sub-agents per file area

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-2 -b run-1/agent-2-pipeline-audit
cd ../worktrees/run-1-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

Audit the entire attribution pipeline for bugs. Ian wants to walk into Monday's meeting confident there are no surprises. The pipeline lives across:
- src/retrieval_targets.py (BaselineCosSimTarget, ABTTCosSimTarget — Captum forward wrappers)
- src/attribution_metrics.py (Suff/Comp/Compactness/rho_LOO computation)
- scripts/ig/sample_random_test_pairs.py
- scripts/ig/extract_pcs_from_npzs.py
- scripts/ig/merge_retrieval_mark_into_canonical.py
- scripts/ig/run_attribution_metrics.py
- scripts/ig/run_retrieval_mark_pair_examples.py
- scripts/_archive/run_phase12e_pair_explanations.py (the IG NPZ generator referenced by the sbatch chain)
- slurm/ig/run_attribution_200pair.sbatch and persist_attribution_methods.sbatch

Focus on:
- Metric formula correctness vs the canonical sources: Suff/Comp/Compactness from ERASER (DeYoung et al. 2020); ρ_LOO closer to Atanasova et al. 2020. MaRC method itself from Brinner & Zarriess 2023.
- Direction of "higher is better" vs "lower is better" being consistent end-to-end (sampling, computation, table presentation).
- ABTT consistently applied to BOTH the model-side pooled vector AND the partner_emb (not one and not the other).
- Dimension / index off-by-ones, especially around top-k token selection at the 25% / 0.8 thresholds.
- NaN, zero-denominator, and small-cos edge cases (the FULL_COS_FLOOR=0.05 filter — does it fire when expected?).
- partner_emb provenance: which layer, which pooling, which post-processing.
- Token-fraction-vs-count math at the 25% / 0.8 boundaries (rounding direction).
- random / inverse sanity rows actually being random / inverse.

Use /team to split this — one sub-agent per file or file group is reasonable. Output: docs/analyses/attribution_pipeline_audit.md with findings classified as critical / important / cosmetic. Critical findings must include reproducer steps and a recommended fix sketch. Don't apply patches yourself — Run 2 incorporates them.

Coordinate with Agent 1.1: their cosine investigation may overlap with parts of this audit. Don't duplicate; cite their report if relevant.

Commit the audit report. Push the branch.
````

</details>

### Agent 1.3: Positives-only 200-pair sampler + new sbatch
> **Area**: `scripts/ig/`, `slurm/ig/`
> **Worktree branch**: `run-1/agent-3-positives-sampler`
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-3 -b run-1/agent-3-positives-sampler
cd ../worktrees/run-1-agent-3

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

Build a new sampler that draws 200 POSITIVE pairs per model from the test split (same directory, both files in test, both winnable). The existing scripts/ig/sample_random_test_pairs.py is the closest reference — it currently draws 100 pos + 100 neg. We want 200 positives per model, zero negatives. Do NOT do "100 positives with the negative half stripped"; we want a full 200-positive sample.

Models scoped to LaTa and PhilTa only. Reuse the FEATURED_MODELS dict from the existing sampler verbatim — keep layer / D / tau / baseline_tau / abtt_tau values as-is unless Agent 1.1's cosine investigation calls for a different layer. Output the per-model pair file in the same shape the downstream pipeline expects (look at the existing input file referenced by run_attribution_200pair.sbatch).

Also write the matching slurm/ig/run_attribution_200pos.sbatch chaining the same four pipeline stages (IG NPZ generation → MaRC mask optimization → persist_attribution_methods → metrics computation) with output dir runs/active/ig_examples_200pos/. Mirror the working configuration from slurm/ig/run_attribution_200pair.sbatch including the PYTHONPATH export. Add SLURM email notification: --mail-user=ian.rowe.rojas@gmail.com, --mail-type=END,FAIL. Ask for ~4h on gpuA100x4 (we have ~7-11h budget remaining and want headroom).

Do NOT submit the job — Run 2 does that after the audit and investigation findings are merged. Just stage the script and the sbatch.

Commit the new sampler and sbatch. Push the branch.
````

</details>

### Merge Agent 1.M: Integrate Run 1 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the three Run 1 branches into main and clean up worktrees.

1. git checkout main
2. Merge order (least-conflicting first, since each agent produced disjoint files):
   - git merge run-1/agent-3-positives-sampler   # all new files (sampler + sbatch)
   - git merge run-1/agent-2-pipeline-audit       # new doc only
   - git merge run-1/agent-1-cosine-investigation # new doc + possibly src/ changes
   - Resolve any conflicts preferring the incoming change unless it breaks an obvious invariant.

3. Verify after merging:
   - docs/analyses/cosine_inflation_investigation.md exists and is non-empty
   - docs/analyses/attribution_pipeline_audit.md exists and is non-empty
   - The new positives sampler exists under scripts/ig/
   - slurm/ig/run_attribution_200pos.sbatch exists
   - If Agent 1.1 included a code change, it landed in src/

4. git worktree remove ../worktrees/run-1-agent-1
   git worktree remove ../worktrees/run-1-agent-2
   git worktree remove ../worktrees/run-1-agent-3
   git branch -d run-1/agent-1-cosine-investigation run-1/agent-2-pipeline-audit run-1/agent-3-positives-sampler

5. Read the cosine investigation report. If it says "do not run yet" or describes an unresolved fundamental issue, surface this clearly in your final message — Run 2 needs to know.
````

</details>

---

## Run 2: GPU experiment
> **Prerequisite**: Run 1 complete + merged. Cosine investigation + audit findings landed; new sampler + sbatch in place. **Two pre-flight steps must run before the GPU job**: (a) patch stale D values in `scripts/ig/sample_positive_test_pairs.py` (cosine report §6), (b) refit PCs via `scripts/ig/refit_pcs_for_attribution.py` (cosine report §7). Skipping either silently reproduces the 0.985 inflation.
> **Agents**: 1 sequential
> **Worktrees**: No (single agent, no parallelism benefit)
> **Why sequential**: Just submit and monitor — nothing to parallelize.

### Agent 2.1: Submit + cron-poll the 200-positives experiment
> **Area**: SLURM, cron, scripts/ig/
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
Run 1 produced:
- scripts/ig/sample_positive_test_pairs.py (NEW positives-only sampler, 200 per model)
- slurm/ig/run_attribution_200pos.sbatch (NEW chained sbatch)
- scripts/ig/refit_pcs_for_attribution.py (NEW helper, written by Agent 1.1)
- docs/analyses/cosine_inflation_investigation.md (Agent 1.1 — root-caused the 0.985 inflation: stale PCs from old phase12c, fit on a different distribution)
- docs/analyses/attribution_pipeline_audit.md (Agent 1.2)

Read both reports first. If either flags a critical unresolved issue or says "do not run yet", STOP and surface it to Ian — do not submit blindly. The cosine report does NOT say stop, but it ships TWO mandatory prerequisites you must execute before the GPU job:

PREREQUISITE A — Patch FEATURED_MODELS in the positives sampler (cosine report §6):
  Agent 1.3 wrote scripts/ig/sample_positive_test_pairs.py BEFORE Agent 1.1's investigation and DUPLICATED FEATURED_MODELS inline. Lines ~35/54/63 still have the broken per-model D values (LaTa=2, LaBSE=1, Qwen3-0.6B=3). Patch them all to D=10 universal so they match scripts/ig/sample_random_test_pairs.py. Without this patch, even after the PC refit the script will slice pcs[:2] for LaTa and silently reproduce the inflation.
  Verify with: grep -n '"D"' scripts/ig/sample_positive_test_pairs.py — every model must show D=10.

PREREQUISITE B — Refit the PCs (cosine report §7 step 1):
  conda run -n localLatin python scripts/ig/refit_pcs_for_attribution.py
  This overwrites runs/phase12_release/pcs/<slug>/layer{L}_pcs.npz with fresh D=10 PCs fit on the same mean-pool path forward_pooled produces. Without this, the existing PC files have only 2 rows for LaTa (1 for LaBSE, 3 for Qwen3-0.6B) and the D=10 config is a silent no-op. Empirically the old PCs are off-distribution: old-vs-new PC1 cosine = 0.10, mean_vec L2 diff = 5728.
  Sanity-check after refit: load one of the rewritten NPZs and confirm pcs.shape == (10, hidden_dim).

Then proceed:
1. Run the patched sampler to generate the input pair file (LaTa + PhilTa, 200 positives each).
2. Sanity-check: open the generated file, confirm 200 rows per model, all same-directory pairs, all in the test split, all winnable, AND that the per-row D column is 10 for every model.
3. sbatch slurm/ig/run_attribution_200pos.sbatch and capture the job ID.
4. Verify the SLURM email is configured (squeue / scontrol show job <id>). If not, scontrol update jobid=<id> MailUser=ian.rowe.rojas@gmail.com MailType=END,FAIL.
5. Set up a 30-minute polling cron (use the schedule skill or CronCreate). Each tick: squeue -j <id>; if completed, send a macOS notification ("Attribution 200pos done, status: <state>") and write a one-line status to /tmp/attribution_200pos_status.txt; then delete the cron.

Sanity checkpoint: at the 30-minute mark, the job should have moved past sampler-input prep and be inside IG NPZ generation. If it's still queued or still in setup at 30 min, investigate.

Expected post-fix numbers (from cosine report §4.3, computed on the same canon test pool the pipeline samples from): LaTa L4 abtt full_cos_mean ~0.30 (pos ~0.58, neg ~0.00), down from the broken 0.985. If the new summary.csv still shows abtt full_cos > 0.9, the refit didn't take or the sampler patch didn't land — STOP and surface to Ian rather than handing broken numbers to Run 3.

Final state expected: runs/active/ig_examples_200pos/attribution_metrics/summary.csv exists with rows for both models × baseline+abtt × all attribution methods. Run 3 starts once that file is present.
````

</details>

---

## Run 3: Bundle build
> **Prerequisite**: Run 2 complete; `runs/active/ig_examples_200pos/attribution_metrics/summary.csv` exists.
> **Agents**: 3 in parallel
> **Worktrees**: Yes
> **Why parallel**: Three independent artifacts (PDF + charts, HTML walkthrough, Q&A doc). Disjoint files.

### Agent 3.1: PDF table + charts
> **Area**: `professor_share/`
> **Worktree branch**: `run-3/agent-1-pdf-charts`
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-3-agent-1 -b run-3/agent-1-pdf-charts
cd ../worktrees/run-3-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

The new 200-positives experiment finished and dumped its summary at runs/active/ig_examples_200pos/attribution_metrics/summary.csv. Generate the professor-facing artifacts:

1. A standalone PDF table at professor_share/attribution_main_200pos_standalone.tex. Mirror the existing professor_share/attribution_main_200pair_standalone.tex — same layout, same metrics columns, same cross-bolding (better of {baseline, ABTT} per cell). Update the title to say "200 POSITIVE test pairs/model" and update the generation date. Update the per-cell ABTT-wins count and the bottom-line bullets honestly — if the story is now stronger or weaker than the 200-random version, say so plainly.

2. Refreshed charts: write professor_share/make_charts_200pos.py (mirror make_charts_200pair.py, point it at the new summary.csv). Generate chart_rho_loo_t5_200pos.{pdf,png} and chart_suff25_t5_200pos.{pdf,png}.

3. Use the cosine investigation finding from docs/analyses/cosine_inflation_investigation.md. Add a one-line note in the table caption about whether the new full_cos values are now sane (close to demo-range, not 0.95+ everywhere), or — if the investigation concluded the inflation was expected anisotropy — explain that briefly.

4. Compile the PDF (latexmk -pdf in professor_share/) and confirm it builds.

Every numeric value in the PDF must trace back to a row in the new summary.csv. Don't guess or extrapolate.

Commit the new tex, the new charts script, the generated PDF + chart files. Push the branch.
````

</details>

### Agent 3.2: Comprehensive HTML walkthrough
> **Area**: `professor_share/`
> **Worktree branch**: `run-3/agent-2-html-walkthrough`
> **Use /team**: Yes — split sub-agents per HTML section

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-3-agent-2 -b run-3/agent-2-html-walkthrough
cd ../worktrees/run-3-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

Write a comprehensive plain-English HTML walkthrough at professor_share/MEETING_PREP_2026_04_27.html that the professor can read end-to-end and walk away understanding everything. Pattern: similar look/feel to docs/meetings/POSTMEETING_2026_04_19_WALKTHROUGH.html — same MathJax setup for formulas, same readable typography.

Sections to cover (use /team to write these in parallel — one sub-agent per section):
- The role of the attribution experiment in the broader paper. Explicit framing: it's a SUPPORTING plank ("ABTT improves explanation quality"), not the headline. The headline is the Task A + Task B retrieval improvements, which already stand independently.
- What MaRC is (Brinner & Zarriess 2023) and how we adapted it for retrieval — the per-pair soft mask optimization, the cosine-similarity target replacing the classification logit, the optimization mechanics in plain language.
- What IG is (Sundararajan et al. 2017) and why we don't "adapt" it — IG is scalar-agnostic, we just use cosine as the scalar. Walk through the Layer Integrated Gradients mechanic at the embedding layer with the zero baseline. The four invocations per pair (query/candidate × baseline-cos/abtt-cos).
- Each metric defined precisely with PROPER attribution: Suff@25%, Comp@25%, Compactness@0.8 from ERASER (DeYoung et al. 2020); ρ_LOO closer to Atanasova et al. 2020. EXPLICITLY correct Ian's in-meeting misstatement that "all formulas come from the MaRC paper" — they don't.
- Why negative pairs don't fit these metrics. The "trivially win by masking everything" argument. Justify the move to positives-only.
- The cosine inflation investigation: read docs/analyses/cosine_inflation_investigation.md and summarize the finding in plain English. What was the bug (or non-bug)? What's the fix? How do we know it's actually fixed?
- The new 200-positives results. Read runs/active/ig_examples_200pos/attribution_metrics/summary.csv DIRECTLY — do not trust the PDF table for numbers. State per-cell what changed vs the 200-random version.
- Honest reading: does the experiment now support the paper's "ABTT improves explanation quality" claim? If yes, narrowly or broadly? If no, what's the fallback framing (e.g., "diagnostic showing which tokens drive the cosine")?

Quality bar: plain English, NO oversimplification. The professor is sophisticated — he wants to understand mechanism, not see a textbook intro. Inline the new charts (PNG paths from Agent 3.1). Every numeric claim must trace to a CSV row.

Self-contained HTML — MathJax via CDN is fine, but no external dependencies that would break offline.

Commit the HTML. Push the branch.
````

</details>

### Agent 3.3: Q&A prep doc
> **Area**: `professor_share/`
> **Worktree branch**: `run-3/agent-3-qa-prep`
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-3-agent-3 -b run-3/agent-3-qa-prep
cd ../worktrees/run-3-agent-3

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree — the merge agent handles that.

Write a tight Q&A prep doc at professor_share/QA_PREP_2026_04_27.md anticipating questions the professor is likely to ask Monday after reading the new 200-positives bundle. Each entry: the question in his voice, then a one-paragraph answer Ian can read off if asked.

Likely questions to cover (not exhaustive — add more as you think of them):
- "Why this attribution layer for each model? Did it match the retrieval pipeline?"
- "Did you pool the same way as the retrieval system, or did you mean-pool and the retrieval uses SIF?"
- "What does ρ_LOO=X actually mean for the paper claim?"
- "Why does ABTT not improve metric Y on cell Z?"
- "How confident are you in the cosine fix? Could the same bug be hiding elsewhere?"
- "What would you change for a v3 experiment if we had more time?"
- "Is MaRC pulling its weight as a method, or is IG the real story we should write up?"
- "The 200-random version told a softer story than 20-curated. Which set should we believe?"
- "Are negatives meaningfully gone, or did you just not look at them?"

Pull a few questions directly from Friday's meeting transcript (it's earlier in the conversation) — those are guaranteed re-asks.

Read first to keep your answers consistent: docs/analyses/cosine_inflation_investigation.md, docs/analyses/attribution_pipeline_audit.md, professor_share/MEETING_PREP_2026_04_27.html (if it landed before you started — if not, read its draft branch).

Commit the Q&A doc. Push the branch.
````

</details>

### Merge Agent 3.M: Integrate Run 3 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the three Run 3 branches into main and clean up worktrees.

1. git checkout main
2. Merge order (smallest first to surface conflicts early):
   - git merge run-3/agent-3-qa-prep              # single new doc
   - git merge run-3/agent-1-pdf-charts           # new tex + chart script + generated artifacts
   - git merge run-3/agent-2-html-walkthrough     # large HTML, may reference chart paths from 3.1
   - Resolve any conflicts preferring incoming changes.

3. Verify after merging:
   - professor_share/attribution_main_200pos_standalone.{tex,pdf} exist
   - professor_share/chart_rho_loo_t5_200pos.{pdf,png} and chart_suff25_t5_200pos.{pdf,png} exist
   - professor_share/MEETING_PREP_2026_04_27.html exists
   - professor_share/QA_PREP_2026_04_27.md exists

4. git worktree remove ../worktrees/run-3-agent-1
   git worktree remove ../worktrees/run-3-agent-2
   git worktree remove ../worktrees/run-3-agent-3
   git branch -d run-3/agent-1-pdf-charts run-3/agent-2-html-walkthrough run-3/agent-3-qa-prep
````

</details>

---

## Run 4: Final cross-check
> **Prerequisite**: Run 3 merged.
> **Agents**: 1 sequential
> **Worktrees**: No

### Agent 4.1: Cross-check bundle, write READY.md
> **Area**: `professor_share/`
> **Use /team**: No

<details>
<summary>Prompt (click to expand)</summary>

````
The professor-facing bundle is now in main under professor_share/. Cross-check everything before Ian sends it.

1. Read end-to-end: the new HTML walkthrough (MEETING_PREP_2026_04_27.html), the new PDF table source (attribution_main_200pos_standalone.tex), the chart script (make_charts_200pos.py), the Q&A doc (QA_PREP_2026_04_27.md), and the cosine investigation report (docs/analyses/cosine_inflation_investigation.md).

2. Verify every numeric claim in the HTML, PDF, and Q&A traces to a row in runs/active/ig_examples_200pos/attribution_metrics/summary.csv. Flag any discrepancy in your final report.

3. Verify the HTML opens cleanly in a browser context: no broken image links (chart PNGs resolve), MathJax CDN loads without error, no dead anchors.

4. Recompile the standalone PDF (cd professor_share && latexmk -pdf attribution_main_200pos_standalone.tex). Must succeed.

5. Verify the cosine investigation's claimed fix (if any) is actually present in the code on main, and that the new full_cos values in summary.csv look sane (consistent with the demo range, not uniformly inflated like the 200-random version).

6. Write professor_share/READY.md — a one-page pre-meeting summary for Ian: what's in the bundle, what fix landed, what story to lead with on Monday, what to flag as still-uncertain, what NOT to overclaim. This is the doc Ian reads on the way to the meeting.

If anything fails verification, do NOT write READY.md as if everything is fine. Report what's broken so Ian can fix it before the meeting.
````

</details>

---

## Post-Completion Checklist

- [ ] Run 1 merged: cosine investigation + pipeline audit + new positives sampler/sbatch all in main
- [ ] Run 2 complete: 200-positives experiment ran successfully, summary.csv exists, email notification fired
- [ ] Run 3 merged: PDF + charts + HTML + Q&A all in professor_share/
- [ ] Run 4 complete: READY.md exists, all numbers cross-checked, HTML and PDF render cleanly
- [ ] Ian has read READY.md before the Monday meeting
- [ ] No orphan worktrees, all branches deleted after merge
- [ ] GPU budget: only Run 2 uses GPU (~3-4h A100). Stay under remaining ~7-11h.

## Critical files at end-state

- `professor_share/MEETING_PREP_2026_04_27.html` — the main thing Ian sends/walks through
- `professor_share/attribution_main_200pos_standalone.pdf` — refreshed table
- `professor_share/chart_rho_loo_t5_200pos.png`, `chart_suff25_t5_200pos.png` — refreshed charts
- `professor_share/QA_PREP_2026_04_27.md` — Ian's prep notes
- `professor_share/READY.md` — pre-meeting summary
- `docs/analyses/cosine_inflation_investigation.md` — root cause + fix
- `docs/analyses/attribution_pipeline_audit.md` — bug audit findings
- `runs/active/ig_examples_200pos/attribution_metrics/summary.csv` — source of truth

## Risks & escalation paths

- **Cosine investigation finds a fundamental design problem.** Run 1 escalates to Ian via the merge agent's final message. Run 2 should NOT proceed; Ian decides whether to redesign or run anyway with caveats.
- **Audit finds a critical bug that invalidates prior numbers.** Same path — surface in merge 1.M output.
- **GPU job fails or times out.** Cron in 2.1 fires email; Ian decides whether to retry or downscope (e.g., 100 positives instead of 200).
- **Numbers come back even worse than the 200-random version.** Run 3's HTML walkthrough must say so honestly. Don't dress up bad numbers. The fallback framing is "attribution as diagnostic, not as proof of explanation quality" — the HTML should land that softly if needed.
