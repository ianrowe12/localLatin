# Orchestra: Layer Selection and Attribution Evidence

Generated: 2026-04-29

## Overview
- **Total runs**: 5
- **Total agents**: 11 agents plus 3 merge agents
- **Estimated parallelism**: Run 1 has 3 agents in parallel, Run 2 has 1 agent, Run 3 has 2 agents in parallel, Run 4 has 3 agents in parallel, Run 5 has 2 agents in parallel
- **Key dependencies**: Layer-selection evidence must come before mT5 attribution, because the professor's main concern is avoiding reverse-engineered layer choices. Attribution pipeline support must be in place before expensive reruns. Reporting and paper-ready synthesis come after results are merged.

Important preflight before launching any agent: this repository currently has uncommitted work. Worktrees created from a branch will not inherit uncommitted edits in the main checkout. Before running this orchestra, make a deliberate checkpoint commit or otherwise ensure the base branch contains the professor-share organization, current 200-positive attribution work, and cosine-fix changes that should be part of the starting point.

## Dependency Graph

```text
Run 1A layer diagnostics  \
Run 1B metric provenance   > Run 2 layer-rule decision > Run 3 attribution pipeline/results > Run 4 reporting > Run 5 synthesis
Run 1C mT5 readiness      /
```

Run 1 separates evidence gathering into independent tracks. Run 2 makes the layer-selection decision that controls which layers attribution should explain. Run 3 then extends and runs the attribution experiment for LaTa, PhilTa, and mT5. Run 4 turns results into paper-ready tables, charts, and methodological notes. Run 5 packages the final evidence for the later paper rewrite.

---

## Run 1: Establish Evidence and Readiness
> **Prerequisite**: Preflight checkpoint complete; current base branch contains the work to build on
> **Agents**: 3 agents in parallel
> **Worktrees**: Yes
> **Why parallel**: Layer diagnostics, citation provenance, and mT5 pipeline readiness touch different concerns and can proceed independently

### Agent 1.1: Layer-Selection Diagnostics
> **Area**: cross-cutting research pipeline
> **Worktree branch**: `run-1/agent-1-layer-diagnostics`
> **Use /team**: Yes - complex

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-1 -b run-1/agent-1-layer-diagnostics
cd ../worktrees/run-1-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

We need a defensible answer to the professor's question: why inspect intermediate layers, and can readers get guidance for choosing a layer without already knowing the final test metric? Build an evidence package comparing layer-level geometry diagnostics against retrieval outcomes for the main-paper models LaTa, PhilTa, and mT5, with appendix coverage for other models when existing artifacts make that cheap.

Focus on unsupervised or low-label diagnostics: anisotropy, top-PC dominance, effective rank, cosine concentration/spread, and ABTT-induced geometry changes. Supervised metrics such as AUROC and cosine gap are validation targets, not the primary claimed selector.

The output should be a clear memo and machine-readable tables under the project docs/runs area. It should recommend whether attribution should explain the retrieval-selected layer, the recovered-collapse diagnostic layer, or a two-stage rule. If no intrinsic diagnostic cleanly predicts best layers, say that and propose the fallback rule honestly.

Run lightweight checks that your tables regenerate and that any new scripts compile, for example `python -m py_compile` on changed Python files and a small dry-run/sample mode if you add one. Commit your work.
````

</details>

### Agent 1.2: Metric and Method Provenance
> **Area**: paper support
> **Worktree branch**: `run-1/agent-2-metric-provenance`
> **Use /team**: No - straightforward research

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-2 -b run-1/agent-2-metric-provenance
cd ../worktrees/run-1-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Prepare the citation and wording support for the attribution section. The paper needs to be precise about which parts are borrowed, which parts are adapted, and which parts are ours.

Cover Integrated Gradients as a scalar-function attribution method, MaRC/MarK as the learned-mask method being adapted from classification to retrieval cosine, ERASER-family sufficiency/comprehensiveness/compactness metrics, leave-one-out rank-correlation faithfulness adapted to cosine retrieval, and ABTT. Produce a concise memo with recommended citations, claim wording, and reviewer-risk notes.

Do not rewrite the full paper. Create patch-ready paragraphs and a bibliography checklist that the final rewrite can consume. Run whatever text/build checks are appropriate for files you touch, then commit.
````

</details>

### Agent 1.3: mT5 Attribution Readiness
> **Area**: attribution pipeline
> **Worktree branch**: `run-1/agent-3-mt5-readiness`
> **Use /team**: Yes - complex

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-1-agent-3 -b run-1/agent-3-mt5-readiness
cd ../worktrees/run-1-agent-3

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

mT5 must join the attribution experiment, but it is not currently part of the 200-positive attribution bundle. Make the attribution pipeline ready to include mT5 once Run 2 decides the attribution layer rule.

This is pipeline readiness, not the expensive final run. Ensure the positive-pair sampler, PC refit path, model metadata, artifact generation, MaRC sidecar flow, metric rendering, and method labels can handle mT5 consistently with LaTa and PhilTa. Preserve the existing LaTa/PhilTa behavior. Add dry-run or small-sample checks where useful so later agents can verify mT5 without launching the full experiment.

Run compile checks and any cheap dry runs you add, then commit.
````

</details>

### Merge Agent 1.M: Integrate Run 1 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the parallel branches from Run 1 into the base branch and clean up worktrees.

1. Confirm the base branch is the intended checkpoint branch.
2. For each branch, merge in this order:
   - git merge run-1/agent-2-metric-provenance
   - git merge run-1/agent-1-layer-diagnostics
   - git merge run-1/agent-3-mt5-readiness
3. After each merge, resolve conflicts carefully without discarding existing user work. Run:
   - python -m py_compile src/attribution_metrics.py scripts/ig/run_attribution_metrics.py scripts/ig/sample_positive_test_pairs.py scripts/ig/refit_pcs_for_attribution.py
   - any new dry-run command introduced by the branch being merged
4. After all branches merge, run a final smoke check:
   - python scripts/ig/run_attribution_metrics.py --dry_run --examples_csv runs/active/ig_examples_200pos/positive200_examples.csv --artifacts_root runs/active/ig_examples_200pos/artifacts --out_root runs/active/ig_examples_200pos/attribution_metrics_smoke
5. Clean up:
   - git worktree remove ../worktrees/run-1-agent-1
   - git worktree remove ../worktrees/run-1-agent-2
   - git worktree remove ../worktrees/run-1-agent-3
   - git branch -d run-1/agent-1-layer-diagnostics run-1/agent-2-metric-provenance run-1/agent-3-mt5-readiness
6. Commit the integrated result if needed.
````

</details>

---

## Run 2: Decide the Layer Rule
> **Prerequisite**: Run 1 complete + merged (because attribution layers depend on the diagnostics and mT5 readiness)
> **Agents**: 1 agent
> **Worktrees**: No

### Agent 2.1: Layer-Rule Decision Memo
> **Area**: cross-cutting research synthesis
> **Worktree branch**: N/A - single agent run
> **Use /team**: No - synthesis task

<details>
<summary>Prompt (click to expand)</summary>

````
Read the Run 1 layer-diagnostic memo, the professor-share meeting materials, and the current retrieval/attribution artifacts. Decide the attribution layer rule for LaTa, PhilTa, and mT5.

The decision should explicitly compare two defensible choices: explaining the retrieval-selected layer versus explaining the recovered-collapse diagnostic layer. The output must name the selected layer for each of LaTa, PhilTa, and mT5, explain why the rule is not reverse-engineered from test attribution metrics, and state the fallback guidance if unsupervised diagnostics are imperfect.

This decision is a hard contract for Run 3. Write it as a short memo in the project docs/runs area and update any small config file or README that later attribution agents should read. Do not run the full attribution experiment here.

Run compile checks if you touch code and commit the decision.
````

</details>

---

## Run 3: Run the Three-Model Attribution Experiment
> **Prerequisite**: Run 2 complete + committed (because selected attribution layers are now fixed)
> **Agents**: 2 agents in parallel
> **Worktrees**: Yes
> **Why parallel**: One agent owns experiment generation and execution; the other owns metric-sweep support and validation. They coordinate through the Run 2 layer-rule contract.

### Agent 3.1: Generate Three-Model Attribution Artifacts
> **Area**: attribution pipeline
> **Worktree branch**: `run-3/agent-1-three-model-artifacts`
> **Use /team**: Yes - complex

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-3-agent-1 -b run-3/agent-1-three-model-artifacts
cd ../worktrees/run-3-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Using the layer-rule memo from Run 2, produce the 200-positive attribution artifacts for LaTa, PhilTa, and mT5. The paper needs the same positive-pair regime across all three main models, with IG and retrieval-adapted MaRC available under baseline and ABTT.

Preserve the existing LaTa/PhilTa results where they remain valid under the decided layer rule, and regenerate only what the rule requires. For mT5, make sure PCs are refit through the same pooling path as the metrics code before ABTT artifacts are trusted. Keep the outputs in a clearly named active run directory so later reporting agents can find them unambiguously.

Run the necessary pipeline checks, including small-sample validation before the full run when possible. Commit the scripts/configs and result manifests; large generated artifacts should follow the repository's existing convention.
````

</details>

### Agent 3.2: Metric Sweep Support and Validation
> **Area**: attribution metrics
> **Worktree branch**: `run-3/agent-2-metric-sweeps`
> **Use /team**: Yes - complex

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-3-agent-2 -b run-3/agent-2-metric-sweeps
cd ../worktrees/run-3-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

The attribution section needs a threshold sweep, not a single arbitrary Suff@25 and Cmpct@0.8 table. Extend the metric/reporting path so it can summarize Sufficiency and Comprehensiveness at 10, 25, and 50 percent, Compactness at 0.7, 0.8, 0.9, and 0.95, and rho_LOO as the primary ranking-faithfulness metric.

Keep the main-text threshold choice global across models and methods. Do not create model-specific thresholds. The appendix should expose the full sweep so the main table is defensible rather than cherry-picked.

Validate on a small subset first, then against the current 200-positive artifacts if available. Run compile checks and metric self-tests, then commit.
````

</details>

### Merge Agent 3.M: Integrate Run 3 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the parallel branches from Run 3 into the base branch and clean up worktrees.

1. Merge in this order:
   - git merge run-3/agent-2-metric-sweeps
   - git merge run-3/agent-1-three-model-artifacts
2. Resolve conflicts by preserving the metric-sweep interface and adapting the artifact-generation branch to it where needed.
3. Run:
   - python -m py_compile src/attribution_metrics.py scripts/ig/run_attribution_metrics.py scripts/ig/sample_positive_test_pairs.py scripts/ig/refit_pcs_for_attribution.py scripts/ig/run_retrieval_mark_pair_examples.py
   - python -m src.attribution_metrics
   - the small-sample attribution metric command documented by the agents
4. Confirm the final summary includes LaTa, PhilTa, and mT5, and includes IG, retrieval_mark, random, and inverse rows at minimum.
5. Clean up:
   - git worktree remove ../worktrees/run-3-agent-1
   - git worktree remove ../worktrees/run-3-agent-2
   - git branch -d run-3/agent-1-three-model-artifacts run-3/agent-2-metric-sweeps
6. Commit the integrated result if needed.
````

</details>

---

## Run 4: Turn Results Into Paper-Ready Artifacts
> **Prerequisite**: Run 3 complete + merged (because reporting depends on final attribution and sweep summaries)
> **Agents**: 3 agents in parallel
> **Worktrees**: Yes
> **Why parallel**: Tables/figures, interpretation memo, and appendix packaging use the same result set but touch mostly independent files

### Agent 4.1: Main-Text Tables and Figures
> **Area**: paper artifacts
> **Worktree branch**: `run-4/agent-1-main-artifacts`
> **Use /team**: No - focused reporting

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-4-agent-1 -b run-4/agent-1-main-artifacts
cd ../worktrees/run-4-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Create the main-text candidate attribution table/figure for LaTa, PhilTa, and mT5 using the globally selected thresholds from the metric sweep. The main text should foreground rho_LOO as the primary faithfulness signal, while showing the ERASER-style metrics compactly enough for the paper.

The artifacts should be easy to include in Overleaf and should not hide metric disagreements. If the conventional 25 percent / 0.8 thresholds are not the final choice, explain why the chosen global thresholds are more representative.

Rebuild any generated TeX/PNG/PDF artifacts you create and run the relevant script checks. Commit the reporting artifacts.
````

</details>

### Agent 4.2: Appendix Sweep Package
> **Area**: paper artifacts
> **Worktree branch**: `run-4/agent-2-appendix-sweeps`
> **Use /team**: No - focused reporting

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-4-agent-2 -b run-4/agent-2-appendix-sweeps
cd ../worktrees/run-4-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Package the full attribution hyperparameter sweep for the appendix. Readers should be able to see Sufficiency and Comprehensiveness across 10, 25, and 50 percent; Compactness across 0.7, 0.8, 0.9, and 0.95; and the corresponding rho_LOO result for the same model/method/variant cells.

Keep the appendix honest and readable: separate main-paper models from any extra methods or baselines, include clear directionality labels, and make threshold-dependent tradeoffs visible.

Run generation checks for any tables/figures and commit.
````

</details>

### Agent 4.3: Interpretation and Caveat Memo
> **Area**: research synthesis
> **Worktree branch**: `run-4/agent-3-interpretation-memo`
> **Use /team**: No - synthesis task

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-4-agent-3 -b run-4/agent-3-interpretation-memo
cd ../worktrees/run-4-agent-3

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Write the interpretation memo that will let us rewrite the paper after the orchestra finishes. It should answer: what did the layer-selection analysis show, what layer rule was used for attribution, did rho_LOO improve consistently for IG and MaRC, which ERASER-style metrics agree or disagree, and what caveats must stay in the main prose.

Treat IG and retrieval-adapted MaRC as equal attribution views. Explain MaRC's retrieval adaptation as a contribution, but do not make the empirical ABTT claim depend on MaRC alone. Include recommended paper wording and a short list of claims we should avoid.

No broad edit to the main LaTeX draft is needed. Commit the memo.
````

</details>

### Merge Agent 4.M: Integrate Run 4 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the parallel branches from Run 4 into the base branch and clean up worktrees.

1. Merge in this order:
   - git merge run-4/agent-3-interpretation-memo
   - git merge run-4/agent-2-appendix-sweeps
   - git merge run-4/agent-1-main-artifacts
2. Resolve conflicts without deleting generated outputs that later paper work needs.
3. Rebuild or verify the generated tables/figures with the commands documented by the agents.
4. If LaTeX artifacts were changed, run a focused LaTeX build or at minimum confirm the generated TeX files are syntactically valid.
5. Clean up:
   - git worktree remove ../worktrees/run-4-agent-1
   - git worktree remove ../worktrees/run-4-agent-2
   - git worktree remove ../worktrees/run-4-agent-3
   - git branch -d run-4/agent-1-main-artifacts run-4/agent-2-appendix-sweeps run-4/agent-3-interpretation-memo
6. Commit the integrated result if needed.
````

</details>

---

## Run 5: Final Synthesis for the Paper Rewrite
> **Prerequisite**: Run 4 complete + merged (because synthesis depends on final tables, figures, and memos)
> **Agents**: 2 agents in parallel
> **Worktrees**: Yes
> **Why parallel**: One agent prepares the paper rewrite brief; the other audits reproducibility and result consistency

### Agent 5.1: Paper Rewrite Brief
> **Area**: paper support
> **Worktree branch**: `run-5/agent-1-paper-brief`
> **Use /team**: No - synthesis task

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-5-agent-1 -b run-5/agent-1-paper-brief
cd ../worktrees/run-5-agent-1

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Prepare the final brief for rewriting the paper in a later chat. It should map evidence to paper sections: motivation for layerwise analysis, layer-selection guidance, ABTT repair story, attribution setup, MaRC retrieval adaptation, metric provenance, main attribution result, appendix support, and caveats.

Write this as patch-ready prose blocks and section-level instructions, not as a full direct rewrite of the main LaTeX file. Make clear what belongs in main text versus appendix.

Commit the brief.
````

</details>

### Agent 5.2: Reproducibility and Consistency Audit
> **Area**: research QA
> **Worktree branch**: `run-5/agent-2-repro-audit`
> **Use /team**: No - audit task

<details>
<summary>Prompt (click to expand)</summary>

````
Before starting, create an isolated worktree:
git worktree add ../worktrees/run-5-agent-2 -b run-5/agent-2-repro-audit
cd ../worktrees/run-5-agent-2

Work entirely in this worktree. When done, commit and push your branch. Do NOT merge or remove the worktree; the merge agent handles that.

Audit the final outputs for consistency before the paper rewrite. Check that every headline number in the main artifacts traces to a CSV/JSON source, the selected attribution layers match the Run 2 decision, metric directionality is correct, mT5 is included where intended, and older superseded 200-random or two-model-only artifacts are not accidentally presented as current.

Produce a concise audit report with pass/fail items and remaining risks. Fix small labeling or provenance mistakes if they are clearly local; otherwise flag them for the rewrite pass.

Run the verification commands needed to support the audit and commit.
````

</details>

### Merge Agent 5.M: Integrate Run 5 branches

<details>
<summary>Prompt (click to expand)</summary>

````
Merge the parallel branches from Run 5 into the base branch and clean up worktrees.

1. Merge in this order:
   - git merge run-5/agent-2-repro-audit
   - git merge run-5/agent-1-paper-brief
2. Resolve conflicts, preserving the audit findings and the final rewrite brief.
3. Run final checks:
   - python -m py_compile src/attribution_metrics.py scripts/ig/run_attribution_metrics.py scripts/ig/sample_positive_test_pairs.py scripts/ig/refit_pcs_for_attribution.py scripts/ig/run_retrieval_mark_pair_examples.py
   - any artifact-generation or LaTeX checks documented in Run 4
4. Clean up:
   - git worktree remove ../worktrees/run-5-agent-1
   - git worktree remove ../worktrees/run-5-agent-2
   - git branch -d run-5/agent-1-paper-brief run-5/agent-2-repro-audit
5. Commit the integrated result if needed.
````

</details>

---

## Post-Completion Checklist
- [ ] All runs completed
- [ ] All worktree branches merged and worktrees removed
- [ ] Tests passing in each service
- [ ] No merge conflicts between agent outputs
- [ ] Manual smoke test of key flows
- [ ] Deploy via CI/CD (push to main)
- [ ] Final rewrite brief and reproducibility audit are ready for the next paper-rewrite chat
