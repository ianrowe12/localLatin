# Gameplan: Meeting Action Items (2026-03-31)

Generated: 2026-04-04

## Overview
- **Total runs**: 2
- **Total agents**: 6
- **Estimated parallelism**: Run 1 has 3 agents in parallel, Run 2 has 3 agents in parallel
- **Key dependencies**: Run 2 agents depend on Run 1 outputs (new split code, new attribution methods, UI polish)

## Dependency Graph

```
Run 1 (parallel):
  Agent 1.1: Task B Split Redesign ─────────────────> Agent 2.1: M-Seed Evaluation Framework
  Agent 1.2: DLA + Attention Score Methods ─────────> Agent 2.2: Regen Plots + Leiden Examples
  Agent 1.3: Web App UI Polish + Reviewer Session ──> Agent 2.3: Web Auto-highlight + Deployment
```

All Run 1 agents touch independent files. All Run 2 agents depend on their respective Run 1 agent.

## Non-Code Action Item

**Find undergrad researcher**: NSF REU funding available for 2 students. Must be US national or permanent resident. Any year. Goal: research + paper. Reach out ASAP. Caitlyn already on team; need one more.

---

## Run 1: Core Code Changes
> **Prerequisite**: None (first run)
> **Agents**: 3 agents in parallel
> **Why parallel**: Each agent touches completely independent files (split logic vs attribution scripts vs webapp)

### Agent 1.1: Task B Split Redesign + Positive Pair Verification
> **Scope**: `src/`, `scripts/`
> **Key files**: `src/canon_split_v2.py`, `scripts/run_resubmit_data_prep.py`, `scripts/evaluate_vectors.py`, `scripts/run_phase_resubmit_taskb_topk.py`

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to redesign the Task B evaluation split for our Latin manuscript retrieval project. See CLAUDE.md for full project context.

## Background

Our professor identified a fundamental flaw in how we create Task B evaluation data. Currently, for a directory with N files in the test set, ALL N files can be queries and the reference set contains near-duplicate versions of the same directory. This inflates metrics and creates confusing duplicate-directory comparisons.

## What to do

### Part 1: Redesign the split logic

Modify `src/canon_split_v2.py` to implement a new split paradigm where files are either **queries** OR **reference directory members**, not both:

**New split rules (professor's instructions):**
1. **Singleton directories (1 file)**: 50% of the time, take the file out as a query (it should match to "none" — a new directory). 50% of the time, keep it in the reference set as a negative for other queries.
2. **Directories with 2 files**: Take 1 out as query, keep 1 as reference.
3. **Directories with 3 files**: Take 1 out as query, keep 2 as reference.
4. **Directories with 4+ files**: Take ~50% out as queries, keep the rest as reference. E.g., 8 files → take 4-5 out.
5. **Only ONE query set per directory per experiment** — no permutations. Once you decide which files are queries, that's locked.
6. Accept a `--seed` parameter so the experiment can be repeated M times with different random seeds.

The key difference from current code: currently `canon_train_test_split_v2()` splits files into train/test but ALL test files from the same folder can query against each other. The new approach splits into **query_files** (to be matched) and **reference_directories** (each directory has its remaining files as the reference). A query file compares against ALL reference directories to find its best match.

**Critical**: Each reference directory appears exactly ONCE. If a directory had 4 files and 2 were taken as queries, the reference directory contains only the 2 remaining files.

### Part 2: Update the evaluation

Modify `scripts/evaluate_vectors.py` to support the new paradigm:
- Instead of computing a file-to-file similarity matrix within the test set, compute a **query-to-directory** similarity matrix.
- For each query, its similarity to a reference directory = max cosine similarity between the query embedding and any file embedding in that directory.
- **Acc@K (K=1..5)**: For each query, rank all reference directories by similarity. Is the correct directory in the top K?
- **Assignment accuracy**: Use a threshold τ (learned on train). If max similarity < τ, the query is classified as "new" (no matching directory). If ≥ τ, the top-1 directory is the prediction.
- **"None" queries**: Queries from singleton directories that were taken out should have NO correct directory in the reference set. They are correct if classified as "new" (max sim < τ).

### Part 3: M-seed repetition framework

Create a wrapper script `scripts/run_taskb_mseed.py` that:
1. Runs the full Task B evaluation M times (default M=5) with seeds 42, 43, 44, 45, 46
2. For each seed: generates the split, runs evaluation across all models, collects results
3. Aggregates results: mean ± std for each metric across M runs
4. Outputs a summary CSV with per-seed and aggregated results

### Part 4: Verify positive pair count

The professor noted that Task A has only 354 positive pairs, which seems too low. Verify the computation in `src/canon_split_v2.py` `generate_pairs_tsv()`. For a directory with N files in the same split, there should be N*(N-1)/2 positive pairs. Directories with 3+ files should contribute significantly more than 1 pair each. Check if the current v2 split (which keeps doubletons together) is correctly counting all combinatorial pairs. Log the per-directory pair counts to help debug.

### Part 5: Update Task B top-K script

Update `scripts/run_phase_resubmit_taskb_topk.py` to work with the new query-vs-directory paradigm instead of the current file-vs-file approach. The cumulative top-K table should reflect the new evaluation.

## Scope
- Files you SHOULD modify: `src/canon_split_v2.py`, `scripts/run_resubmit_data_prep.py`, `scripts/evaluate_vectors.py`, `scripts/run_phase_resubmit_taskb_topk.py`
- Files you SHOULD create: `scripts/run_taskb_mseed.py`
- Files you should NOT modify: anything in `web/`, `src/sif_abtt.py`, `src/canon_retrieval.py` (use existing functions from canon_retrieval.py)

## Technical details

Current dataset (canon_labelled): 840 directories, 1,705 files. Of these:
- 545 singleton directories (1 file each)
- 108 doubleton directories (2 files each = 216 files)
- 187 multi-file directories (3+ files = 944 files)

The current v2 split in `src/canon_split_v2.py` keeps doubletons together (whole folder to train or test). The new split should NOT separate into train/test — instead it separates into **queries** and **reference set** from ALL files.

IMPORTANT: We still need a train/test split for SIF token probabilities and ABTT fitting (leak-free protocol). The query/reference split should operate WITHIN the test set. The train set is used only for fitting EmbeddingCleaner and learning threshold τ.

## When done
- Run `python scripts/run_resubmit_data_prep.py --canon_root canon_labelled --seed 42` to verify the new split generates correctly
- Print summary stats: number of queries, number of reference directories, number of "none" queries
- Verify positive pair counts are higher than 354
- Commit with a descriptive message
````

</details>

### Agent 1.2: DLA + Standalone Attention Score Attribution Methods
> **Scope**: `src/`, `scripts/`
> **Key files**: `scripts/run_resubmit_ig_comparison.py`, new `src/direct_logit_attribution.py`

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to implement two new token attribution methods for our Latin manuscript retrieval project. See CLAUDE.md for full project context.

## Background

We have a comparison script `scripts/run_resubmit_ig_comparison.py` that currently compares 4 attribution methods on token-to-token pair matrices:
- Panel A: IG + ABTT (integrated gradients, current baseline)
- Panel B: BERTScore greedy alignment
- Panel C: Optimal Transport / EMD
- Panel D: Attention-weighted cross-similarity (Ditto-inspired)

The professor wants us to add 2 more methods: **Direct Logit Attribution (DLA)** and **standalone Attention Score**. Both should be integrated into the existing comparison framework.

## What to do

### Method 1: Direct Logit Attribution (DLA)

Create `src/direct_logit_attribution.py` with the DLA implementation.

DLA is a mechanistic interpretability technique. Unlike IG (which requires ~50 gradient forward passes), DLA is a single-pass geometric decomposition. For retrieval, it measures how much each token's hidden-state vector contributes to the final pooled similarity.

**Implementation:**

For a query token at position `i` with hidden state `h_i` at layer L:

    # Baseline DLA: contribution to raw cosine similarity
    candidate_pooled = mean(candidate_hidden_states)  # pooled candidate embedding
    direction = candidate_pooled / norm(candidate_pooled)  # unit direction
    dla_score_i = h_i @ direction / N  # N = number of query tokens

    # For pair matrix (token-to-token):
    # DLA_ij = how much query token i aligns with candidate token j's representation
    dla_matrix[i,j] = cosine(h_i, c_j) * norm(h_i) * norm(c_j) / (N_q * N_c)

For the ABTT variant: apply ABTT cleaning to hidden states first, then compute DLA on cleaned representations.

The key advantage over IG: DLA measures *geometric alignment* (which tokens point in the same direction as the target), while IG measures *causal contribution* (which tokens, if removed, change the output). DLA is cheaper and more interpretable for mechanistic analysis.

**Integration into comparison script:**

In `scripts/run_resubmit_ig_comparison.py`, add a function `build_dla_pair_matrix(q_hidden_clean, c_hidden_clean)` that:
- Computes cosine similarity matrix between query and candidate token embeddings
- Weights by the geometric mean of token norms (tokens with larger norms contribute more)
- Normalizes weights to [0, 1]
- Returns `cos * weight`

### Method 2: Standalone Attention Score

Currently, attention is only used as *weighting* in the Ditto-inspired cross-similarity (Panel D). The standalone version uses attention more directly.

Add a function `build_attention_standalone(q_hidden_clean, c_hidden_clean, q_attention, c_attention)` that:
- Computes cosine similarity matrix between query and candidate tokens
- Computes mean attention received per token (column mean of attention matrix, excluding self-loops) as per-token importance
- Normalizes importance to a probability distribution
- Builds weight matrix as geometric mean of query and candidate importance scores
- Returns `cos * weight`

The difference from Panel D (Ditto): Panel D uses the self-attention *diagonal* (how much a token attends to itself). This standalone method uses the attention *column mean* (how much other tokens attend to this token). Tokens that are "important" in the attention economy get higher weight.

### Expand the comparison framework

1. Update `render_comparison()` in `scripts/run_resubmit_ig_comparison.py` from a 2x2 grid to a 2x3 grid (6 panels):
   - Panel A: IG + ABTT (existing)
   - Panel B: BERTScore greedy (existing)
   - Panel C: Optimal Transport (existing)
   - Panel D: Attention cross-sim / Ditto (existing)
   - Panel E: DLA (new)
   - Panel F: Standalone Attention Score (new)

2. Update the metrics computation to include the two new methods in `comparison_metrics.csv`.

3. Add **top-K highlighting** to ALL panels: in each heatmap, draw small markers (red squares or circles) around the top K=5 cells by absolute value. This makes it easy to see at a glance which token pairs each method considers most important.

## Scope
- Files you SHOULD create: `src/direct_logit_attribution.py`
- Files you SHOULD modify: `scripts/run_resubmit_ig_comparison.py`
- Files you should NOT modify: anything in `web/`, split/evaluation scripts, `src/canon_retrieval.py`

## Technical details

The NPZ artifacts at `runs/phase12f_examples/artifacts/*.npz` contain everything needed:
- `query_hidden` (shape: seq_len x hidden_dim) — per-token hidden states
- `candidate_hidden` — same for candidate
- `query_ig_abtt`, `candidate_ig_abtt` — IG attribution scores
- `query_attention_diag`, `candidate_attention_diag` — self-attention diagonals
- `cosine_sim` — pre-computed dense cosine matrix
- `pcs`, `mean_vec` — for ABTT cleaning

For the standalone attention method, you'll need the full attention matrix, not just the diagonal. Check if the NPZ files contain it. If not, you may need to:
1. Check what's in the NPZ files (read one and list its keys)
2. If full attention isn't available, use the diagonal as a proxy (sum of attention = 1, diagonal approximation)

## When done
- Run the comparison script on a few examples to verify 6-panel output
- Check that DLA and Attention Score panels produce meaningfully different patterns from existing methods
- Verify comparison_metrics.csv has 6 rows (one per method)
- Commit with a descriptive message
````

</details>

### Agent 1.3: Web App UI Polish + Reviewer Session
> **Scope**: `web/`
> **Key files**: `web/frontend/src/components/layout/`, `web/frontend/src/components/feedback/`, `web/frontend/src/contexts/`, `web/frontend/src/App.tsx`

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to improve the web application UI for our Latin manuscript review tool. See CLAUDE.md for full project context. The webapp is at `web/` (FastAPI backend + React/TypeScript/Vite frontend).

## Background

Our professor reviewed the webapp and said it looks good but needs labels and headings for non-tech-savvy Latin scholars. He also wants the reviewer name to be set once at the beginning (session-level) rather than entered per review.

## What to do

### Part 1: Add labels and headings throughout the UI

The professor said: "not tech savvy, we need to tell everything." Add clear, descriptive headings and labels:

1. **Left Sidebar** (`web/frontend/src/components/layout/LeftSidebar.tsx` or equivalent):
   - Add heading: "Unlabeled Manuscripts" above the query list
   - Add brief subtitle: "Select a manuscript to review predictions"

2. **Right Sidebar** (prediction panel area):
   - Add section label "Model Selection" above the model dropdown
   - Add section label "Predicted Sources (ranked by similarity)" above the prediction cards
   - Add section label "Your Assessment" above the feedback panel

3. **Document panels** (center area):
   - Query side: Add subtitle "Unlabeled Manuscript" below the filename
   - Candidate side: Add subtitle "Predicted Source (Rank N)" with the similarity score displayed clearly, like "Similarity: 0.847"

4. **Header** (`web/frontend/src/components/layout/Header.tsx`):
   - Add a subtitle line under the app name: "Review model predictions for Latin manuscript source identification"

5. **Prediction cards** (`web/frontend/src/components/predictions/PredictionCard.tsx` or similar):
   - Ensure rank number and similarity score are clearly labeled (not just raw numbers)

Keep styling consistent with existing Tailwind classes. Use muted/secondary text colors for subtitles. Don't add emojis.

### Part 2: Reviewer login / session persistence

Currently, the reviewer name is entered per-review in the feedback panel. Change this to a session-level login:

1. **Create `web/frontend/src/contexts/ReviewerContext.tsx`**:
   - Store `reviewerName: string | null` in React context + localStorage (key: `locallatin-reviewer`)
   - Provide `setReviewerName(name)` and `clearReviewer()` functions
   - On first load, check localStorage; if no reviewer set, show login modal

2. **Create `web/frontend/src/components/auth/ReviewerLoginModal.tsx`**:
   - Simple modal: text input for name + "Start Reviewing" button
   - Cannot be dismissed without entering a name (blocks the main UI)
   - Clean, minimal design consistent with the rest of the app

3. **Update `web/frontend/src/App.tsx`**:
   - Wrap with `ReviewerProvider`
   - Conditionally render `ReviewerLoginModal` when no reviewer is set

4. **Update feedback panel** (`web/frontend/src/components/feedback/FeedbackPanel.tsx`):
   - Remove the inline reviewer name input field
   - Show a read-only display: "Reviewing as: [Name]" with a small "change" link
   - Pre-populate the reviewer field in feedback drafts from the session context

5. **Show reviewer in header**:
   - Display "Reviewing as: [Name]" in the header bar (right side, near theme toggle)
   - Small initial circle/badge with first letter of name

## Scope
- Files you SHOULD modify: components in `web/frontend/src/components/`, `web/frontend/src/contexts/`, `web/frontend/src/App.tsx`
- Files you SHOULD create: `web/frontend/src/contexts/ReviewerContext.tsx`, `web/frontend/src/components/auth/ReviewerLoginModal.tsx`
- Files you should NOT modify: anything outside `web/`, no backend changes needed for this task

## Technical details

The frontend uses:
- React 18 + TypeScript + Vite
- Tailwind CSS 3 with dark mode support
- Framer Motion for animations
- Existing contexts: `AppContext`, `TokenContext`, `FeedbackContext`
- localStorage keys already in use: `locallatin-theme`, `locallatin-feedback-drafts`

The feedback panel currently has fields for: rank selection (pill buttons), notes textarea, reviewer name input, submit button. The reviewer name input should be removed and replaced with a read-only display pulling from the new ReviewerContext.

## When done
- Verify the app builds: `cd web/frontend && npm run build`
- Visually check that headings appear in left sidebar, right sidebar, document panels, and header
- Verify the login modal appears on first load (clear localStorage to test)
- Verify reviewer name persists across page refreshes
- Verify feedback submissions use the session-level reviewer name
- Commit with a descriptive message
````

</details>

---

## Run 2: Integration, Evaluation & Deployment
> **Prerequisite**: Run 1 complete (new split code from 1.1, new attribution methods from 1.2, UI polish from 1.3)
> **Agents**: 3 agents in parallel

### Agent 2.1: M-Seed Evaluation Framework + SLURM Jobs
> **Scope**: `scripts/`, `slurm/`
> **Key files**: `scripts/run_taskb_mseed.py` (created in Run 1), `scripts/evaluate_vectors.py`, `slurm/`

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to create the SLURM infrastructure and run the new Task B evaluation with M-seed repetition. See CLAUDE.md for full project context.

## Background

In Run 1, Agent 1.1 redesigned the Task B split in `src/canon_split_v2.py` and created `scripts/run_taskb_mseed.py` for M-seed repetition. Now we need to:
1. Create SLURM batch scripts to run the M-seed evaluation on the HPC cluster
2. Verify the new evaluation produces reasonable results
3. Generate summary tables with mean ± std across M=5 seeds

## What to do

### Part 1: Create SLURM scripts

Create `slurm/resubmit_taskb_mseed.sbatch`:
- Account: `beto-delta-gpu`, partition: `gpuA100x4`
- Activate conda env: `conda run -n localLatin python ...`
- Run `scripts/run_taskb_mseed.py` with M=5 seeds for all 6 models
- Each model's embeddings are already extracted at `runs/phase_resubmit/bases/<model_slug>/hidden_mean_tokempty/`
- Output to `runs/phase_resubmit/taskb_mseed/`

### Part 2: Create summary visualization

Create `scripts/visualize_taskb_mseed.py` that reads the M-seed results and generates:
1. A paper-ready table (LaTeX + PNG) showing mean ± std for each model at K=1,2,3,5
2. A bar chart with error bars comparing models at each K value
3. A breakdown showing "existing accuracy" vs "new file detection accuracy" vs "overall assignment accuracy"

### Part 3: Update the meeting summary figures

After results are available, the figures in `overleaf_drafts/figures/` will need to be regenerated with the new Task B numbers. Create a script or update `scripts/visualize_phase_resubmit.py` to include the M-seed aggregated Task B results.

## Context from Run 1

Agent 1.1 modified the split logic so that:
- Files are split into **queries** and **reference directories** (not just train/test)
- Each directory appears exactly once in the reference set
- Singleton directories are 50% query / 50% reference
- The experiment is repeated M=5 times with different seeds
- Results are aggregated as mean ± std

Check `scripts/run_taskb_mseed.py` to understand the output format before creating visualization scripts.

## Scope
- Files you SHOULD create: `slurm/resubmit_taskb_mseed.sbatch`, `scripts/visualize_taskb_mseed.py`
- Files you SHOULD modify: `scripts/visualize_phase_resubmit.py` (add M-seed Task B figures)
- Files you should NOT modify: `src/`, `web/`

## When done
- Verify SLURM script has correct module loads, conda activation, and path setup
- Dry-run the M-seed script locally with M=1 and a single model to check output format
- Commit with a descriptive message
````

</details>

### Agent 2.2: Regenerate Attribution Plots + Leiden Demo Examples
> **Scope**: `scripts/`, `runs/`
> **Key files**: `scripts/run_resubmit_ig_comparison.py` (modified in Run 1), `scripts/run_leiden_examples.py` (new)

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to regenerate attribution comparison plots and prepare demo examples for our Leiden collaborators. See CLAUDE.md for full project context.

## Background

In Run 1, Agent 1.2 added DLA and standalone Attention Score methods to `scripts/run_resubmit_ig_comparison.py`, expanding from 4 to 6 panels. Now we need to:
1. Regenerate all comparison plots with the 6-method comparison
2. Prepare a curated set of ~10 example pairs for the Leiden meeting
3. The professor wants slightly lower sparsity than BERTScore (not just 1-2 tokens, but a few top tokens highlighted) — the top-K highlighting added in Run 1 addresses this

## What to do

### Part 1: Regenerate comparison plots

Run the updated `scripts/run_resubmit_ig_comparison.py` on all available examples from `runs/phase12f_examples/artifacts/`. This will regenerate:
- 6-panel comparison figures in `runs/phase_resubmit/ig_comparison/comparisons/`
- Detail heatmaps in `runs/phase_resubmit/ig_comparison/details/`
- Updated `comparison_metrics.csv` with all 6 methods

Create a SLURM script `slurm/resubmit_ig_comparison.sbatch` to run this on the cluster (it needs GPU for IG forward passes).

### Part 2: Create Leiden demo script

Create `scripts/run_leiden_examples.py` that:
1. Selects 10 diverse example pairs:
   - Mix of correct predictions and incorrect predictions
   - Diverse models (at least LaBSE, Qwen3-0.6B, PhilTa represented)
   - Mix of high-similarity and borderline pairs
2. For each pair, generates the 6-panel comparison figure
3. Creates a simple HTML summary report that can be shared with collaborators:
   - Each example shows: query filename, candidate filename, similarity score, model name
   - Below each: the 6-panel comparison figure (embedded as base64 or linked PNG)
   - Brief labels explaining what each panel shows
4. Output to `runs/phase_resubmit/leiden_demo/`

### Part 3: Create a comparison summary table

Generate a summary table comparing all 6 methods across the metrics:
- Sparsity (fraction of matrix above 5% of max)
- Content Focus (fraction of weight on content-token pairs)
- Shared Token Match (fraction of weight on exact-match tokens)
- Add a new metric: **Top-5 Precision** — of the top 5 highlighted pairs, how many are same-token or content-word pairs?

Output as both CSV and a paper-ready LaTeX table.

## Context from Run 1

Agent 1.2 added to `scripts/run_resubmit_ig_comparison.py`:
- `build_dla_pair_matrix()` — Direct Logit Attribution
- `build_attention_standalone()` — Standalone Attention Score
- 2x3 grid rendering (6 panels)
- Top-K=5 highlighting markers on all panels

Check the updated script to understand the new function signatures before creating the Leiden demo script.

## Scope
- Files you SHOULD create: `scripts/run_leiden_examples.py`, `slurm/resubmit_ig_comparison.sbatch`
- Files you MAY modify: `scripts/run_resubmit_ig_comparison.py` (only if needed for summary table generation)
- Files you should NOT modify: `src/`, `web/`, split/evaluation scripts

## When done
- Verify SLURM script has correct setup
- Dry-run the Leiden examples script on 2-3 examples to check output
- Verify HTML report renders correctly
- Commit with a descriptive message
````

</details>

### Agent 2.3: Web App Auto-Highlight + Deployment Preparation
> **Scope**: `web/`, `deploy/`
> **Key files**: `web/services/token_map_svc.py`, `web/frontend/src/contexts/TokenContext.tsx`, `web/config.py`, deployment configs

<details>
<summary>Prompt (click to expand)</summary>

````
I need you to add auto-highlighting of top tokens and prepare the webapp for deployment. See CLAUDE.md for full project context. The webapp is at `web/` (FastAPI backend + React/TypeScript/Vite frontend).

## Background

The professor wants top K=5 tokens pre-highlighted when a pair is loaded, instead of requiring manual hover/click. He also wants the app deployed to the `ai.csr.uky.edu` VM for the Leiden collaborators to use.

Run 1 Agent 1.3 already added labels, headings, and reviewer session login. This agent adds auto-highlighting and deployment infrastructure.

## What to do

### Part 1: Backend — compute auto-highlight tokens

Modify `web/services/token_map_svc.py`:
- After computing token similarity matrices, identify the top K=5 query tokens by absolute IG score (`query_ig_abtt`)
- For each top query token, find its top 2 candidate matches by cosine similarity
- Add an `auto_highlights` field to the response: a dict mapping query token index to a list of candidate_idx/score objects

Modify `web/models.py`:
- Add `auto_highlights: Optional[Dict[str, List[dict]]]` to the `TokenMapResponse` model

### Part 2: Frontend — auto-pin tokens on load

Modify `web/frontend/src/contexts/TokenContext.tsx`:
- Add state: `autoHighlightedTokens: Set<number>` (to distinguish auto-pins from manual pins)
- Add action: `setAutoHighlights(highlights: Map<number, {candidateIdx: number, score: number}[]>)`
- When auto-highlights arrive, automatically populate `pinnedTokens` with these entries

Modify the document panel component (find the right file in `web/frontend/src/components/document/`):
- On mount / when token map data loads, if `auto_highlights` exist and no tokens are manually pinned, call `pinToken()` for the top K query tokens
- Add a small "Clear highlights" button or "Auto" badge to indicate these were auto-selected

Modify `web/frontend/src/components/document/TokenSpan.tsx`:
- Add visual distinction for auto-highlighted tokens (e.g., dashed ring or softer opacity) vs manually pinned ones (solid ring)

### Part 3: Deployment preparation for ai.csr.uky.edu

1. **Create `web/config.production.yaml`**:
   - Update `cors.allow_origins` to include `https://ai.csr.uky.edu`
   - Set paths appropriate for the target VM
   - Set `app.debug: false`
   - Point `feedback_db` to a persistent location

2. **Create `deploy/locallatin.service`** (systemd unit file) with:
   - Description: LocalLatin Manuscript Review
   - After: network.target
   - User: irowerojas
   - WorkingDirectory pointing to the repo
   - ExecStart: python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000
   - Restart=on-failure
   - Environment variable LOCALLATIN_CONFIG=web/config.production.yaml

3. **Create `deploy/nginx.conf`** (reverse proxy snippet):
   - Proxy `/api/` to `http://127.0.0.1:8000`
   - Serve built frontend from `web/static/`

4. **Create `deploy/deploy.sh`** (build + deploy script):
   - Build frontend: `cd web/frontend && npm ci && npm run build`
   - Copy `dist/` to `web/static/`
   - Restart systemd service

5. **Update `web/frontend/vite.config.ts`**:
   - Make the build output directory configurable (default: `../static`)
   - Add `VITE_API_BASE` environment variable support for production API URL

6. **Update `web/config.py`**:
   - Support `LOCALLATIN_CONFIG` environment variable to specify config file path
   - Fall back to `web/config.yaml` if not set

## Context from Run 1

Agent 1.3 already added:
- Labels and headings throughout the UI
- ReviewerContext with localStorage persistence
- ReviewerLoginModal
- Session-level reviewer name in header and feedback panel

Check `web/frontend/src/App.tsx` to see where ReviewerProvider was added and follow the same pattern for any new providers.

## Scope
- Files you SHOULD modify: `web/services/token_map_svc.py`, `web/models.py`, `web/config.py`, `web/frontend/src/contexts/TokenContext.tsx`, `web/frontend/src/components/document/TokenSpan.tsx`, `web/frontend/vite.config.ts`
- Files you SHOULD create: `web/config.production.yaml`, `deploy/locallatin.service`, `deploy/nginx.conf`, `deploy/deploy.sh`
- Files you should NOT modify: anything outside `web/` and `deploy/`, split/evaluation scripts

## Technical details

The token map NPZ files contain:
- `query_ig_abtt` (shape: seq_len) — IG attribution scores per query token
- `cosine_sim` (shape: n_q x n_c) — pre-computed token-to-token cosine matrix

The current `token_map_svc.py` already loads these and computes `top_matches`. The auto-highlight computation is similar but selects the globally top-K most important query tokens instead of returning matches for every token.

Database persistence is critical — the professor emphasized: "whatever they do, must stay there." The SQLite file at `feedback.db` must be in a backed-up location, not a temp directory.

## When done
- Verify the app builds: `cd web/frontend && npm run build`
- Test auto-highlighting works with mock data
- Verify deployment configs are syntactically correct
- Verify `deploy/deploy.sh` is executable and correctly ordered
- Commit with a descriptive message
````

</details>

---

## Post-Completion Checklist
- [ ] **Run 1 agents completed** — all 3 committed independently
- [ ] **No merge conflicts** between agent outputs (they touch independent files)
- [ ] **Run 2 agents completed** — all 3 committed independently
- [ ] **Task B evaluation**: New split generates correct query/reference partition
- [ ] **Positive pairs**: Count verified to be higher than 354
- [ ] **Attribution methods**: 6-panel comparison renders correctly
- [ ] **Leiden demo**: 10 example pairs with HTML report ready
- [ ] **Web app builds**: `cd web/frontend && npm run build` succeeds
- [ ] **Web app features**: Labels visible, login modal works, auto-highlight works
- [ ] **Deployment**: Config files ready for ai.csr.uky.edu
- [ ] **M-seed results**: 5 seeds × 6 models evaluation complete with mean ± std
- [ ] **Figures updated**: Paper figures regenerated with new Task B numbers
- [ ] **Find undergrad**: Reached out to potential candidates (US national/PR, any year)
- [ ] Changes committed and pushed
