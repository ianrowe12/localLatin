# MaRC: Model Interpretability and Rationale Extraction by Input Mask Optimization

Survey notes prepared 2026-04-07 in response to the resource bundle Prof. Siddique
shared after the 2026-04-07 meeting (`docs/New resources April 7/`). His proposed
direction — learned attention masks for attribution — turns out to have been
already published as **MaRC**, so the goal of this note is to pin down what MaRC
actually does so we can later design a variant that is (a) novel relative to MaRC
and (b) compatible with our existing IG pipeline for Latin manuscript retrieval.

> **Two corrections to flag up front**, since both the meeting notes and the
> filename suggest otherwise:
>
> 1. **`docs/New resources April 7/github link.txt` is not empty** (the meeting
>    report claimed no public code existed). The file contains exactly one line:
>    `https://github.com/inas-argumentation/Explainability`. The same URL also
>    appears inside the paper's Appendix A.
> 2. **The paper is not a fresh August 2025 paper.** It is **Brinner & Zarrieß,
>    Findings of ACL 2023** (Toronto, pp. 13722–13744). The arXiv ID
>    `2508.11388` (uploaded 2025-08-15) is a late re-deposit of the existing
>    ACL Findings paper, not a new contribution.

---

## 1. Citation

- **Authors:** Marc Brinner and Sina Zarrieß (Bielefeld University, Faculty
  for Linguistics and Literary Studies).
- **Title:** *Model Interpretability and Rationale Extraction by Input Mask
  Optimization.*
- **Venue:** Findings of the Association for Computational Linguistics: ACL
  2023, pages 13722–13744. Toronto, Canada.
- **ACL Anthology:** <https://aclanthology.org/2023.findings-acl.867/>
- **arXiv (late re-deposit, 2025-08-15):** 2508.11388 —
  <https://arxiv.org/abs/2508.11388>
- **Source code (public, confirmed live 2026-04-07):**
  <https://github.com/inas-argumentation/Explainability>

The repository is Python, contains `run_movie_reviews.py` and `run_imagenet.py`
as the two entry points (text and image experiments), with subfolders
`movie_reviews/` and `imagenet/`. There is **no LICENSE file visible** as of
2026-04-07 — relevant if we ever want to copy code rather than re-implement.

---

## 2. Method Summary (≈260 words)

MaRC produces a per-instance, per-token, soft mask `λ ∈ [0,1]^n` that explains
why a frozen classifier predicts class `c` for input `x`. Rather than parameterise
`λ` directly, MaRC introduces two latent vectors per instance — a weight vector
`w ∈ ℝ^n` and a bandwidth vector `σ ∈ ℝ^n_{>0}` — and constructs `λ` from them
through an unnormalised Gaussian influence kernel followed by a sigmoid (Eq. 5–6
below). This Gaussian reparameterisation gives spatial smoothness "for free":
neighbouring tokens or pixels can only get very different mask values if the
optimiser actively shrinks `σ` for them. There is **no hard continuity
constraint**, unlike Fong & Vedaldi (2017).

The masked input is formed by **interpolation against an uninformative baseline**
`b` rather than by hard zeroing: `x̃ = λ·x + (1−λ)·b`. For BERT-style text the
baseline is a sequence of PAD tokens; for images it is a constant or blurred
background. A complementary input `x̃^c = (1−λ)·x + λ·b` is also constructed, and
the loss includes both: it pushes `x̃` to keep the original prediction
(*sufficiency*) while pushing `x̃^c` away from it (*comprehensiveness*). Two
regularisers shape `λ`: a squared-mean sparsity penalty on `λ` itself, and a
log-barrier on `σ` that biases the Gaussian kernels to stay broad until the
fidelity terms force them to localise.

Optimisation is per-instance Adam over `(w, σ)` with the underlying classifier
frozen; convergence requires hundreds of forward + backward passes per example
(~2–3 min per BERT-base sample, ~1 min per ImageNet image).

---

## 3. Core Equations (verified against the published PDF)

> Equation numbers and forms below were verified by extracting text directly
> from the ACL Findings 2023 PDF (`pymupdf`, 2026-04-07). The presentation
> below is faithful to the paper's labeling, but pages 4–5 introduce the
> objective in stages, so equations 2 and 4 are *intermediate* objectives that
> get superseded by equation 8 — this is sometimes glossed over in summaries.

**Eq. 1 — Masked input** (`λ ∈ [0,1]ⁿ`, `b` is an uninformative baseline):

```
x̃ = λ · x + (1 − λ) · b
```

**Eq. 2 — Sufficiency objective.** Find `λ` that keeps the model's score for
class `c` high while masking as much of the input as possible. The squared
mean of `λ` is the sparsity term, which the paper underbraces and labels
`Ω_λ`:

```
arg min       −L(x̃, c)  +  α_λ · [ (1/n) Σᵢ λᵢ ]²
λ∈[0,1]^n                  └──────────┬──────────┘
                                     Ω_λ
```

`L(x̃, c)` is the model's score for class `c` on the masked input — either
the log-likelihood of `c` (which suppresses other classes and yields
class-*discriminative* explanations) or the log-sigmoid of the logit for `c`
(which yields class-*indicative* explanations).

**Eq. 3 — Complementary masked input** (Yu et al., 2019):

```
x̃^c = (1 − λ) · x + λ · b
```

**Eq. 4 — Sufficiency + comprehensiveness** (intermediate; combines deletion
and preservation games from Fong & Vedaldi 2017 into one loss):

```
arg min       −L(x̃, c)  +  L(x̃^c, c)  +  Ω_λ
λ∈[0,1]^n
```

**Eq. 5 — Gaussian influence kernel.** Reparameterize `λ` through new
parameters `w ∈ ℝⁿ` and `σ ∈ ℝⁿ_{>0}`. The influence of weight `wᵢ` on
position `j` decays with distance `d(i, j)`:

```
w_{i→j} = wᵢ · exp( − d(i, j)² / σᵢ )
```

**Eq. 6 — Mask construction:**

```
λⱼ = sigmoid( Σᵢ w_{i→j} )
```

This gives spatial smoothness without a hard constraint: when `σᵢ` is large,
neighbouring `λ` values are forced to be similar; when `σᵢ` is small, sharp
boundaries are still allowed if the loss prefers them. This is the explicit
contrast the paper draws against Fong & Vedaldi (2017), who used a
fixed-resolution upsampling+blur scheme that *cannot* produce sharp masks.

**Eq. 7 — Log-barrier on the bandwidth vector** (keeps `σ` away from zero
unless the data forces sharpening):

```
Ω_σ = − α_σ · (1/n) Σᵢ log(σᵢ)
```

**Eq. 8 — Final objective**, optimised per instance over `(w, σ)` with the
classifier frozen:

```
arg min       −L(x̃, c)  +  L(x̃^c, c)  +  Ω_λ  +  Ω_σ
w, σ ∈ ℝⁿ
```

**Image-only addition:** for images, a further regulariser penalising squared
differences between 8-connected pixel mask values is added. This is *not*
used for text and is therefore not relevant to our manuscript pipeline.

### Hyperparameters used in the paper's BERT-base text experiments

These are stated explicitly in Appendix A.1 of the published PDF and pin
down everything ambiguous in §3:

| Quantity                                          | Value                          |
| ------------------------------------------------- | ------------------------------ |
| Sparsity weight `α_λ`                             | **1.0**                        |
| Bandwidth log-barrier weight `α_σ`                | **1.2**                        |
| `w` initialisation (uniform across positions)     | **1.2**                        |
| `σ` initialisation (uniform across positions)     | **2.0**                        |
| Per-step Gaussian noise added to `x̃` and `x̃^c`  | zero-mean, **σ_noise = 0.03**  |
| Per-step random snap of mask values to {0, 1}     | **5%** of positions per step   |
| Scoring function `L(·, c)`                        | log-likelihood of class `c`    |
| Optimiser                                         | Adam                           |
| Distance function for text                        | `d(i, j) = |i − j|` (word idx) |
| Long-input handling (BERT-base 510 limit)         | split into segments, 100-token overlap, separate mask per segment |

The paper does not state a fixed Adam step count — it characterises the
budget only as "hundreds of forward and backward passes" per instance, and
gives wall-clock figures of **2–3 minutes per BERT-base sample** and
**~1 minute per image** (ResNet-101 or ViT-B/16) on modern hardware.

**Word-piece handling.** When BERT's WordPiece tokenizer splits a word into
multiple sub-word tokens, MaRC ties a single `(wᵢ, σᵢ)` parameter pair
across all pieces of that word, so the resulting explanations are word-level
rather than wordpiece-level. This is exactly what we'd want for Latin
fragments where wordpiece boundaries are essentially arbitrary.

The image hyperparameters (ResNet-101 / ViT-B/16) live in Appendix A.2 of
the paper; we have not transcribed them here because they are not relevant
to Latin manuscript retrieval.

---

## 4. How MaRC Differs from Integrated Gradients

1. **Computation model.** IG is a single-pass attribution: a Riemann sum of
   gradients along a straight line from a baseline to the input, with `n_steps`
   forward + backward passes and **no learnable parameters**. MaRC is a full
   per-instance optimisation over a parameter set `(w, σ) ∈ ℝ^{2n}`, with
   hundreds of Adam steps and explicit convergence dynamics.
2. **Output type.** IG outputs signed, per-token gradient × input scores with
   no sparsity prior; positive and negative attributions cancel. MaRC outputs a
   non-negative `[0, 1]` soft mask with explicit sparsity (`Ω_λ`) and explicit
   spatial smoothness (Gaussian `σ`), so contiguous spans emerge naturally and
   can be thresholded into a hard rationale.
3. **What is being explained.** IG explains the model's gradient with respect
   to a target output. MaRC explains by *perturbation*: it directly tests
   whether keeping only the masked tokens preserves the prediction
   (*sufficiency*) and whether keeping only the complement collapses it
   (*comprehensiveness*). MaRC's loss is therefore aligned with the standard
   ERASER faithfulness metrics by construction; IG is not.
4. **Compute.** IG is cheap (~tens of forward passes, no parameters, no
   per-example state). MaRC is expensive: ~2–3 min per BERT-base example,
   ~1 min per image. For our IG-comparison example pool that's a real
   budget item, not a rounding error.
5. **What needs to live in memory.** IG needs the model and the input. MaRC
   needs the model, the input, *and* per-instance optimiser state for `(w, σ)`
   plus the persistent computation graph through the masked-input path — every
   evaluation step has to backprop through `λ → x̃ → M(x̃)`.

---

## 5. Implications for Our Latin Manuscript Retrieval Pipeline

Grounded in what is currently cached vs. what would have to be recomputed:

- **Our IG comparison consumes only cached NPZ artifacts.**
  `scripts/resubmit/run_resubmit_ig_comparison.py` and
  `scripts/ig/run_phase12f_visualize.py` never load a model — they only read
  pre-baked NPZ files under `runs/active/ig_examples/artifacts/`. Each NPZ
  carries `query_input_ids`, `query_hidden`, `query_attention`,
  `query_ig_baseline`, `query_ig_abtt`, plus `pcs` and `mean_vec` for ABTT.
  **No computation graph, no model state, no on-demand per-token gradients.**
- **A faithful MaRC implementation cannot be built on the cached artifacts
  alone.** MaRC needs to re-evaluate `L(x̃, c)` and `L(x̃^c, c)` after every
  Adam step on `(w, σ)`, which requires fresh forward passes through a live
  frozen model. The cached `query_hidden` is a frozen snapshot; we cannot
  backprop through it.
- **The minimum viable integration is a fork of the IG extraction step**, not
  a new analysis script. The current IG runner is
  `scripts/_archive/run_phase12e_pair_explanations.py` (lines 143–295 — IG
  computation via Captum's `LayerIntegratedGradients` with `n_steps = 40`).
  The change would be: add a new attribution target alongside
  `PC1DotTarget` / `ABTTNormTarget` in `src/attribution_targets.py` that
  exposes the masked-input scoring function MaRC needs (i.e. a callable
  `λ → cosine_similarity(M(λ·x + (1−λ)·PAD), candidate)`), then run a per-pair
  Adam loop over `(w, σ)` instead of Captum's IG path integral.
- **Where we *could* cheat** for a cheap first pass: optimise a mask in
  *cached embedding space* — i.e. learn weights over the rows of `query_hidden`
  before mean-pool, then compare the resulting pooled vector against the
  cached candidate vector. This is **not** MaRC (no live model, no
  fidelity-on-the-classifier, no real perturbation), but it reuses 100% of the
  existing artifacts, runs in seconds per example, and would be a useful
  ablation / sanity check before paying for the real thing. It also gives us
  something to compare *against* MaRC if we ever do run MaRC for real.
- **Compute estimate for a real MaRC pass over our current IG example pool**:
  4 models (LaTa, PhilTa, LaBSE, Qwen3-0.6B) × roughly 50 example pairs × ~100
  Adam steps × per-step forward+backward ≈ **single-digit GPU-hours per model**
  on `gpuA100x4`, but only after the new target and the per-pair Adam loop are
  written. This would be a Phase-13-style experiment, not a quick patch.
- **Variant design hooks** to keep in mind for our follow-up meeting:
  - Replace the Gaussian distance kernel `d(i, j)` with one indexed by
    *line/folio* position from the manuscript metadata, not just sequential
    token index. This is a real linguistic difference from MaRC's sequence
    prior and would matter for fragmentary inputs.
  - Tie the fidelity loss `L(·, c)` to **ABTT-cleaned cosine similarity
    against the candidate vector**, not classification log-likelihood, so the
    explanation aligns with our actual retrieval objective rather than a
    synthetic classifier head.
  - Drop the per-instance optimisation in favour of an *amortised* mask
    predictor (a small head that emits `(w, σ)` from a forward pass over the
    cached hidden states). This gives up some faithfulness but turns the
    method into something that scales to all 2,238 unlabelled queries instead
    of a hand-curated example set. Being amortised is also a real
    methodological difference from MaRC.

---

## 6. Limitations the Authors Admit (Section 7)

The paper's Section 7 (Limitations) is short — exactly two paragraphs — and
makes only the following two claims explicitly:

1. **Human-likeness is upper-bounded by the model.** Quoting directly:
   *"the similarity to human rationales is always limited by the inner
   workings of the respective neural network: If a network's reasoning does
   not mirror human reasoning, the resulting rationales will be
   incomprehensible to humans."* MaRC cannot make a bad classifier
   interpretable.
2. **Computational cost.** *"Rationales created by MaRC are the result of a
   complete input optimization process. Therefore, the rationale creation
   usually requires hundreds of forward passes and gradient evaluations …
   creating a rationale for BERT-base can take two to three minutes
   depending on the length of the input text, while ResNet-101 and ViT-B/16
   are faster at about one minute."* The authors explicitly call this
   infeasible for real-time applications.

A third limitation — that MaRC requires a spatial structure on the input
(token index, pixel grid) and does not apply to unstructured feature vectors
— is **implicit** in the method (Eq. 5 needs `d(i, j)`) but is **not stated**
in Section 7. The paper actually frames the spatial-structure requirement as
a *generality* claim (it works for text, images, and "auditory data") rather
than as a limitation, in the Conclusion.

---

## 7. Source Code Status

- **Public, confirmed live 2026-04-07:**
  <https://github.com/inas-argumentation/Explainability>
- 100% Python; folders `movie_reviews/` and `imagenet/`; entry points
  `run_movie_reviews.py` and `run_imagenet.py`; PyTorch-style stack inferred
  from `requirements.txt` and the experiments described.
- **No LICENSE file visible** — flag if we ever want to copy code rather than
  re-implement from the paper.
- The repo URL is also written into the paper's Appendix A and was already
  saved in `docs/New resources April 7/github link.txt` (one-line file).

---

## 8. Discrepancy Log

- The 2026-04-07 meeting notes say the GitHub link file was empty and that no
  public source was found. This is wrong on both counts: the file contains the
  URL, and the repo is live with both text and image implementations.
- The 2025 arXiv re-deposit (`2508.11388`) makes the paper *look* like an
  August 2025 submission. It is not — it is an existing **ACL Findings 2023**
  paper. Our future related-work section should cite the 2023 venue, not the
  arXiv year.
- Equation forms and hyperparameters in §3 were extracted directly from the
  PDF using `pymupdf` (installed into the `localLatin` conda environment on
  2026-04-07 specifically for this verification pass). The previously
  reported values in the first draft of this note (which were transcribed
  second-hand by an exploration agent) all checked out, with one
  presentation fix: the paper's Equation 2 is the *full* sufficiency arg-min
  (`−L(x̃, c) + Ω_λ`), not the regularizer `Ω_λ` alone, and an intermediate
  Equation 4 (sufficiency + comprehensiveness, before the `(w, σ)`
  reparameterisation) was missing entirely. Both are fixed above.
