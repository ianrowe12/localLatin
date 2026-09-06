# Story Memo: ARR October Submission

**To:** Siddique, James
**From:** Ian
**Date:** 2026-08-23 (for the meeting of Wed 2026-08-26)
**Paper:** `overleaf_drafts/acl_latex.tex`

## 1. Thesis

We are writing a benchmark and analysis paper, not a method paper. ABTT is prior art (Mu and Viswanath, 2018) and we say so plainly. Our contribution is a new evaluation resource plus a diagnosis. We release a Medieval Latin canon-law benchmark of 840 labelled fragments attested by 1,705 manuscript witnesses, with a pairwise task and an open-set routing task. We then show that every one of six pre-trained encoders produces anisotropic passage embeddings on this corpus, that the anisotropy destroys cosine retrieval in the intermediate layers of the Latin and multilingual T5 encoders, and that a train-only geometric correction fitted on 847 training files repairs all six models to within one accuracy point of each other. Finally we look inside the score and ask which tokens the corrected similarity actually rests on. The paper's value to a reader is a resource, a failure mode with a name and a measurement, and an honest account of how far the token-level explanation goes.

## 2. The Three Planks

### Plank 1: The benchmark

The resource is the reason this paper belongs at an ARR venue rather than in a methods workshop. The corpus holds 1,705 Medieval Latin canon-law manuscript witnesses arranged into 840 directories, where a directory is one labelled fragment and its surviving witnesses. Directory sizes run from one to ten files. We define two tasks on it. Task A is pairwise duplicate detection scored from the test n-by-n cosine matrix. Task B is open-set directory routing, where the system either files a query into an existing directory or marks it new, with the threshold tau learned on train.

Evidence in the draft: Section 2 (Task and Dataset), Table `tab:split`. The split is 847 train and 858 test files, 565 positive train pairs and 595 positive test pairs, with 535 existing and 323 new files on the test side. It is generated deterministically by `src/canon_split_v2.py` with seed 42 and stored at `runs/active/resubmit/data/phase_resubmit_split.csv`.

Resolved (issue #39): the provenance paragraph and citations are now in Section 2. The corpus comes from the Carolingian Canon Law project (CCL, <https://ccl.rch.uky.edu>), created by Prof. Abigail Firey at the University of Kentucky in 2009 and supervised by her since. There is no Latin department involvement. Transcription is manual, mostly from manuscript images and for some texts from 18th-century printings. Pre-2019 transcribers worked from microfilm into word-processed files that CCL contributors converted to TEI-P5; post-2019 they use IIIF images in the Transcription Desk, after which an approved proofreader on the Proofreading Desk checks the text to the letter and adds TEI markup. The proofreading stage covers the post-2019 route only, so do not claim it for the whole corpus. Transcribers and proofreaders are credited per unit in the CCL interface. Most directories in our corpus correspond to CCL source keys (a conciliar canon, papal decretal, royal capitulary or patristic excerpt), but 150 of 840 labels are not key-shaped: modern edition citations, `Can.apost.N`, biblical references, and a handful of unidentified sources, which are all singletons and so contribute no positive pairs. Files are witnesses, individual hand-copied manuscripts identified by siglum. Citations added: `firey2009ccl` (project) and `eichbauer2014ccl` (Digital Philology review of the project). Still open: licensing and usage terms, and how our plain-text files were extracted from the published TEI. See question 1 below.

### Plank 2: Anisotropy is universal, and ABTT repairs it in all six models

We move the three sentence-embedding models back into the main text so the claim covers all six: LaTa, PhilTa, mT5-base, LaBSE, Qwen3-0.6B, KaLM-mini.

Two facts carry this plank. First, anisotropy is present in all six. At each model's most collapsed layer, meaning the layer with the highest top-PC variance ratio, the mean pairwise cosine is 0.226 (LaTa L4), 0.548 (PhilTa L6), 0.342 (mT5-base L5), 0.872 (LaBSE L8), 0.952 (Qwen3-0.6B L16) and 0.884 (KaLM-mini L5). ABTT with D=10 drives all six to roughly 0.00 and raises effective rank from as low as 1.00 to between 152 and 178. Source: `runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv` (400 rows), summarised in Appendix Table `tab:layer_diagnostics_main`.

Second, the retrieval consequence is largest where the collapse is deepest, and the repair is uniform. Baseline mean-pooled AUROC bottoms out at 0.497 for LaTa (layer 6, chance), 0.539 for PhilTa (layer 10) and 0.654 for mT5-base (layer 5). ABTT at those same layers gives 0.963, 0.982 and 0.977. Source: `runs/active/resubmit/results/phase_resubmit_results.csv` (700 rows), rendered as Figures `fig:dip` and `fig:gap` and, per layer, as Table `tab:taskA_main` for the headline models and `tab:taskA_appendix` for the other three.

The cleanest headline number is Task B under five query and reference reseedings. Best-layer baseline top-1 accuracy spans 36.63 points across the six models, from 50.6 for mT5-base to 87.3 for KaLM-mini. Under the best per-model SIF plus ABTT configuration, which is `sif_abtt_fixed` at D=10 for five models and `sif_abtt_optimal` at D=7 for LaBSE, that spread collapses to 1.0 point, from 89.4 to 90.4, and every model clears 95 percent by top-2. Source: `runs/active/resubmit/taskb_mseed/aggregated_results.csv` (400 rows), rendered as Table `tab:taskb`. Per-layer tables live in `runs/active/resubmit/results/perlayer_tables/`.

That last table already uses SIF plus ABTT while the main Task A tables are ABTT-only. SIF therefore returns to the main text and the main tables become baseline, SIF, ABTT, SIF plus ABTT. The framing is that SIF reweights tokens by frequency before pooling and ABTT removes dominant directions after pooling, so the two corrections are complementary rather than competing.

### Plank 3: What the repaired score rests on

We explain the retrieval score at layers picked by a predeclared train-only rule (earliest layer within 0.5 points of best train directory accuracy at rank 1 under ABTT), which locks LaTa layer 7, PhilTa layer 1 and mT5-base layer 1 before any attribution metric is computed. We use two views of the same scalar: integrated gradients, and MaRC adapted from classification to bi-encoder retrieval by optimising a soft mask on one side against a fixed partner embedding (Equation `eq:retrieval_mark`).

The result we stand behind is rank faithfulness. Leave-one-out Spearman correlation improves under ABTT in all six model-by-view cells: LaTa IG -0.001 to 0.396, LaTa MaRC 0.042 to 0.401, PhilTa IG 0.182 to 0.607, PhilTa MaRC 0.190 to 0.331, mT5-base IG 0.164 to 0.597, mT5-base MaRC 0.250 to 0.392. Evidence: Table `tab:attribution_metrics_main`, Figure `fig:attribution_rho_loo_main`, and Figure `fig:pairmatrix` for the qualitative PhilTa example. Source data: `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv`, 200 positive pairs per model.

## 3. What We Do Not Claim

We do not claim a new repair method. ABTT is borrowed and the paper says so in the introduction. The one methodological contribution we do claim is narrow and sits inside the attribution analysis: the retrieval adaptation of MaRC, which replaces the classification target with pairwise cosine against a fixed partner embedding.

We do not claim that ABTT makes similarity depend on a small sufficient token set. Attribution ranking becomes more faithful under ABTT, and that is the whole of the token-level claim. ERASER-style sufficiency and comprehensiveness improve in only 6 of 12 main-threshold comparisons (9 of 18 for sufficiency and 10 of 18 for comprehensiveness across the 0.10, 0.25 and 0.50 sweep), and MinFrac at 0.80 recovery worsens in all six cells, meaning the corrected score generally needs more retained tokens, not fewer. This tradeoff stays visible in the main prose, not buried in an appendix. See `docs/runs/run4_interpretation_caveat_memo.md`.

We do not claim that label-free geometry selects the best retrieval layer. It diagnoses collapse. The operational layers come from a train-only retrieval rule and differ from the strongest diagnostic collapse layers for all three attribution models.

We do not claim a uniform mid-layer dip shape. Anisotropy is universal, but the retrieval dip is not. LaBSE, Qwen3-0.6B and KaLM-mini improve close to monotonically with depth under the baseline, so the honest phrasing is that all six models are anisotropic and all six gain from ABTT, while the catastrophic mid-depth collapse is a T5-encoder phenomenon. James and Siddique should sign off on this wording, since it is slightly narrower than "all six models show a mid-layer dip".

We do not claim global explanations of model behaviour. Every attribution result is local to one query and candidate pair. Baseline attribution explains raw cosine and ABTT attribution explains corrected cosine, so cross-variant comparisons are descriptive.

## 4. Questions for Siddique

**1. Dataset provenance and citation.** Mostly answered by Prof. Abigail Firey, who created the Carolingian Canon Law project (CCL) in 2009 and has supervised it since; the corpus derives from CCL, not from any Latin department. Section 2 now carries the provenance paragraph and cites the project (`firey2009ccl`) plus a peer-reviewed description of it (`eichbauer2014ccl`). Two items remain for the in-person meeting Firey offered. (a) Co-authorship: the dataset is her project's output, so decide whether to offer her co-authorship rather than an acknowledgment. (b) Licence and usage: her email does not state any terms, so we must ask before releasing the benchmark and before writing the ARR checklist and ethics language. Item (c), data derivation, is answered (issue #111): `scripts/data/tei_to_canon.py` converts a CCL TEI-P5 export into our layout and reproduces all 279 existing BN2123 files after whitespace normalisation, so Section 2 can state the derivation step instead of implying we redistribute the TEI verbatim. See `docs/research/data_derivation.md`.

**2. Validation split.** James asked why there is no dev set. Our current protocol is train and test only. Everything fitted comes from train: ABTT principal components, the D value, the routing threshold tau, and the attribution layer rule. Nothing touches test before evaluation. Two options. Keep the two-way split and add one paragraph explaining that all selection is train-internal, which is cheap and defensible. Or carve a validation split out of the 847 training files and refit, which costs a rerun of the sweep and shrinks the already thin positive-pair pool of 565 train pairs. Recommendation is the first, but we should decide on Wednesday because it changes Section 2.3.

**3. SIF scope in the main text.** Confirm the plan: main tables show baseline, SIF, ABTT and SIF plus ABTT, one short paragraph explains why the two corrections are complementary, and the full SIF variant sweeps stay in the appendix.

**4. Final title.** See below. The current title advertises "Geometric Repair", which reads like a method contribution and invites the wrong review.

## 5. Candidate Titles

1. *Anisotropy and Semantic Matching in Medieval Latin: A Benchmark and Layerwise Analysis*
2. *Where Retrieval Collapses Inside the Encoder: A Medieval Latin Benchmark for Semantic Matching*
3. *Matching Manuscript Fragments in Medieval Latin: A Benchmark and a Layerwise Study of Embedding Anisotropy*

All three drop "Geometric Repair" and lead with the resource and the analysis, which is what we are actually submitting.
