# Analysis-paper reframe: placeholder handoff and decisions for Ian

Branch `paper-reframe-analysis`, 2026-09-27. James's session restructured
`overleaf_drafts/acl_latex.tex` and `related_work.tex` into the analysis
framing agreed on 2026-09-26/27 (summary below). Every claim that still needs
a run is marked in the LaTeX with `\pending{<id>: ...}` (a red bracketed note)
or `\pendingnum{<id>}` (a red cell). Remove every use before submission;
`grep -n pending overleaf_drafts/acl_latex.tex` lists them.

## Work division

| Who | Experiments | Paper regions owned (to avoid edit conflicts) |
|---|---|---|
| James | E1, E2, P2x2 | Sec. 5 paragraphs "E1" and "E2" and `tab:e1_coordinate_ablation`; Sec. 6 paragraph "P2x2" and `tab:panel_2x2`; the status cells of `tab:model_panel` in Sec. 3 |
| Ian | E3, H1, FT, GEN, CI, PQ, WHITEN; the "Route to Ian" list below; submission housekeeping | Everything else |

James re-extracts the per-layer embeddings locally for E1 and E2 (E2 needs
token-level states, which the cached pooled `.npy` files do not hold). The
corpus, extraction CLIs and models are all available; the one input that is
not is `runs/active/resubmit/data/phase_resubmit_split.csv`, which is
gitignored and cannot be regenerated from the seed alone (the corrected split
carried an earlier assignment over, see `benchmark_v1.md`). Ian: please share
that file. James will confirm the extraction by reproducing a few baseline
AUROC cells from `runs/active/resubmit/results/phase_resubmit_results.csv`
before running E1. P2x2 needs GPU extraction for the new models (minutes each);
James will size the reservations at about one hour.

Results flow back the same way for both: replace the `\pending` text with the
finding, and produce any new table or figure from a script under
`scripts/paper/reframe/` so it can be regenerated. The facts file behind
Section 4 is reproduced with
`python scripts/paper/reframe/geometry_vs_retrieval.py --facts_md <path>`.

## Framing in one paragraph

The paper is an analysis paper: a controlled case study of mid-depth retrieval
collapse in T5-family encoders on a medieval Latin testbed with edition labels.
The practice under test is "take an off-the-shelf encoder, read intermediate
layers, diagnose with an anisotropy statistic, apply a post-hoc correction".
The testbed is the instrument: model-independent labels, a train-only protocol
at every layer, a character n-gram reference that ties the best encoder (so a
failure to retrieve is a property of the representation), and Latin encoders in
two architectures. The benchmark framing is dropped; attribution is in the
appendix and is no longer a contribution. Every "first" is scoped to "the first
per-layer retrieval profile of T5 encoder states in isolation that we know of".

## Placeholders, by experiment

| Id | What fills it | Where in the paper | Cost (from the framing report) |
|---|---|---|---|
| **E1** | Per-layer coordinate ablation at every layer of LaTa, PhilTa, mT5-base, LaBSE, Qwen3-0.6B. Rank residual coordinates on TRAIN passages two ways, by mean \|value\| and by across-passage variance (or Timkey's cosine-contribution measure), and report both; zero the top k (k = 1, 3, 5, 10) under each ranking; per-coordinate standardization; centering only (D=0); ABTT D=1 as a reference (it repairs a rank-1 nuisance under either account). AUROC and top-PC share after each. Predictions as written in Sec. 5: zeroing k <= 5 by variance restores collapsed T5 layers to AUROC >= 0.90; LaBSE/Qwen3 change by at most 0.03 at their train-selected layers and recover only a small part of the ABTT gain at their weakest (layer 1); if standardization works but zeroing k <= 5 fails, the nuisance is "a few coordinates, but more than 5"; if only a projection works, it is distributed. Also recompute the two cosine shares now shown as `\pendingnum` in Sec. 5 (LaTa layer 7, mT5-base layer 1), using Timkey's summed per-coordinate contribution; and the ratio r (below) | Sec. 5 paragraph "E1", table* `tab:e1_coordinate_ablation` (Base column filled from z1_numbers.md), intro contribution (3) | about 1 GPU-hour with E2/E3, prototypable on a laptop from the cached `.npy` |
| **E2** | Token audit: which tokens carry the top coordinates (sign, magnitude), the ratio r of the pooled coordinates, relation of the top-component score to log length and frequent-token share. SIF differs from mean pooling in TWO ways (frequency down-weighting AND zero weight on special tokens, `sif_abtt.py` L86-88; mean pooling keeps `</s>`), so add a control: mean pooling with special tokens dropped and no frequency weighting. Prediction reversed from the draft: if EOS carried mT5-base's coordinates, SIF would delete it and rescue mT5-base; since it does not, mT5's carriers are tokens SIF keeps. Length check for LaTa's negative gap: \|delta log length\| for same- vs DIFFERENT-directory pairs (checkable now) | Sec. 5 paragraph "E2"; pointers in Sec. 4 | with E1 |
| **E3** | Subspace split: Task A AUROC of the projection onto PC1 alone, onto PCs 2..D, and onto the retained subspace; compare removed vs retained subspace; pre-trained vs fine-tuned. Cleanest contrast: LaTa layer 12, pre-trained (ABTT 0.938 -> 0.971) vs fine-tuned (0.984 -> 0.970). Corrected third falsifier: "the removed subspace ranks as well as the retained one in pre-trained T5s" | Sec. 5 paragraph "E3"; Sec. 7 "Where the repair costs" | CPU on cached vectors |
| **H1** | ABTT D ablation, D in {0,1,2,3,5,7,10,15,20,30,50}, all 100 model-layers, Task A AUROC. Already known and now stated in the paper: train-DirAcc@1 selection never picks D <= 2 (abtt_optimal: D=10 in 87/100, D=7 10, D=5 2, D=3 1; sif_abtt_optimal: D=10 in 81/100). Rank-1 test: D=1 recovers >= 80% of the D=10 AUROC gain at collapsed T5 layers, D=0 recovers little | Sec. 5 paragraph "H1", placeholder figure `fig:d_ablation`; Discussion `\pending{H1: ...}` | 1-2 CPU-hours |
| **P2x2** | Factors redefined: "T5 encoder (span-corruption pretraining) vs encoder-only (MLM)" x "no embedding objective vs contrastive embedding objective". Add raw encoder-only (LaBERTa, PhilBERTa, Latin BERT), embedding-trained T5 (Sentence-T5; GTR-T5 as alternative), AND raw T5-v1.1-base on the Latin corpus as Sentence-T5's matched raw partner. The raw encoder-only cell is the discriminating test (healthy -> collapse follows T5 pretraining; collapse -> follows the missing embedding objective). For the Sentence-T5 cell use label-free top-PC share / effective rank as the primary readout. The fine-tuning result is written as "does not separate the accounts" | Sec. 3 `tab:model_panel` (status cells), Sec. 6 and `tab:panel_2x2`, intro contribution (1), abstract, Limitations | about 1-1.5 GPU-hours; minutes per Latin model |
| **FT** | Import the layerwise fine-tuned LaTa row (layers 2-11 at AUROC 0.50-0.57; last layer 0.938 -> 0.984; 0.997 is KaLM-mini's) and its peak top-PC share into `tab:panel_2x2` | Sec. 6 | none |
| **GEN** | Label-free generality, crossed model x text: mT5-base and PhilTa (pretraining covers English) on an English sample; T5-v1.1-base on BOTH the Latin corpus and the English sample. Top-PC share and effective rank per layer. Fail branch: if mT5-base stays high-rank on English, collapse depends on the input text; if T5-v1.1-base stays high-rank on English while mT5-base collapses there, it depends on the pretraining data. (Flan-T5 dropped) | Sec. 6 paragraph "GEN" | minutes of forward passes |
| **CI** | Directory-level bootstrap confidence intervals on every headline cell of `tab:taskA_headline` and `tab:taskB_headline` | Sec. 7, Limitations | CPU |
| **PQ** | Five per-query routing checks: threshold-free existing-vs-new AUROC, oracle-vs-train threshold gap, hubness statistic, centering-only (D=0), and a threshold refit on a fine or quantile grid (the current grid is linspace(0,1,200), step ~0.005, against Qwen3 test-cosine SD as small as 0.007). The paper does not attribute the routing gain to the offset geometry until these run | Sec. 7 paragraph "The offset regime and the open-set decision" | CPU on cached vectors |
| **WHITEN** | PCA whitening with reduced dimension k in {64, 128, 256} (CPU, fit on train), Task A AUROC per model at the train-selected layer. Full-rank whitening on 847 vectors in 768-1,024 dims is ill-conditioned, which the paper now says | Sec. 3 "Post-processing" | CPU on cached vectors |

Refined hypothesis (Sec. 5), now measurable: r = across-passage SD of a pooled coordinate / \|mean\| over training passages. Predicted r >= 1 for the top coordinates at collapsed T5 layers; r < 1 at mT5-base layer 1 and in Qwen3-0.6B. Computed within E1/E2.

## Claim scoping against prior work

Every result sentence should stay inside the "wording that survives" column.
The papers are in `custom.bib`.

| Claim | Closest prior work | Wording that survives |
|---|---|---|
| Layerwise retrieval profile of T5 encoder states; mid-depth collapse | Mikkelsen 2026 (JMIR; retrieval-trained BERT encoders collapse mid-depth, held back as contemporaneous, cite only as a contrast if P2x2 supports a T5-specific account); WhiteningBERT (Huang 2021: T5 gains most from whitening, first+last layers only); Sentence-T5 (final layer only); Godey 2024 (encoder+decoder concatenated); Zhao et al. NAACL 2025 (T5 encoder outliers) | "the first per-layer retrieval profile of T5 encoder states in isolation that we know of"; "T5-specific" only as "in our panel" |
| Two anisotropy statistics vs retrieval | Timkey & van Schijndel 2021 (offset vs high-variance dimensions); IsoScore (Rudman 2022: mean cosine measures the offset); Razzhigaev 2024; Parupudi 2026 (top-dimension share predicts cosine failure across 19 final-layer encoders); Rottach/Rudman 2025 (effective rank best geometry predictor of retrieval); Nastase & Merlo 2025, Kulkarni 2026 (geometry predicts performance unreliably) | a layerwise, within-testbed comparison including raw T5 states; top-PC share (centred) separates collapsed from healthy layers and ranks weakly within a regime; never "we identify two forms", "opposite effects" |
| Massive-coordinate mechanism | Timkey 2021 (rogue dimensions drive cosine; standardization/ABTT at every layer); Luo et al. ACL 2021 (zeroing outlier neurons helps mean-pooled STS); Hämmerl 2023 (zeroing one mid-layer dimension improves retrieval); Sun et al. COLM 2024 (massive activations, decoder LLMs); Queipo-de-Llano 2026 (massive activations force near-rank-1 middle layers); Puccetti 2022 (outliers track token frequency) | confirmation in a new setting: in T5 encoders the known mechanism drives mean-pooled passage retrieval to chance; constant per-token values become per-passage variation through mean pooling; SIF's partial rescue runs through the tokens carrying the outliers |
| Fine-tuning fixes only the output layer | Hämmerl 2023 (S-BERT tuning removes outliers at the output, not layer 8); Mikkelsen 2026; 2DMSE (Li 2024); Merchant 2020 | holds also for a T5 encoder; cite as a replication; "small-scale supervision at the output does not reach the middle layers" |
| ABTT hurts models fine-tuned on the target task | Rajaee & Pilehvar Findings-EMNLP 2021 (ABTT lowers STS of Siamese-fine-tuned BERT); Rudman et al. 2023/2024; counter-evidence: Reichbauer 2026 (whitening still helps fine-tuned Latin/Greek encoders), Su 2021, Li 2020, Ren 2026 | "consistent with", not "replicates"; scoped to pairwise ranking, since routing still rises; Qwen3/KaLM drops are 0.002-0.003 and within noise pending CI |
| The repair itself | Mu & Viswanath 2018 (ABTT); Timkey applies it at every layer of BERT-family models | not a contribution; "a train-only top-D projection restores every layer, so the signal was present but masked"; novel only as passage retrieval, T5 mid-depth at chance, train-only fit, cross-family contrast |

## Submission housekeeping (Ian)

- No link to the public upstream repository anywhere in the submission or
  supplementary material; the public commit history is itself an anonymity
  exposure, so do not cite the repo name in the paper.
- OpenReview profiles with ORCID for every author before the deadline.
- Suggested track: Semantics (sentence-level); reviewers there will expect the
  whitening comparison (WHITEN) and the fine-tuning contrast.
- Realistic target is Findings: soundness and reproducibility over excitement.
  Prefer confidence intervals (CI) and the clean 2x2 over any new claim.
- Deadline 12 October AoE. Results should land by about 5 October so the
  placeholders can be filled and the main text cut to eight pages.

Conditional citation: two `% CONDITIONAL: cite mikkelsen2026 ...` comments
(related work, Discussion). James's decision: Mikkelsen 2026 (JMIR Med Inform,
retrieval-trained BERT encoders) is contemporaneous and in another subfield;
cite it only as a contrast if P2x2 supports a T5-specific account.

## P2x2 as run and as reported (2026-09-30, issue #248)

James's decisions, which replace the P2x2 row above.

- **Run**: ten 12-layer encoders scored per layer (Task A test AUROC; train top-PC
  share and effective rank): LaTa, PhilTa, mT5-base and LaBSE from the existing
  caches, and T5-v1.1-base, T5-base, Sentence-T5, LaBERTa, PhilBERTa and
  SPhilBERTa extracted for this issue. Latin BERT and GTR-T5 were dropped.
  Everything is in `runs/active/reframe/p2x2/` (`p2x2_layers.csv`,
  `p2x2_facts.md`, tokenization and extraction checks), regenerated by
  `scripts/paper/reframe/p2x2_panel.py`.
- **Reported in the paper**: the architecture dimension only, on two matched
  pairs of raw models from the same authors and the same data: LaTa with
  LaBERTa, PhilTa with PhilBERTa. The raw-versus-embedding-trained dimension is
  not in the paper. `tab:panel_2x2` and `tab:p2x2_layerwise` hold the four pair
  models and the two encoder-only siblings.
- **Results kept in the files, not in the paper**: raw T5-base does not collapse
  (lowest AUROC 0.816, peak top-PC share 0.545), and Sentence-T5 has top-PC
  share of 0.6 or more at layers 2 to 11 (peak 0.943) while its AUROC stays at
  0.80 to 0.89. Any sentence about the pairs must stay true with these in view:
  attribute the collapse to what differs within each pair, "in our panel", and
  make no general claim about T5 pretraining.

## Decisions taken on the branch that reverse or extend earlier ones

1. **Lexical baselines are disclosed** (reverses issue #197). Section 3 carries
   a "lexical reference" paragraph, Section 7 quotes 0.987 / 89.9 / 91.1, and
   `tables/lexical_baselines.tex` is now `\input` in the reference-systems
   appendix. The headline tables still lack the lexical row: regenerate them
   with `scripts/resubmit/build_headline_tables.py --lexical_csv
   runs/active/resubmit/results/lexical_baselines.csv` (two `% TODO(Ian)`
   comments mark the `\input` lines).
2. **Attribution is an appendix** (`app:attribution`: methods, metrics, setup,
   results, verbatim from the old main text). Contribution (4) on attribution
   is cut. The generated caption of `tables/attribution_metrics_secondary.tex`
   still says "the main table" three times; fix in its generator.
3. **Title is provisional**: "Mid-Depth Retrieval Collapse in T5 Encoders: A
   Layerwise Case Study on Medieval Latin Witnesses".
4. **New generated artefacts** come from
   `scripts/paper/reframe/geometry_vs_retrieval.py` (CPU, reads the two
   committed CSVs): `tables/geometry_regimes.tex`,
   `tables/geometry_correlations.tex`, `figures/fig_geometry_vs_auroc.pdf`,
   `figures/fig_standardized_gap.pdf`. The two figures are force-added on the
   branch because `overleaf_drafts/figures/*.pdf` is gitignored.
5. **Figure 1 is now AUROC vs depth** (`fig_release_aucroc_6model.pdf`,
   `fig:dip`); the cosine-gap figure (`fig:gap`) moved to Appendix B.
6. **Not in the paper by decision**: the decretals collection, the
   HuggingFace final-norm question, the four rejected motivation premises
   (few labels rule out fine-tuning; only Latin encoders are T5s; controls are
   as anisotropic; scholars need an encoder).
7. **Numbers recomputed**: the framing report's facts sheet had six ranges
   wrong (collapsed top-PC share is 0.76–1.00, effective rank 1.0–4.6, mean
   pairwise cosine of collapsed layers 0.23–0.62, non-T5 effective rank
   16–150, D=10 selected in 81/100 under sif_abtt_optimal, SIF rescue
   0.84–0.94). The paper uses the recomputed values; see the script.
8. **Wording that was corrected**: no "anisotropy is universal", no "0.23 to
   0.95" cone, no "SIF helps only collapsed models", no "repair is universal",
   no "first report of the collapse". Within-family correlations between
   top-PC share and AUROC are weak (about -0.5), so the claim is that the
   variance statistics separate the collapsed regime from the healthy one. The
   earlier "predict where the repair gains most" claim is dropped: ABTT AUROC is
   nearly constant (0.962-0.987), so the gain correlations mirror the baseline
   ones.

## Length

Main text (Discussion) ends on page 14 under the review class after the review
fixes (limit 8); `\appendix` starts on page 17. James asked not to cut to fit
while drafting. Sections 5 and 6 are hypothesis-test scaffolds that will shrink
once results replace the placeholders; the model-panel and 2x2 tables can be
merged once P2x2 lands (layout review I5).

## Review fixes applied (2026-09-27)

Five Opus reviews (technical, logic, consistency, writing, layout) of commit
88b303e were merged and applied to `acl_latex.tex`, `related_work.tex` and the
captions/figure labels of `scripts/paper/reframe/geometry_vs_retrieval.py`
(rerun; table numbers and the generated facts are unchanged). Main fixes:
fine-tuned LaTa 0.984 (not 0.997) and "improves its last layer"; SIF described
as frequency down-weighting plus dropping special tokens, with the E2
prediction reversed and a special-token control; the "shared offset preserves
ranking" error removed and PQ given a fifth (threshold-grid) check; the
ABTT-gain correlations described as mirroring the baseline; the standardized
gap presented as label-based; the separation stated as a top-PC share
threshold (0.6 marks all 26 collapsed layers, one false alarm) with the
clustering caveat; E1/E3/H1/GEN/P2x2 redesigned as in the table above;
contributions rewritten one sentence per item; lexical claim scoped to
pre-trained configurations; "predeclares ... before any test number is read"
replaced; whitening n~d reason plus WHITEN placeholder; appendix anisotropic
layer table labelled as test-split; attribution layer rule described as a
variant; `\onecolumn` wrappers removed around Appendices D and H (the two wide
generated `table` floats are set as `table*` instead); stale drafting comments
removed; terminology (top-PC share, DirAcc@1, testbed, regime) and American
spelling unified.

## Route to Ian (generated artefacts)

These fixes touch generated files or plotting scripts outside this branch's
reframe script, so they were not applied here. Exact wording where applicable.

1. `tables/taskA_headline.tex` (generator `scripts/resubmit/build_headline_tables.py`):
   caption "Cosine gap is defined in Figure~\ref{fig:gap}" -> "Cosine gap is
   defined in Section~\ref{sec:tasks}" (fig:gap is now an appendix figure).
2. `tables/taskA_headline.tex`, `tables/taskB_headline.tex`: regenerate with
   `--lexical_csv runs/active/resubmit/results/lexical_baselines.csv` so the
   character 3-5-gram TF-IDF row appears under the rule (two `% TODO(Ian)`
   comments). Section 3 no longer claims the row is there.
3. `tables/taskB_ranking_appendix_mseed.tex` caption: "the mseed sweep was not
   run for the ABTT-only variant of Tables ... so we include this table" is
   partly stale, because five-seed ABTT-only values exist for LaTa, Qwen3-0.6B
   and KaLM-mini (quoted in Sec. 7: 87.7, 90.5, 90.9). Suggested: "This table
   covers SIF+ABTT; five-seed ABTT-only values for LaTa, Qwen3-0.6B and
   KaLM-mini are in Appendix~\ref{app:reference_systems}."
4. `tables/finetune_ceiling.tex`: the five-seed values Sec. 7 quotes (fine-tuned
   + ABTT 92.3 / 93.0 / 87.7 vs zero-shot ABTT 90.5 / 90.9 / 87.7) exist only in
   comments; add them to the table or its caption so the appendix pointer has a
   printed source.
5. `tables/attribution_metrics_secondary.tex` caption: "the main table" (three
   times) -> "Table~\ref{tab:attribution_metrics_main}".
6. Figure 1 `fig_release_aucroc_6model.pdf` and Figure `fig_release_gap_6model.pdf`
   (upstream plotting script): give each series its own linestyle and marker
   (Baseline dashed/square, SIF dotted/triangle, ABTT solid/circle); gray
   baseline is hidden under SIF in four panels. Tick/legend fonts >= 8 pt at
   print size (e.g. `figsize=(7, 2.6)`), `bbox_inches="tight"`, legend into a
   panel or the top margin; `matplotlib.rcParams["pdf.fonttype"] = 42`.
7. Type 3 fonts: Figures 1, 4 (gap), 6 (density) and 8-14 embed DejaVu as Type 3.
   Set `pdf.fonttype = 42` in their plotting scripts.
8. `paper_fig_density_2x2.pdf` (Fig. `fig:density`): illegible at column width
   (~2-3 pt text). Regenerate at 3.2 in wide with 8 pt fonts.
9. Attribution figures `fig_pair_matrix_philta`, `fig_attention_philta`,
   `fig_retrieval_mark_pair_philta`: token tick labels ~2-3 pt; raise font size,
   stack the four MaRC strips at full width with a taller figsize or truncate.
10. `tables/attribution_metrics_sweep_main_methods.tex`,
    `tables/attribution_metrics_sweep_supplemental_methods.tex`: `\scriptsize` +
    `\resizebox{\textwidth}` over 24 columns gives ~4.5 pt; split (Suff/Comp vs
    MinFrac) or `sidewaystable` at `\footnotesize` without `\resizebox`.
11. `tables/appendix_lasttok_comparison.tex`: caption is above the tabular; move
    it below (formatting.md).
12. t-SNE/UMAP figures: drop the in-figure suptitle that duplicates the caption,
    or pair t-SNE and UMAP per model group in one `figure*` (saves ~2 pages).
13. Generated captions (headline tables, finetune ceiling): add a one-sentence
    takeaway, and use one name for the uncorrected condition ("baseline" vs
    "raw view").
14. Appendix J silhouette sentence "rises by 0.07 to 1.16" (hand-written but
    sourced upstream): silhouette is bounded by 1; check the source and fix
    (likely "0.07 to 0.16").
15. `tables/attribution_delauc_sensitivity.tex` is never `\input`; input it in
    `app:attribution_sweeps` or delete it deliberately.
16. Author decision (layout I3c): Table 2's Mean cos./Cos. SD range columns
    repeat the prose, and Table 3's two gap rows repeat Figure 5; trimming
    either would let Table 3 go single-column.
17. Hand-written appendix table `tab:layer_diagnostics_main` has no generator
    (values: `geometry_per_layer.csv`, test split); a script would protect it.
18. Citation-faithfulness pass still needed: nastase2025geometry /
    kulkarni2026geometry / machina2024anisotropy for "geometry statistics predict
    task performance unreliably"; rudman2022isoscore "mainly measures the
    offset"; parupudi2026anisotropy; zhao2025outlier "grow with depth";
    haemmerl2023anisotropy "removes an outlier at the output but not at layer 8";
    reichbauer2026alignment "whitening still helps fine-tuned Latin and Greek
    encoders"; Sentence-T5 starting from an English T5 checkpoint (T5-v1.1 as
    its matched raw partner).
19. Check that the fine-tuned Qwen3-0.6B rows with and without ABTT sit at
    different train-selected layers (technical review: 28 vs 27), as Sec. 7 now
    says.
20. `chung2022flan` is no longer cited (Flan-T5 and T5-base dropped from GEN).
21. `scripts/paper/reframe/geometry_vs_retrieval.py` (layout M2c, optional): in
    `fig_geometry_vs_auroc` the healthy cluster overplots ~60 markers; add
    `alpha=0.8` or slightly smaller hollow markers. (`bbox_inches="tight"` is
    already set; the side padding comes from `width=0.9\textwidth` in the tex.)
