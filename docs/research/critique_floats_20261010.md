# Critique of Ian's floats, 10 October 2026 (main at 2d5529d)

Read-only review. Nothing in the repo was edited. I built a scratch copy of `overleaf_drafts/` with latexmk (57 pages; the main text ends on page 8 with 1.70 spare lines, which I measured) and rendered every main-text page and every one of Ian's floats. Then I prototyped the main-text fixes in two scratch variants and rebuilt them to measure how they change the page count.

Scope. Ian owns Figure 1 (`fig_depth`), panel (b) of Figure 3 (`fig_localize`) and Table 3 (`headline_main`). In the appendix he owns everything except James's E1, E2 and P2x2 floats (Table 8 `p2x2_layerwise`, Table 13 `e1_coordinate_ablation`, Table 16 `e2_token_audit`, Figure 8 `fig_e1_k_sweep`). `all_models` (Table 7) has both authors in its history, so it gets only the light pass.

## Short answer for James

The three main-text floats are the right ones. No existing artifact under `figures/`, `tables/` or the generators does their job better (see "Swap candidates" below). Each still needs a fix before submission:

1. **Table 3 can contradict the text.** Its per-cell brackets overlap where Section 7 says a paired interval excludes zero.
2. **Figure 1 does not say which panels are raw T5 encoders.**
3. **Figure 3(b) has an uneven D axis that nothing discloses**, and its caption gives no takeaway.

All three fixes fit without lengthening the main text. I measured this: spare space stays at 1.70 lines.

## Renders (all in the scratchpad)

Base dir: `/tmp/claude-92139/-projects-beto-irowerojas-localLatin/72d7648b-b52c-4c9d-b1d3-74d26e9f8d4f/scratchpad/`

| What | PNG |
|---|---|
| Main-text pages 1-9 (150 dpi) | `renders/page01_150dpi.png` ... `renders/page09_150dpi.png` |
| Figure 1, print crop (300 dpi) / laptop (100 dpi) | `renders/F1_fig_depth_300dpi.png`, `renders/F1_fig_depth_100dpi_laptop.png` |
| Figure 3, print crop | `renders/F3_fig_localize_300dpi.png` |
| Table 3, print crop | `renders/T3_headline_300dpi.png` |
| Figure 1 prototype (group labels, method key in the empty band) | `renders/F1_v2_prototype_300dpi.png`, with caption: `renders/VA_F1_with_caption_300dpi.png` |
| Figure 3 with the proposed caption | `renders/VA_F3_with_caption_300dpi.png` |
| Table 3, variant A (markers) / variant B (no bracket rows) | `renders/VA_T3_300dpi.png`, `renders/VB_T3_300dpi.png` |
| Page 7 under variant A | `renders/VA_page07_150dpi.png` |
| Appendix pages 10-57 | `renders/page10_150dpi.png` ... `renders/page57_150dpi.png` |

Prototype code and PDFs: `proto/paper_figures_v2.diff` (Figure 1 generator change), `proto/code/patch_tex.py` (caption and Table 3 edits), `proto/variantA_paper.pdf`, `proto/variantB_paper.pdf`.

Measured main-text spare lines on page 8: current 1.70; variant A (new Figure 1, new Figure 1 and 3 captions, Table 3 markers and one caption sentence) 1.70; variant B (A plus Table 3 without bracket rows) 7.46.

---

## Main-text floats

### Figure 1 (`fig_depth`, full width, page 3)

**(1) What a reader should take from it, and whether they get it in ten seconds.** Three encoders fall to chance in the middle layers, and ABTT puts every layer back near 0.97. The three U-shaped curves come across in about three seconds, so the shape works. What a first-time reader cannot see is *which kind* of model collapses. The figure sits on page 3 and is cited from the introduction, but "panel", "raw T5" and "embedding-trained" are only defined in Section 3, below it. Nothing in the figure or caption separates LaTa/PhilTa/mT5-base from LaBSE/Qwen3/KaLM. The "T5 collapses, embedding-trained does not" message therefore depends on the reader already knowing the model names.

**(2) Legibility.**
- Fonts: all text is DejaVu Sans at 8.3 pt printed, which is fine. Line weights: data 1.04 pt, axes 0.62 pt, chance line 0.62 pt in #888.
- Colours: Okabe-Ito, one colour per model, matching Figures 2 and 3. Panels separate the models, and dotted/open versus solid/filled separates the methods, so the figure survives grayscale and colour-blind viewing.
- Problems:
  - (a) In the Qwen3-0.6B (28 layers) and KaLM-mini (24 layers) panels, the markers merge into a thick bar. At laptop resolution (`F1_fig_depth_100dpi_laptop.png`) you cannot tell dotted from solid where the two curves meet at late layers.
  - (b) The chance line at 0.5 is faint and lies on the bottom edge of the gray band.
  - (c) "Layer" is printed six times. Harmless, but it costs nothing to keep.
  - (d) The legend says "ABTT (fit on training embeddings)" and the caption repeats it.

**(3) Caption.** It defines baseline, ABTT, the band and chance. It does not:
- say which panels are raw T5 versus embedding-trained;
- give D (top D <= 10, chosen per layer);
- state the takeaway (three raw T5 encoders collapse).

"Horizontal line: chance" should say 0.5 or "gray line".

**(4) Does it carry the prose?** Yes. It carries claims 1 and 2 (collapse; 0.962 to 0.987 after ABTT). The "routing still varies by layer" sentence is correctly sent to the appendix.

### Figure 3, panel (b) (`fig_localize`, full width, page 7)

**(1) Takeaway.** Removing one direction is not enough; three directions bring every collapsed layer above 0.90, and mT5-base keeps rising to ten. A reader gets this in about ten seconds, helped by the dashed 0.90 line.

**(2) Legibility.**
- Fonts 8.3 pt. Lines 1.24 pt, bands at alpha 0.13. Line style and marker differ per model (solid/circle, dashed/square, dash-dot/triangle), so grayscale works. Where the bands overlap they turn olive, which is acceptable.
- Problem 1: the x axis is categorical. Ticks 0, 1, 2, 3, 5, 7, 10 sit at equal spacing, so the slope after D=3 looks flatter than it is in D. Figure 9's caption discloses this; Figure 3's does not.
- Problem 2: the vertical line at k=10 in panel (a), which the caption refers to, is a 0.6 pt dotted #888 line drawn on top of the grid. It is nearly invisible in print (`F3_fig_localize_300dpi.png`).
- Model colours and markers match Figures 1 and 2.

**(3) Caption.** Panel (b) gets no takeaway sentence. Panel (a) spends a line on "within 0.001 of the baseline up to k = 10 and within 0.03 at 400". Table 2 already carries that ("<=0.0004 at 10"), so the line can go.

**(4) Does it carry the prose?** Yes. It carries "removing one recovers a median 45%" and "three components bring all 26 to 0.914 or above". It does not show the raw (uncorrected) point, so the sentence "centering alone ... falls from 0.541 to 0.503" cannot be read off the figure; the reader has Table 2 for that. Figure 9 (`fig_d_ablation`) does show raw, but only at one layer per model. The median-and-range view over all 26 layers is the stronger artifact for the main text, so keep panel (b).

### Table 3 (`headline_main`, column, page 7)

**(1) Takeaway.** After ABTT the six encoders route within 3.3 points of each other (Base spread 39.3), and the character n-gram row is as good as or better than every frozen encoder. The Spread row shows the first point quickly. The second point is where the table works against the text:
- The text says LaTa and Qwen3-0.6B rank *below* the n-gram reference with intervals excluding zero, and that LaTa routes 3.7 points below it.
- The brackets in Table 3 are marginal per-cell intervals, and they overlap heavily: LaTa ABTT [.948,.991] against n-gram [.976,.996], and Qwen3 [.948,.990].
- A reviewer who compares brackets will conclude "tie" and will see the text as overclaiming. The supporting evidence is the paired differences in Table 18, which Table 3 never points to. **This is the most important fix among Ian's floats.**

**(2) Legibility.**
- Booktabs; `\footnotesize` (9 pt) values with `\scriptsize` (8 pt) intervals. Readable in print, but the interval format "[.910,.963]" has no space after the comma and no leading zero, while Tables 17-19 print "[0.910, 0.963]".
- Decimals are consistent: three for AUROC, one for percent.
- Model order matches Figures 1-3.
- The block header "Fine-tuned; Base = fine-tuned, ABTT = fine-tuned + ABTT" is awkward.
- Small mismatch: Spread for ABTT ranking prints 0.016 (from rounded cells), while Table 18 prints 0.015. The caption defines Spread from printed values, so it is defensible, but a reviewer may still notice.

**(3) Caption.** It defines every symbol. It lacks the reading instruction that matters: that marginal brackets are not the test.

**(4) Does it carry the prose?** It carries the convergence and fine-tuning paragraphs. It does not carry the "below the reference" claims (see (1)). No existing figure or table would serve Section 7 better. A Base-to-ABTT dumbbell plot would show convergence faster, but it would cost space and drop the fine-tuned and n-gram rows.

### Consistency across the main-text floats (all six, including James's)

- Fonts: every figure uses DejaVu Sans at 8.0-8.6 pt printed; every table uses Times at 9/8 pt. This is consistent.
- Model colours and markers are consistent across Figures 1, 2 and 3 (from the shared `PANEL` table in `scripts/paper/reframe/paper_figures.py`).
- Captions sit below every float, as `formatting.md` requires ("Captions should be placed below figures/tables").
- Panel-label style varies: Figure 3 uses left titles "(a) ...", Figure 2 puts "(a) ..." in the x label, and Figure 1 uses model titles. This is minor; leave it.
- Tables 1-3 all use `\footnotesize`, booktabs and no bold headers. They are consistent with each other but not with the appendix (below).

---

## (a) Fixes to apply now, by severity

| # | Sev. | Float | Problem | Exact fix | Main-text effect | Effort |
|---|---|---|---|---|---|---|
| A1 | High | Table 3 | Overlapping marginal brackets contradict "below the reference, interval excluding zero" (LaTa and Qwen3 ranking, LaTa routing). | In `scripts/paper/reframe/spine_tables.py::render_headline`, add `$^{\downarrow}$` to the ABTT cells whose paired "Char. n-grams - ABTT" interval in Table 18 excludes zero (LaTa AUROC 0.971, Qwen3 AUROC 0.973, LaTa DirAcc 86.1; mT5-base AUROC is borderline at [0.000, 0.027], so leave it unmarked, matching the text's "touching zero"). Better to drive this from the `ci_pq` difference CSV than to hard-code it. Append to the caption: "$\downarrow$: below the n-gram reference, paired interval excluding zero (Table~\ref{tab:headline_ci_diffs}); overlapping brackets do not imply a tie." | 0 lines (measured: page 7 glue absorbs the two extra caption lines; spare stays 1.70) | Quick |
| A2 | Med | Figure 1 | Raw T5 and embedding-trained groups are not identified; the caption has no takeaway and no D; Qwen3/KaLM markers merge into a bar. | Generator `scripts/paper/reframe/paper_figures.py::fig_depth` (prototype diff in `proto/paper_figures_v2.diff`): gridspec of 7 columns with a 0.22-wide spacer between panels 3 and 4; group labels "Raw T5 encoders (no embedding objective)" and "Embedding-trained encoders" with a rule, in the row the legend used; the Baseline/ABTT key moved into the empty gray band of the LaBSE panel; `markersize=0` for models with more than 12 layers; line width 1.2. Printed height is unchanged (115 pt). New caption (measured, same 4 lines): "Test ranking AUROC at every layer of the six panel models. Dotted, open markers: baseline (mean-pooled hidden states). Solid, filled: ABTT (remove the training mean and the top $D \le 10$ principal components), fit on training embeddings only. Gray band: AUROC below 0.70, the collapsed range; gray line: chance. The three raw T5 encoders collapse at mid-depth; after ABTT every layer scores 0.962 to 0.987." | 0 lines (measured) | Quick (about 20 min incl. test update in `tests/test_paper_figures.py` if it pins the layout) |
| A3 | Med | Figure 3 | Categorical D axis not disclosed; no takeaway for (b); k=10 line invisible. | Generator: `axk.axvline(10, color="#555555", lw=0.8, ls=":")` in `fig_localize`. Caption (measured, 5 lines as now): "Test ranking AUROC at the 26 collapsed layers of LaTa (10 layers), PhilTa (9) and mT5-base (7); lines: median over a model's collapsed layers; bands: their range; dashed line: AUROC 0.90. (a)~Zeroing the $k$ coordinates of largest mean absolute value; dotted: $k$ random coordinates (mean of five draws). Up to $k=10$ (vertical line) the best layer reaches 0.794. (b)~Removing the top $D$ principal components ($D=0$: centering only; ticks follow the selection grid, not evenly spaced in $D$). From $D=3$ every layer passes 0.90. All fits use training embeddings." (Type a plain apostrophe in "model's". My prototype's `\'s` became an accent: "modelś".) | 0 lines (measured) | Quick |
| A4 | Low | Table 3 | Awkward fine-tuned header. | `spine_tables.py` line 572: replace it with `\emph{Fine-tuned (Base: no correction)}`. | 0 | Quick |

### Appendix, light pass (Ian's floats)

| # | Sev. | Float(s) | Problem | Fix | Effort |
|---|---|---|---|---|---|
| B1 | Med | Tables 23-28, 32-34 (`taskA_*`, `taskB_*`) | Captions use code identifiers (`baseline`, `abtt_optimal`, `sif_only`, `sif_abtt_fixed`, `sif_abtt_optimal`) for methods the paper calls Base/ABTT/SIF/SIF+ABTT. | Caption strings in `scripts/resubmit/build_per_layer_tables.py`. | Quick |
| B2 | Med | Same tables | The bold "selected" row also bolds the **Base** columns at the ABTT-selected layer. These read as the headline Base cell but are not: Table 23 PhilTa bolds Base 0.539 at layer 9, while Table 3's Base is 0.939 at layer 1. | In `_bold` callers (`build_per_layer_tables.py` lines 349-471), bold only the ABTT/selected-method columns and the layer number, or add an underline for the Base-selected layer. | Quick |
| B3 | Med | Tables 24, 25, 27, 28, 29 (`appendix_lasttok_comparison`), 32-34; Figure 9 lower panel | DirAcc@1 and assignment accuracy appear as fractions (0.885), while the main text and Tables 3, 17, 20, 21, 30, 31 use percent. | Multiply by 100 with one decimal in `build_per_layer_tables.py`, `build_lasttok_comparison_table.py`, and the `train_dir_acc_at_1` axis in `scripts/paper/reframe/abtt_subspace_whiten.py`. Fallback: add "(fractions)" to each caption. | Bigger (30-60 min, many tables) |
| B4 | Med | Longtables 26-28, 32-34 | Caption is above the table, against `formatting.md`'s "below". Continuation pages do not repeat the model name (page 41: KaLM-mini layers 14-24 with a blank Model column). | In `build_per_layer_tables.py` (lines 209-216, 283), emit `\caption` inside `\endlastfoot`, and a "(cont.)" row in `\endhead`. | About 30 min |
| B5 | Med | Figure 4 (`fig_release_gap_6model`) | Method colours clash with model colours (blue = ABTT here, LaTa elsewhere; orange = SIF here, PhilTa elsewhere). Legend says "ABTT-only"/"SIF-only" while Figure 1 says "ABTT". Baseline style differs from Figure 1. | `scripts/resubmit/visualize_resubmit.py` `METHOD_COLORS`/`METHOD_LABELS`: use non-model hues (ABTT black, SIF #7B3294 or dark gray dash-dot, baseline #8a8a8a dotted), labels "ABTT", "SIF", "Baseline". Re-render. | Quick |
| B6 | Med | Figure 6 (`paper_fig_density_2x2`) | The caption says "Equivalent / non-equivalent pairs", a term used nowhere else (the paper says same-source / same-directory). It does not define the vertical lines or the "gap=" label. "Layer 8 is not PhilTa's worst layer" reads as defensive without the selection rule. | Caption: "Same: same-directory pairs; Different: different-directory pairs; dashed and dotted lines: their means; gap: their difference. Layer 8 is the collapsed layer with the highest training AUROC (`compute_collapsed_layer`, `visualize_resubmit.py` l.296); PhilTa's lowest test AUROC, 0.538, is at layer 10." | Quick |
| B7 | Low | Figure 7 (`fig_gen_geometry`) | T5-v1.1-base is maroon circles here but black X in Figure 2. It reuses LaTa's circle marker. | `scripts/paper/reframe/gen_ft_geometry.py` line 57: `"#000000", "X"`. Re-render. (Spine items 1-2 are already done: the 0.76 guide line and both captions are correct.) | Quick |
| B8 | Low | Table 23 (`taskA_main`) | Hyphen-minus in the cosine-gap column ("-0.048", 9 cells). The other tables use $-$. | `_fmt3` in `build_per_layer_tables.py`: wrap negatives in `$-$`. | Quick |
| B9 | Low | Table 21 (`lexical_baselines`), Table 20 (`finetune_ceiling`) | Row name "TF-IDF char 3-5" against "Char. n-grams" in Tables 3 and 17. Table 21's caption is about 30 lines of method prose. Table 20's caption repeats "a ceiling at this training budget rather than an asymptote" three times. | Rename the row to "Char. n-grams (TF-IDF 3-5)". Move the BM25/rescaling prose into Appendix H text; collapse the three per-model sentences in Table 20 into one. Generators: `scripts/resubmit/lexical_baselines.py`, `scripts/resubmit/rebuild_finetune_ceiling_tex.py`. | Quick |
| B10 | Low | Table 10 (`d2_controls_abtt`) | Text claims "eight of 48 control layers below the panel band", but the table does not mark them. | In `scripts/paper/reframe/d2_controls.py`, italicise or underline ABTT cells < 0.962 and say so in the caption. | Quick |
| B11 | Low | Appendix layout | Float order leaves large gaps: page 16 right column empty, page 18 holds only Figure 6, pages 27, 28 and 35 have an empty right column, and Table 32 starts on page 47 with 16 rows then breaks. | Move the `fig:density` block in Appendix C above the two `figure*` gap figures (LaTeX keeps figure numbers in order, so the column figure waits behind them). Consider `[p]` for Tables 17-20 and the full-width per-layer tables. | About 30 min, cosmetic |
| B12 | Low | Style across appendix tables | Bold headers in most appendix tables but plain in Tables 1-3 and 8-12. "Max PC1" / "PC1max" / "PC1 before". Type labels "T5 raw / BERT emb. / Decoder emb." (Table 4) against "T5 / Enc. / Dec." (Tables 1, 7). "raw / Base / baseline / (base)". Interval formats "[.910,.963]" and "[0.910, 0.963]". | Pick the main-text convention (plain headers, PC1$_{\max}$, T5/Enc./Dec., Base) and apply it in the generators when next touched. | Bigger, low value before the deadline |

Appendix items that are fine as they are: Tables 4, 5, 6, 9, 11, 12, 14, 15, 17, 18, 19, 22, 30, 31; Figures 5, 9 (apart from units), 10-13 (the t-SNE/UMAP palette is not colour-blind safe for red against pink, but these are illustrations).

---

## (b) Judgement calls for Ian

1. **Table 3: keep the per-cell brackets (variant A) or drop them (variant B)?**
   - Variant B removes the seven bracket rows, keeps the markers, and points to Tables 17 and 18. It saves 5.8 main-text lines (measured: spare goes from 1.70 to 7.46) and makes the table readable at a glance (`renders/VB_T3_300dpi.png`). Every Section 7 claim is a paired-difference claim, and Table 18 holds those intervals, not Table 3.
   - Against B: the spine promised "T3 with intervals", and reviewers like to see uncertainty in the headline table.
   - **Recommendation:** ship A now. Keep B ready as the cheapest way to buy about six lines if any late addition needs room.
2. **Add T5-v1.1-base and T5-base panels to Figure 1 (8 panels)?** This would put the paper's most distinctive result (claim 3b) into the main figure. It costs no height, but the panels get narrower. The caption's "0.962 to 0.987" would have to be scoped to the panel (T5-v1.1-base reaches only 0.927 after ABTT). It also adds a cross-check against Table 1 two days before submission. **Recommendation:** not now. Table 1 and the Section 5 heading already carry the contrast.
3. **Swap candidates.** I checked every artifact in `overleaf_drafts/figures/` and `tables/` and the generators in `scripts/paper/reframe/` and `scripts/resubmit/`. None is better:
   - `fig_release_aucroc_6model` is the older version of Figure 1. It is larger, uses normalized depth, and brings back SIF, which has left the main text.
   - `fig_localize_col` stacks the same panels in one column. It is taller (3.2 in) and would cost about 6 lines.
   - `fig_d_ablation` shows one layer per model, which is weaker than the 26-layer median-and-range of Figure 3(b).
   - `fig_geometry_vs_auroc` and `fig_standardized_gap` belong to James's diagnostics paragraph.
   - No routing figure exists that would beat Table 3 per line of space.
   - **Recommendation:** keep all three.
4. **Raw point in Figure 3(b).** Adding a "raw" tick before D=0 (as in Figure 9) would show centering's drop from 0.541 to 0.503 directly. It costs no height, but the caption needs another clause, and Table 2 already states the number. **Recommendation:** optional; skip unless A3's caption has room after review.
5. **Unused files** in `overleaf_drafts/` that are not compiled (`taskA_headline*.tex`, `taskB_headline*.tex`, `panel_2x2.tex`, `fig_release_aucroc_6model.*`, `fig_geometry_vs_auroc.*`, `fig_localize_col.pdf`) could be pruned before the code release. This has no effect on the PDF.

## Notes on the prototype

- The `paper_figures_v2.py` prototype ran from the repo root with the `localLatin` env and wrote only to the scratchpad. The current figures regenerate from the committed CSVs (byte differences are matplotlib metadata only).
- The variant builds share every other file with main 2d5529d. The spare-line figures come from comparing the bottoms of the left and right columns on page 8 (13.55 pt per line).
