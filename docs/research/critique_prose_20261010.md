# Prose critique of Ian's sections, 10 October 2026

Paper: "Masked, Not Missing: Mid-Depth Retrieval Collapse in T5-Family Encoders", `overleaf_drafts/acl_latex.tex` and `related_work.tex` at `main` 2d5529d (the 8 October PDF in the repo matches this source: the main text ends at the foot of page 8 and Limitations starts on page 9).

Read-only review. Nothing in the repo was changed.

**How to use this.** Each flagged item gives the current LaTeX, which you can search for, and a replacement. **(a)** means apply now. **(b)** means a judgement call for you. Δ is the change in words. The running total for the (a) items in the main text is at the end of Section 1. Applied together, they come out shorter than the current text, so they pay for the few that add words. Each replacement keeps its numbers, hedges, scope and citations, and stays inside the spine's wording limits.

---

## 0. Verdict

James asked whether your sections are clear on a first read and whether they get across the points you want. Mostly yes. The argument is sound, the numbers agree across sections, and the house rules hold: no em-dashes, no "Latin department", no "family" for directories, no "notably", "crucially" or "importantly". Read in order, the bold run-in headings give the argument. What a first-time reviewer will stumble on falls into four groups:

1. **Terms used before they are defined.** "panel", "the projection", "train-selected", "T5 v1.1 layout", plus "raw" and "baseline" each used in two senses.
2. **One paragraph whose point comes last.** Intro paragraph 5 opens with "Controls show which models collapse." Its finding, the layout, arrives in its fifth sentence, after a 37-word sentence about T5-efficient-base.
3. **Points that arrive late in Sec. 7.** The fine-tuning paragraph states its heading's claim in its fifth sentence. The threshold paragraph never says what worry it answers.
4. **Templated rhythm that reads as machine-written.** All four analysis sections open with "This section asks…". Thirteen sentences in the main text turn on ", so". About 15 colons join claim to evidence in three pages. "X, not Y" aphorisms cluster in the Discussion. ABTT is also called "the projection", "the correction", "the repair" and "the restoration", often in neighbouring sentences.

| Part | Verdict |
|---|---|
| Abstract | Small fixes (one sentence is hard to parse; consider swapping two sentences) |
| 1 Introduction | Small fixes, but paragraph 5 needs a real rewrite (given below) |
| 2 Related Work | Small fixes (gap sentences; one sentence with three unrelated clauses) |
| 3 Testbed and Protocol | Small fixes (define "train-selected" and "baseline"; make the model count add up to thirteen) |
| 4 Collapse and projection paragraphs | Ready, two small fixes |
| 5 English and fine-tuning paragraphs | Small fixes (the fine-tuning paragraph needs a motive) |
| 6 Three-directions paragraph | Small fixes (one should-fix on the removed-subspace sentence) |
| 7 Routing and Limits | Small fixes; the fine-tuning paragraph needs reordering |
| 8 Discussion | Small fixes (aphorism cluster, the Mikkelsen sentence, "raw states") |
| Limitations | Small fixes (split three long sentences) |
| Ethical Considerations | Ready (one sentence split, optional) |
| Appendix (yours) | Small fixes; the Reference Systems appendix is one 500-word paragraph and needs breaking up |

### The five changes that would most improve the first read

1. **Rewrite Intro paragraph 5 so it opens on the finding** (item I-7). Open with "Across the thirteen models we test, the collapse tracks the T5 v1.1 layout" and define the layout in that sentence. Recast the T5-efficient-base sentence as "T5-base also saw supervised tasks in pretraining, so we add…", which says why the control exists. Cut the duplicate closing sentence. This is the paper's most distinctive result, and it currently reads as a list of controls with the conclusion buried at the end. Net −6 words.
2. **Define each term at first use and keep one name per thing.** In the intro, say "panel" (I-5) and "the projection" (I-6) where they first appear. Define "train-selected" in Sec. 3 (T-2). Give "raw" and "baseline" one meaning each (T-3, T-7, D-3). In the body, use "ABTT" and keep "the projection" for the abstract and headings. In the Sec. 7 fine-tuning paragraph, "the projection", "ABTT" and "the zero-shot projection" all appear within three sentences (R-6).
3. **Fix the abstract's hardest sentence** (A-2: "Four of the six raw T5-family encoders we test, these three and T5-v1.1-base, collapse, and…" reads as garden-path). Consider moving "masked, not missing" up so it comes right after the collapse (A-1). The title's claim currently arrives fifth.
4. **Put the point first in Sec. 7.** State the spread before and after ABTT as one comparison (R-2). Say the worry the threshold-free check answers ("the gain could come from a better-placed threshold") (R-3). Open the fine-tuning paragraph with its heading's claim, and say in plain words what "five reseedings" means (R-6).
5. **Break the template.** Vary or cut the four "This section asks…" openers (X-1). Turn roughly half the claim-colon-evidence colons and the "…, so…" sentence endings into two sentences or plain verbs (X-2, X-3). Drop "a screen, not a verdict" so the Discussion keeps one aphorism (D-2). James's "AI tells" comment is mostly about these patterns. No single sentence is to blame.

The related-work gap sentences (W-1, W-3) would be sixth on this list.

---

## 1. Main text, item by item

### Abstract

Reader test. A first-time reader takes away: T5 encoders collapse at mid-depth on a Latin retrieval task, mean cosine misses it, a train-only projection fixes all layers, the collapse goes with the T5 v1.1 layout, about three directions are responsible, and after the fix encoders route alike but do not beat n-grams. That matches spine sentences 1 to 6 and the routing coda. Two problems. The title's claim, "masked, not missing", arrives fifth, after the diagnostic. And sentence 6 has to be read twice.

**A-1 (b). Swap sentences 4 and 5** so the title claim follows the collapse directly, in the spine's order (collapse, masked, diagnostic). Δ 0.

**A-2 (a). Sentence 6 is hard to parse.** It runs to 41 words, its appositive ("these three and T5-v1.1-base,") makes "collapse," look like the end of a list, and it ends in a semicolon chain.
```tex
Four of the six raw T5-family encoders we test, these three and T5-v1.1-base, collapse, and all four share the T5 v1.1 layout; encoder-only siblings pretrained on the same corpora and two checkpoints with the original T5 layout do not.
```
→
```tex
Of the six raw T5-family encoders we test, four collapse: these three and T5-v1.1-base. All four share the T5 v1.1 layout; encoder-only siblings pretrained on the same corpora and two checkpoints with the original T5 layout do not.
```
Δ −1. Keeps the D4 scope statement word for word in substance. Optional, under (b): spend +4 words on "(gated feed-forward, untied embeddings)" after "layout". "T5 v1.1 layout" is otherwise undefined in the abstract, and reviewers will not know the version label (D4 says so itself).

**A-3 (b).** Sentence 8, "at its train-selected layer", is jargon in an abstract. "at a layer chosen on training data" is clearer but costs +3 words. Keep it if space is tight. The spine wording limit for claim 8 only needs the qualifier to be present.

### 1 Introduction

Paragraph-by-paragraph reader test:

| ¶ | First-time reader takes away | Spine requires | Gap |
|---|---|---|---|
| 1 | People read middle layers, check mean cosine, correct; does the check warn? | the practice | none |
| 2 | Latin canon law gives expert labels; two tasks; n-grams solve it; sibling models exist | the setting | the three reasons Latin "suits this test" are split by the task definitions; the sibling-model reason comes last |
| 3 | Three T5 encoders fail at mid-depth; mean cosine misses it; top-PC share is high | the surprise | "our panel" used before it is defined; "in either direction" is opaque here |
| 4 | ABTT fixes every layer, so the signal was there | masked, not missing | "The projection" appears as a new name with no introduction |
| 5 | A list of controls… then the layout | which models | **the point arrives in sentence 5 of 6**; "T5 v1.1 layout" is used, then defined; the T5-efficient-base sentence needs a reread; sentences 5 and 6 repeat each other |
| 6 | About three directions; zeroing fails; tokens in LaTa, none in mT5 | what masks it | fine; it never says why zeroing is the comparison to make (option below) |
| 7 | Four contributions; scope | four contributions | (1) names a profile, not a finding; (2) is a 46-word semicolon chain |

**I-1 (a). ¶2, move the sibling-model sentence up** so the three reasons Latin suits the test sit together: labels, then models, then the tasks and the lexical point. Move
```tex
Latin also offers T5 and encoder-only models from the same authors and pretraining corpora.
```
to just after "…with expert labels that no model shaped." Δ 0.

**I-2 (b). ¶2, "routing each witness to its source or to ``new''".** A reader does not yet know why "new" exists. Optional: "…or, when its source has no other witness, to ``new''" (+7). Sec. 3 covers it, so this can wait.

**I-3 (b). ¶3, "in either direction" says nothing until Sec. 4.**
```tex
The usual check, mean pairwise cosine, gives no reliable warning in either direction.
```
→
```tex
The usual check, mean pairwise cosine, gives no reliable warning: collapsed layers score both below and above healthy ones.
```
Δ +6. This is true, given 0.22–0.58 for the panel against 0.87–0.96 for T5-v1.1-base. Apply it if you take I-7, which saves 6.

**I-4 (b). ¶3, last sentence**, the colon followed by a fragment ("…: necessary for collapse in our models, but not sufficient"). It is acceptable as written. If you touch it, use "…at every collapsed layer; in our models a high share is necessary for collapse, but not sufficient." Δ +4.

**I-5 (a). ¶3, define "panel" where it first appears.**
```tex
Their first and last layers still retrieve, and the three embedding-trained encoders of our panel never collapse.
```
→
```tex
Their first and last layers still retrieve. Three embedding-trained encoders, which complete our six-model panel, never collapse.
```
Δ 0.

**I-6 (a). ¶4, introduce "the projection" as ABTT's other name**, and drop the repeated "linear".
```tex
All-but-the-Top \citep[ABTT;][]{mu2018allbutthetop}, fit on training embeddings only, removes the mean and the top principal components from every embedding, with no parameter update.
```
→
```tex
All-but-the-Top \citep[ABTT;][]{mu2018allbutthetop}, a linear projection fit on training embeddings only, removes the mean and the top principal components from every embedding, with no parameter update.
```
and
```tex
The projection is linear and sees no test label, so the ranking it recovers is already present in each collapsed layer.
```
→
```tex
The projection sees no test label, so the ranking it recovers is already present in each collapsed layer.
```
Δ 0.

**I-7 (a). ¶5, rewrite so the paragraph opens on the finding.** This is the most important edit in the report.
Current:
```tex
Controls show which models collapse.
The encoder-only siblings LaBERTa and PhilBERTa never fall below 0.826 and 0.883.
Within the T5 family, T5-v1.1-base collapses on the same Latin text (lowest AUROC 0.489), while the original T5-base \citep{raffel2020t5} never falls below 0.816.
T5-base also saw supervised tasks in pretraining, but T5-efficient-base \citep{tay2022scale}, with T5-base's layout and pretrained on C4 alone, without dropout and on about 30 times fewer tokens, does not collapse either.
Four of the six raw T5-family encoders we test collapse, and all four share the T5 v1.1 layout: a gated-GELU feed-forward block \citep{shazeer2020glu} and untied input and output embeddings \citep{t5v11release}.
Across the thirteen models we test, the collapse tracks the T5 v1.1 layout; we do not isolate which change is responsible.
```
New:
```tex
Across the thirteen models we test, the collapse tracks the T5 v1.1 layout: a gated-GELU feed-forward block \citep{shazeer2020glu} and untied input and output embeddings \citep{t5v11release}.
LaBERTa and PhilBERTa, encoder-only siblings of LaTa and PhilTa, never fall below 0.826 and 0.883.
T5-v1.1-base collapses on the same Latin text (lowest AUROC 0.489), while the original T5-base \citep{raffel2020t5} never falls below 0.816.
T5-base also saw supervised tasks in pretraining, so we add T5-efficient-base \citep{tay2022scale}, with T5-base's layout but pretrained on C4 alone, without dropout and on about 30 times fewer tokens; it does not collapse either.
Of the six raw T5-family encoders we test, the four that collapse all have this layout.
We do not isolate which change matters.
```
Δ −6 (word counts in this report exclude citation keys). The scope phrase "Across the thirteen models we test", the "four of six" statement and the disclaimer all survive. The T5-efficient-base sentence now says why the control is there ("…, so we add…"). The current "but" leaves the reader to work out the confound. "Within the T5 family," goes because "the original T5-base" already says it.

**I-8 (b). ¶6, say why zeroing is the comparison to make.** "Zeroing the ten largest coordinates, the usual outlier-dimension fix, restores none of them…" (+4). That one clause tells the reader the result goes against expectation.

**I-9 (a). Contribution (1) names a profile but not what it shows.** Spine contribution 1 includes the finding.
```tex
(1)~The first per-layer retrieval profile of T5 encoder states that we know of, each layer read on its own rather than averaged with another layer (Section~\ref{sec:depth}).
```
→
```tex
(1)~A per-layer retrieval profile of T5 encoder states, each layer read alone, the first we know of; it shows a mid-depth collapse that ABTT reverses (Section~\ref{sec:depth}).
```
Δ 0. "each layer read alone" keeps the "in isolation" wording limit.

**I-10 (a). Contribution (2): 46 words, a semicolon chain, and "high-share middle layers" as jargon.**
```tex
(2)~A within-family contrast: T5-v1.1-base collapses on the same text where the original T5-base does not, and every collapsing model has the T5 v1.1 layout; encoder-only siblings on the same corpora do not collapse, and the high-share middle layers also appear on English text (Section~\ref{sec:models}).
```
→
```tex
(2)~Controls showing that the collapse tracks the T5 v1.1 layout: T5-v1.1-base collapses where the original T5-base does not, encoder-only siblings on the same corpora do not, and the high-share middle layers recur on English text (Section~\ref{sec:models}).
```
Δ −8. The English clause stays "high-share", which is geometry only and not collapse.

**I-11 (b). Contribution (4), "where it stops", is vague.** Possible replacement: "(4)~A measure of what the projection gains for routing: the six encoders end within 3.3 points of each other, and none beats a character n-gram reference (Section~\ref{sec:routing})." Δ +5. It turns the contribution into a claim a reviewer can check.

### 2 Related Work (`related_work.tex`)

The grouping is right. Each paragraph ends with a sentence on what we do, but two of those sentences do not state the gap, and the third paragraph collects three topics.

**W-1 (a). ¶1, last sentence: "both statistics" makes the reader work out which two.**
```tex
We relate both statistics to retrieval at every layer (Section~\ref{sec:depth}).
```
→
```tex
We relate mean cosine and top-PC share to retrieval at every layer (Section~\ref{sec:depth}).
```
Δ +3. Pay for it with W-2.

**W-2 (a). ¶1, merge the 8-word second sentence and clarify "led by".**
```tex
Contextual embeddings occupy a narrow cone, and the usual check is mean cosine over random pairs \citep{gao2019degeneration, ethayarajh2019contextual}.
It is not an appropriate measure of isotropy \citep{rudman2022isoscore}.
```
→
```tex
Contextual embeddings occupy a narrow cone, usually checked by mean cosine over random pairs \citep{gao2019degeneration, ethayarajh2019contextual}, which is not an appropriate measure of isotropy \citep{rudman2022isoscore}.
```
Δ −2. Also: "a topological signature led by effective rank moderately predicts retrieval" → "topological statistics, chiefly effective rank, moderately predict retrieval" (Δ −1). Check that this still describes Rottach et al. correctly.

**W-3 (a). ¶2, put the two BERT sentences together.** Right now "In BERT" opens sentence 2 and again sentence 6. Move
```tex
In BERT most outliers sit on \texttt{[SEP]}, the period and the comma \citep{bondarenko2023quantizable}, and removing punctuation and special tokens before pooling improves sentence similarity \citep{fusterbaggetto2022anisotropy}.
```
to directly after the "In BERT they are tied to a few LayerNorm weights…" sentence. The paragraph then runs BERT, then decoders, then T5, which leads into our T5 comparison. Δ 0.

**W-4 (a). ¶3, one sentence carries three unrelated findings** (a rule-of-three tell):
```tex
Whitening beats zeroing outlier dimensions in a multilingual encoder \citep{haemmerl2023anisotropy}, centering alone gives embedding models gains under one point \citep{ren2026meanbias}, and after fine-tuning isotropy corrections become ineffective \citep{rajaee2021finetuning}.
```
→
```tex
Whitening beats zeroing outlier dimensions in a multilingual encoder \citep{haemmerl2023anisotropy}.
Centering alone gives embedding models gains under one point \citep{ren2026meanbias}, and isotropy corrections stop helping after fine-tuning \citep{rajaee2021finetuning}.
```
Δ 0.

**W-5 (a). ¶3, "We use ABTT as a probe." is stranded** after the Latin lexical sentence, where it reads as a non sequitur. Delete it at the end and attach it to the Timkey sentence:
```tex
\citet{timkey2021rogue} apply them at every layer of four models and find that rogue dimensions obscure representational quality; our reading of the collapse as masked follows theirs.
```
→
```tex
\citet{timkey2021rogue} apply them at every layer of four models and find that rogue dimensions obscure representational quality; our reading of the collapse as masked follows theirs, and we use ABTT as a probe.
```
Δ +2 overall (+7 here, −5 for the deleted standalone sentence). Paragraph 3 then ends on Latin lexical methods versus dense encoders, which leads into Sec. 3's "lexically solvable by design".

**W-6 (b).** Paragraph 3 holds layerwise analysis, post-hoc correction, an OOD precedent for routing, and Latin lexical-versus-dense work. If you have a line to spare, a fourth run-in heading, "Routing and lexical baselines.", before "For routing, the nearest precedent…" would make the grouping honest. It costs about one line, so only do it if the other cuts free one.

### 3 Testbed and Protocol

Clear and procedural. Each setup paragraph says what it says. The fixes are definitions.

**T-1 (a). "positive pairs" appears one paragraph before "positives" is defined.**
```tex
We split the witnesses into 847 training and 858 test witnesses, with 565 and 596 positive pairs;
```
→
```tex
We split the witnesses into 847 training and 858 test witnesses, with 565 and 596 same-source pairs;
```
Δ 0.

**T-2 (a). Define "train-selected".** The abstract and Sec. 7 rely on the term, and right now nothing defines it. This also removes a forward reference.
```tex
Each headline number (Table~\ref{tab:headline}) reads a model at its layer of highest training AUROC for ranking, or of highest training routing accuracy (DirAcc@1, defined below) for routing (Table~\ref{tab:selected_layers}).
```
→
```tex
Each headline number (Table~\ref{tab:headline}) reads a model at its \emph{train-selected} layer: the one of highest training AUROC for ranking, or of highest training DirAcc@1 for routing (Table~\ref{tab:selected_layers}).
```
Δ −1.

**T-3 (a). "baseline" is used for "uncorrected" before it is defined.** Three sentences later, "baselines" also names SIF and whitening (T-7).
```tex
A \emph{collapsed layer} has baseline test AUROC below 0.70.
```
→
```tex
A \emph{collapsed layer} has \emph{baseline} (uncorrected) test AUROC below 0.70.
```
Δ +1.

**T-4 (a). Routing: the witness is a member of its own directory.**
```tex
Each test directory scores a witness by the highest cosine between the witness and any member.
```
→
```tex
Each test directory scores a witness by the highest cosine between the witness and any other member.
```
Δ +1. Without "other", a careful reader concludes the true directory always scores 1.

**T-5 (a). "complete the main text" is odd.**
```tex
Four raw controls complete the main text, two of them the encoder-only siblings LaBERTa and PhilBERTa.
```
→
```tex
The other four are raw controls, two of them the encoder-only siblings LaBERTa and PhilBERTa.
```
Δ −1.

**T-6 (a). The model count does not add up to the "thirteen" of Sec. 1 and Sec. 5.** Sec. 3 accounts for twelve and never mentions T5-efficient-base.
```tex
Sentence-T5 \citep{ni2022sentencet5} and SPhilBERTa \citep{riemenschneider2023sphilberta} appear only in Table~\ref{tab:all_models}.
```
→
```tex
Table~\ref{tab:all_models} lists all thirteen models, adding Sentence-T5 \citep{ni2022sentencet5}, SPhilBERTa \citep{riemenschneider2023sphilberta} and T5-efficient-base (Section~\ref{sec:models}).
```
Δ +5.

**T-7 (a). Keep "baseline" for "uncorrected".**
```tex
SIF frequency weighting \citep{arora2017sif} and reduced-rank whitening \citep{su2021whitening} are baselines in Appendix~\ref{app:sif_variants}.
```
→
```tex
We compare SIF frequency weighting \citep{arora2017sif} and reduced-rank whitening \citep{su2021whitening} in Appendix~\ref{app:sif_variants}.
```
Δ 0.

**T-8 (b).** "we also call it \emph{the projection}" openly announces a second name. Keep the definition, but adopt a rule for the body: "ABTT" in method and result sentences, "the projection" only in the abstract, intro and headings. See X-4.

**T-9 (b).** A first-time reader will want to know passage length and that most sources have one witness. A clause such as "(median 98 mT5 tokens; 545 directories hold a single witness)" after "840 directories" (+9) answers both and explains why "new" is common. Only add it if space allows.

### 4 Collapse by Depth (your paragraphs and the opener)

The bold sentences read as the argument. The reader takeaway for both your paragraphs matches spine claims 1 and 2.

**Q-1 (b). The opener asks two questions and answers one.**
```tex
In three of the six panel models most middle layers rank near chance, and a projection fit on training embeddings restores them all: the signal is masked, not missing.
```
→
```tex
In three of the six panel models most middle layers rank near chance and mean cosine does not warn of it; a projection fit on training embeddings restores every layer.
```
Δ +1. This drops the third use of "masked, not missing", which already appears in the title, abstract and intro (see X-5). If you keep the phrase, leave the sentence as it is.

**Q-2 (a). The Timkey sentence is awkward ("bears out the reading…, that…, here at…").**
```tex
This bears out the reading of \citet{timkey2021rogue}, that a few dominant dimensions hide representational quality, here at layers that rank near chance.
```
→
```tex
This extends the reading of \citet{timkey2021rogue}, that a few dominant dimensions hide representational quality, to layers that rank near chance.
```
Δ −2. Within the spine's novelty line: "the idea is Timkey's; layers at chance… is new".

**Q-3 (a). Two consecutive sentences end in ", so…".** Change the second:
```tex
A cutoff on training AUROC picks the same 26 layers, so no test score defines the collapse.
```
→
```tex
A cutoff on training AUROC picks the same 26 layers; no test score defines the collapse.
```
Δ −1.

### 5 Collapse Across Models (English and fine-tuning paragraphs)

English paragraph. Reader takeaway: "the high-share middle layers are not a Latin artefact; geometry only". That matches 3(c), and the motive comes first. Good.

**M-1 (a). Small trims.** "layers 4 to 11 keep a top-PC share" → "layers 4 to 11 have a top-PC share" (Δ 0; "keep" implies they had it before). "In these models, then, the high-share middle layers do not depend on Latin input." → "In these models the high-share middle layers do not depend on Latin input." (Δ −1; "then" is filler before a summary sentence).

**M-2 (b). Heading "In three T5 v1.1-layout models, …".** The hyphenated compound is awkward, and "high-share" means nothing to a headings-only reader. "In three models with the T5 v1.1 layout, high top-PC share in the middle layers appears on English text too." This adds a word or two in bold, which could cost a line.

Fine-tuning paragraph. Reader takeaway: "fine-tuning fixed the last layer only". That is right, but the paragraph opens with the method and never says what question fine-tuning answers. A reviewer has to infer "does an embedding objective repair the middle?"

**M-3 (a). Give the paragraph a motive and cut the colon connector.**
```tex
We fine-tuned LaTa contrastively on 499 training pairs (Appendix~\ref{app:reference_systems}).
Fine-tuning lifts the last layer from 0.938 to 0.984 AUROC.
Layers 2 to 11 stay at 0.50 to 0.57.
Geometry barely moves: peak top-PC share is 0.945 after fine-tuning and 0.952 before (Table~\ref{tab:ft_lata_layerwise}).
This agrees with \citet{haemmerl2023anisotropy}: a sentence-tuned multilingual encoder has no large outlier dimension at its output layer but one at a middle layer.
```
→
```tex
We fine-tuned LaTa contrastively on 499 training pairs, adding the embedding objective it lacks (Appendix~\ref{app:reference_systems}).
It lifts the last layer from 0.938 to 0.984 AUROC but leaves layers 2 to 11 at 0.50 to 0.57.
Peak top-PC share barely moves (0.952 before, 0.945 after; Table~\ref{tab:ft_lata_layerwise}).
This agrees with \citet{haemmerl2023anisotropy}, whose sentence-tuned multilingual encoder has a large outlier dimension at a middle layer but none at its output layer.
```
Δ +3. The appendix calls this a weak test. The new clause only states what fine-tuning adds and claims nothing more. Two short sentences become one, and the colon connector goes.

### 6 Localizing the Collapse (the three-directions paragraph)

Reader takeaway: "one component holds 94% of the variance yet recovers only 45%; three bring every layer to 0.914; the removed directions carry no signal". That matches claim 4. The heading's claim only arrives in sentence 6, but the build-up is the prediction-then-outcome form the spine asks for, so it is fine.

**L-1 (a). A colon connector plus a restatement.**
```tex
Three components suffice for 0.91: they bring all 26 layers to 0.914 or above.
```
→
```tex
Removing three brings all 26 layers to 0.914 or above.
```
Δ −4. The next sentence (mT5-base keeps improving up to ten) carries the spine's "three suffice for 0.91; mT5-base keeps improving" limit.

**L-2 (a, and should fix). The last sentence rests on a near-tautology** that the appendix itself admits. Appendix E says: "At the collapsed layers the removed subspace holds a median 99% of the centered variance of vectors that already rank at chance, so its chance-level ranking there is close to a tautology." In the main text, the informative half is the retained subspace at 0.977, so lead with it:
```tex
Ranking with the ten removed components alone gives a median AUROC of 0.489, against 0.977 for the retained subspace: at these layers the mask occupies a few directions and the signal lies outside them.
```
→
```tex
Ranking within the retained subspace gives a median AUROC of 0.977, and within the ten removed components 0.489: at these layers the signal lies outside the mask.
```
Δ −7. A reviewer who reads Appendix E will otherwise catch the main text leaning on the half the appendix calls tautological.

**L-3 (b). "Centering alone ($D=0$) does not help, as expected:"** Say why it was expected. Otherwise cut "as expected", which Table 2's dagger already records (Δ −2).

### 7 Routing and Limits

Every bold heading is right. The points arrive late and the paragraphs are number-dense.

**R-1 (b). Opener.** "This section asks what the projection gains for routing and where it stops." is the fourth "This section asks". See X-1. One option: "We now turn from ranking to routing, and to what the projection cannot do." (Δ +1).

**R-2 (a). Paragraph 1: state the spread as one before-and-after comparison, using the table's word.**
```tex
After ABTT, the six encoders route within 3.3 DirAcc@1 points of each other (95\% CI 1.8 to 6.0), from 86.1 to 89.4, at the train-selected layers of Table~\ref{tab:headline}.
Without correction they span 39.3 points (34.6 to 43.7), from 46.6 for mT5-base to 85.9 for KaLM-mini.
```
→
```tex
At the train-selected layers of Table~\ref{tab:headline}, the DirAcc@1 spread across the six encoders falls from 39.3 points without correction (95\% CI 34.6 to 43.7) to 3.3 after ABTT (1.8 to 6.0).
Uncorrected, they run from 46.6 (mT5-base) to 85.9 (KaLM-mini); after ABTT, from 86.1 to 89.4.
```
Δ +1. This is the spine's claim-7 wording ("the spread … falls from 39.3 to 3.3 points"), and it matches Table 3's "Spread" row.

**R-3 (a). Paragraph 2: say what the threshold-free check rules out.**
```tex
Because routing applies a threshold $\tau$ fit on training pairs, we also score the existing-versus-new decision without it, as the AUROC of each test witness's best-match cosine, existing against new.
```
→
```tex
The gain could come from a better-placed threshold $\tau$, so we also score the existing-versus-new decision without one, as the AUROC of each test witness's best-match cosine, existing against new.
```
Δ 0.

**R-4 (a). Paragraph 2: "The assignment-accuracy gain" refers to a gain the paragraph never states.** The gain mentioned so far is in DirAcc@1.
```tex
The assignment-accuracy gain also holds at each setting's best test threshold and under a finer threshold grid (Appendix~\ref{app:ci_pq}).
```
→
```tex
ABTT's gain in assignment accuracy also holds at each setting's best test threshold and under a finer threshold grid (Appendix~\ref{app:ci_pq}).
```
Δ +2.

**R-5 (b). "so their routing gain cannot come from a ranking repair".** LaBSE's AUROC still rises 0.956 → 0.987, so "cannot" invites an objection. "can owe little to a ranking repair" keeps the point at the same length.

**R-6 (a). Paragraph 4: open on the heading's claim, use one name for the method, and say what "reseedings" means.** Right now the claim ("adds about two routing points") arrives in sentence 5. "the projection", "ABTT" and "the zero-shot projection" alternate within three sentences. Sentence 6 has parentheses inside a clause and runs past 40 words. "Reseedings of the query and reference witnesses" also describes a protocol (test witnesses split into queries and references) that Sec. 3 never introduces.
Current:
```tex
As a supervised reference, we fine-tune LaTa, Qwen3-0.6B and KaLM-mini contrastively on 499 training pairs (Appendix~\ref{app:reference_systems}).
These references are flattered: 206 of the 535 existing test witnesses (38.5\%) sit in a directory that supplied a training pair.
Fine-tuned Qwen3-0.6B and KaLM-mini rank above the character n-gram reference in point estimate (0.996 and 0.997 AUROC; Table~\ref{tab:headline}).
After fine-tuning, the projection no longer helps ranking: AUROC falls slightly in point estimate in all three fine-tuned models (by 0.002 to 0.014; every interval includes zero), in line with \citet{rajaee2021finetuning}.
For routing, fine-tuning plus ABTT beats the zero-shot projection by 1.8 points for Qwen3-0.6B (92.3 against 90.5) and by 2.1 for KaLM-mini (93.0 against 90.9); on LaTa the two tie at 87.7.
These are means over five reseedings of the query and reference witnesses (Table~\ref{tab:finetune_ceiling}, Appendix~\ref{app:reference_systems}); Table~\ref{tab:headline} shows the single-split cells, where the gains are 2.0 and 3.1 points (only KaLM-mini's interval excludes zero) and LaTa falls 0.9 points.
```
New:
```tex
As a supervised reference, we fine-tune LaTa, Qwen3-0.6B and KaLM-mini contrastively on 499 training pairs (Appendix~\ref{app:reference_systems}).
For routing, fine-tuning plus ABTT beats zero-shot ABTT by 1.8 points for Qwen3-0.6B (92.3 against 90.5) and by 2.1 for KaLM-mini (93.0 against 90.9); on LaTa the two tie at 87.7.
These are means over five random query/reference splits of the test witnesses (Table~\ref{tab:finetune_ceiling}).
On the single split of Table~\ref{tab:headline} the gains are 2.0 and 3.1 points (only KaLM-mini's interval excludes zero), and LaTa falls 0.9.
These references are flattered: 206 of the 535 existing test witnesses (38.5\%) sit in a directory that supplied a training pair.
Fine-tuned Qwen3-0.6B and KaLM-mini rank above the character n-gram reference in point estimate (0.996 and 0.997 AUROC).
After fine-tuning, ABTT no longer helps ranking: AUROC falls slightly in point estimate in all three models (by 0.002 to 0.014; every interval includes zero), in line with \citet{rajaee2021finetuning}.
```
Δ −6. Every number, both interval statements and the Rajaee citation are kept. "Random query/reference splits of the test witnesses" follows Appendix H's definition of a reseeding (each redraw keeps the train/test split and re-picks which test witnesses are queries). If you prefer to keep "reseedings", define it once in Sec. 3.

**R-7 (a). Paragraph 3: "cells" is table jargon in prose.**
```tex
The best projected cells of Table~\ref{tab:headline} tie the reference (Section~\ref{sec:testbed}):
```
→
```tex
The best ABTT results in Table~\ref{tab:headline} tie the reference:
```
Δ −1. The reference is defined two pages earlier, and the heading names it.

**R-8 (b). Paragraph 3 ends on an abstract summary sentence.**
```tex
Our claims thus concern where the signal sits in the encoders and what masks it.
```
→
```tex
Our claims therefore concern where the signal sits and what masks it, not whether encoders beat surface overlap.
```
Δ +3. This says concretely what the paper does not claim. Alternatively, delete the sentence (Δ −15), since Limitations already says the same.

### 8 Discussion

Reader takeaway: fit ABTT before reading a layer; screen with top-PC share, not mean cosine; why the layout collapses is open; encoder choice matters little after ABTT, and the scholar decides. That matches spine Sec. 4 row 8. The problems are tone: an aphorism cluster ("cheap insurance", "a screen, not a verdict", "not a cutoff validated", "carriers…, not a cause") and colon connectors.

**D-1 (a). Paragraph 1, sentences 1 and 2: remove the colon connector and the repeated "panel".**
```tex
Before reading an intermediate layer of a frozen encoder, fit a projection on training embeddings: in our panel it was cheap insurance.
In the six panel models it scored above the baseline in point estimate at all 100 layers in ranking and routing (Appendix~\ref{app:per_layer}), but routing after it still varies by layer, so the layer still matters.
```
→
```tex
Before reading an intermediate layer of a frozen encoder, fit a projection on training embeddings.
In the six panel models it scored above the baseline in point estimate at all 100 layers in ranking and routing (Appendix~\ref{app:per_layer}), which makes it cheap insurance.
Routing after it still varies by layer, so the layer still matters.
```
Δ −3.

**D-2 (a). Cut the second aphorism.**
```tex
The share is a screen, not a verdict: a high share is necessary for collapse here but not sufficient, and it does not order the layers inside the collapsed regime.
```
→
```tex
A high share is necessary for collapse here but not sufficient, and it does not order the layers inside the collapsed regime.
```
Δ −8.

**D-3 (a). "raw" means "not embedding-trained" everywhere else in the main text.**
```tex
For raw states, check top-PC share, not mean pairwise cosine.
```
→
```tex
To screen uncorrected states, check top-PC share, not mean pairwise cosine.
```
Δ +1.

**D-4 (a). Paragraph 2: lead with the open question, and fix "the layout collapses"** (models collapse, not layouts).
```tex
In decoder language models, gated feed-forward blocks are reported to amplify outlier activations \citep{fishman2025fp8,yang2025activationspikes}, but the original T5 also carries large outliers \citep{zhao2025outlier}; why the T5 v1.1 layout collapses remains open.
```
→
```tex
Why models with the T5 v1.1 layout collapse remains open.
Gated feed-forward blocks are reported to amplify outlier activations in decoder language models \citep{fishman2025fp8,yang2025activationspikes}, but the original T5 also carries large outliers \citep{zhao2025outlier}.
```
Δ +2.

**D-5 (a). The Mikkelsen sentence needs a reread** because of the dangling "recovering…, and recovering…":
```tex
Concurrently, \citet{mikkelsen2026clinical} reports on clinical text a mid-depth retrieval trough in retrieval-trained encoders, recovering in the last layers, and recovering only partly in encoders without a retrieval objective.
```
→
```tex
Concurrently, \citet{mikkelsen2026clinical} reports a mid-depth retrieval trough on clinical text; retrieval-trained encoders recover in the last layers, and encoders without a retrieval objective recover only partly.
```
Δ −2. Before applying, check against the paper that both encoder types show the trough. The current sentence implies that too. The single-author "reports" is correct.

**D-6 (a). Paragraph 3: scope "the choice of encoder" to what was tested.**
```tex
For the editor of a text collection, the projection makes the choice of encoder matter little
```
→
```tex
For the editor of a text collection, the projection makes the choice among these six encoders matter little
```
Δ +2. Claim 7 is about six encoders, and a reader could take "the choice of encoder" to mean any encoder.

### Running word total for the (a) items in the main text

Counted by script on the replacement pairs above, excluding citation keys:

Abstract A-2 −1. Intro I-1 0, I-5 0, I-6 0, I-7 −6, I-9 0, I-10 −8. Related W-1 +3, W-2 −3, W-3 0, W-4 0, W-5 +2. Sec. 3 T-1 0, T-2 −1, T-3 +1, T-4 +1, T-5 −1, T-6 +5, T-7 0. Sec. 4 Q-2 −2, Q-3 −1. Sec. 5 M-1 −1, M-3 +3. Sec. 6 L-1 −4, L-2 −7. Sec. 7 R-2 +1, R-3 0, R-4 +2, R-6 −6, R-7 −1. Discussion D-1 −3, D-2 −8, D-3 +1, D-4 +2, D-5 −2, D-6 +2.

**Net −32 words (roughly 3 lines) across the main text.** This frees room for the (b) items you want most: I-3 (+6), A-2's layout gloss (+4), I-11 (+5) or R-8 (+3). Line breaks are local, so after applying, recompile and check that page 8 still ends where it should. Most of the savings fall on pages 1, 2 and 7 to 8. Three sections grow slightly: Sec. 3 (+5, mostly T-6), Related Work (+2) and Sec. 5 (+2).

---

## 2. Patterns across the main text (AI tells)

**X-1. Four identical section openers.** Sec. 4, 5, 6 and 7 each begin "This section asks…". The spine asked for question-and-answer openers, but the same three words four times reads as a template. Sec. 4 and 7 are yours (5 and 6 are James's openers, so tell him). Suggestions: Sec. 4 "We first ask whether…"; Sec. 7 as in R-1. Or open Sec. 4 directly with its answer sentence and drop the question sentence (Δ −20).

**X-2. ", so" endings.** Thirteen main-text sentences hinge on ", so", sometimes two in a row (Sec. 4 ¶1; Discussion ¶1). Q-3 and D-1 remove three. When you reread, turn one in three into a semicolon or a separate sentence.

**X-3. Colon as connector.** About 15 claim-colon-evidence sentences in three pages. Each one is defensible; the density is the tell. The items above remove six (I-6 indirectly, L-1, M-3, D-1, D-2, D-4). Those that introduce a list or a definition are fine (Sec. 3 "Routing is open-set:"; the layout definition in I-7).

**X-4. Synonym cycling for the method.** ABTT / the projection / the correction / the repair / the restoration / "the zero-shot projection" / "the post-hoc correction". Spine rule: the paper keeps the name ABTT. Proposed rule: "the projection" in the abstract, intro and bold headings; "ABTT" in body sentences; "without correction" only as the name of the baseline condition. R-6 and the T-8 rule cover the worst spot. Also watch "same-source / positive / same-directory / equivalent" pairs. Fig. 6 (density) uses "Equivalent pairs" (see AP-4).

**X-5. "X, not Y" aphorisms.** "masked, not missing" (title, abstract, intro, Sec. 4 opener), "carriers, not a cause" (Sec. 6 James, Discussion), "a screen, not a verdict", "not a cutoff validated…". The title phrase earns a second use in the intro. Three uses beyond the title, plus two others in the Discussion, is where it starts to read as a tic. D-2 and optionally Q-1 thin this out.

**X-6. Summary sentences closing paragraphs.** Yours: "In these models, then, …" (Sec. 5), "…the mask occupies a few directions and the signal lies outside them" (Sec. 6), "Our claims thus concern…" (Sec. 7). M-1, L-2 and R-8 handle these. James's paragraphs do the same ("…therefore do not produce…", "This extends…", "Its mask is thus…"), which is worth mentioning to him.

**X-7. Uniform sentence length.** Intro paragraphs 3, 4 and 6 are runs of 14 to 25-word declaratives. The short sentences already there ("The test split selects nothing." "On routing it does." in the appendix) are the most human-sounding lines in the paper. The edits above add some variation (I-5, M-3). There is no need to force more.

**Checks that pass.** No em-dashes, and no " -- " used as a dash (every `--` is a numeric range). No "Latin department". "family" is used only for model families, never for directories. No "notably / crucially / importantly / delve / showcase / underscore". No "not just X but Y". No puffery. The Ethics statement discloses AI assistance plainly.

**Consistency of claims and numbers** across abstract, intro, body and Discussion. Everything was checked against the source: 26 of 36; 0.50–0.66; 0.799; 0.962–0.987; 0.76; 0.826/0.883; 0.489/0.816; 0.736 and about 30× fewer tokens; 0.914; 0.489/0.977; 39.3 → 3.3; 0.987/89.9; 1.8/2.1 points; 38.5%. All agree. The only mismatches are:
- the model count (thirteen in Sec. 1, Sec. 5 and the App. C title; twelve as listed in Sec. 3), fixed by T-6;
- range style ("0.962--0.987" in the abstract, "0.962 to 0.987" elsewhere), cosmetic;
- "%" in the main text against "percent" in several appendix paragraphs ("80 percent", "38.5 percent"); pick one.

---

## 3. Limitations and Ethical Considerations (light pass)

**LIM-1 (a). (ii) joins two unrelated points with a semicolon.** Split it: "…so the pairs do not say which of them matters. That LaTa and LaBERTa share authors and a pretraining corpus rests on their model cards \citep{…}."

**LIM-2 (a). (iii) runs to 52 words.** Split after "matters;": "(iii)~The T5 v1.1 changes (gated-GELU feed-forward, untied embeddings) vary together, so we do not say which of them matters. T5-efficient-base has the original layout without T5-base's supervised pretraining, but it is a single checkpoint pretrained on about 30 times fewer tokens; it separates layout from pretraining mix only at that budget."

**LIM-3 (a). Labels paragraph, sentences 2 and 3** (a 35-word chain, then a ", so … , and" sentence):
```tex
Its director or supervised graduate students assigned the keys, and verification is by collation in the project's parallel display, not by independent double annotation, so no inter-annotator agreement figure exists for this testbed.
Collation compares the witnesses of one key, so it has nothing to compare when a key has a single witness, and 545 of the 840 directories hold one witness in this testbed.
```
→
```tex
Its director or supervised graduate students assigned the keys, and verification is by collation in the project's parallel display, not by independent double annotation.
No inter-annotator agreement figure therefore exists for this testbed.
Collation needs at least two witnesses of a key, and 545 of the 840 directories hold only one.
```
Limitations does not count toward the 8 pages, but shorter is still better.

**ETH-1 (b).** "The project has confirmed in writing that the license covers…; the project's interface credits…" is 45 words. Split at the semicolon. Ethics is otherwise ready.

---

## 4. Appendix, your parts (light pass: clarity and AI tells)

**AP-1 (a). App. A "Labels": dangling modifier.**
```tex
Counting a directory as key-named when its label begins with a four-letter code, a period, a year and a period, whatever follows, 689 of the 840 directory labels are source keys.
```
→
```tex
A label counts as key-named when it begins with a four-letter code, a period, a year and a period, whatever follows; by this rule 689 of the 840 directory labels are source keys.
```

**AP-2 (a). App. A "Derivation": the last sentence stacks four clauses, and "100 percent … with no character-level disagreement" says the same thing twice.**
```tex
The rule is scripted; checked against a current export of one manuscript (Paris, BnF, lat.\ 2123), it reproduces all 279 units present in both the export and the corpus, 100 percent after whitespace normalization with no character-level disagreement, and places every one of them in the same directory.
```
→
```tex
The rule is scripted. On a current export of one manuscript (Paris, BnF, lat.\ 2123), it reproduces all 279 units present in both the export and the corpus character for character after whitespace normalization, and places every one in the same directory.
```

**AP-3 (a). App. B "SIF": "ABTT-only names ABTT when we contrast the two"** → "SIF+ABTT applies both; ABTT-only marks ABTT alone where we contrast the two."

**AP-4 (a). Fig. 6 (density) caption introduces a fourth term for positive pairs.** "Equivalent pairs share a directory (Same in the legend); non-equivalent pairs do not (Different)." → "Same: pairs that share a directory; Different: pairs that do not."

**AP-5 (a). App. C "Raw and standardized gap": a 40-word sentence.** Split after "computes AUROC.": "…close to the $d'$ from which a binormal model computes AUROC. It tracks AUROC by construction, so we do not count it as a predictor."

**AP-6 (a). App. D "Sentence-T5": "therefore" points the wrong way.** Sentence-T5 shows the share is not sufficient. Necessity comes from the collapsed layers.
```tex
A high top-PC share is therefore necessary for collapse in our data but not sufficient: every one of the 36 collapsed layers of the pretrained checkpoints in Table~\ref{tab:all_models} (26 in the panel, 10 in T5-v1.1-base) has a share of at least 0.76, and high-share layers also occur without collapse.
```
→
```tex
Sentence-T5 thus shows that a high top-PC share is not sufficient for collapse. It is necessary in our data: every one of the 36 collapsed layers of the pretrained checkpoints in Table~\ref{tab:all_models} (26 in the panel, 10 in T5-v1.1-base) has a share of at least 0.76.
```

**AP-7 (a). App. D "English text": a 55-word sentence with a colon connector and a long appositive.**
```tex
Each non-empty Latin passage receives one English passage from a distinct opinion (the two empty Latin files receive an empty one): a run of whole words from a random sentence start in the body text, whose mT5 token count is within $\max(2, 3\%)$ tokens of the Latin passage's, and the English passage inherits its partner's split.
```
→
```tex
Each non-empty Latin passage receives one English passage from a distinct opinion, and the two empty Latin files receive an empty one.
The English passage is a run of whole words from a random sentence start in the body text, with an mT5 token count within $\max(2, 3\%)$ tokens of the Latin passage's, and it inherits its partner's split.
```

**AP-8 (a). App. F "Threshold grid": a tense clash** ("Refitting would trade … and changes no conclusion"). → "Refitting would trade one set of differences of one to a few files for another; it changes no conclusion of Section~\ref{sec:routing}."

**AP-9 (a). App. H "Reference Systems": one paragraph of about 500 words.** Break it at "All three are trained with symmetric InfoNCE…" and at "The dev slice holds out…" into three paragraphs (why these three models / recipe / checkpoint selection and per-model notes). Also rewrite the sentence with the colon connector and the vague "holds it":
```tex
The third answers the remaining question, whether supervision can clear the best cell the correction produces anywhere in the paper: KaLM-mini holds it, at 91.7 assignment accuracy and at 89.4 DirAcc@1 tied with Qwen3-0.6B.
```
→
```tex
The third, KaLM-mini, has the best ABTT cell in the paper (91.7 assignment accuracy; 89.4 DirAcc@1, tied with Qwen3-0.6B), so it tests whether supervision can clear that cell.
```

**AP-10 (a). App. K "Whitening": "It also degenerates the routing threshold."** "degenerate" is not normally transitive. → "It also breaks the routing threshold."

**AP-11 (b).** Your appendices use "ABTT-only", "SIF+ABTT", "the correction" and "the repair" in different places. The appendix can afford the precise names. Just keep "repair" out of sentences that also say "ABTT".

**Fine as written:** App. B Split and Software; App. C Geometry by model, Correlations, Cosine spread, SIF, Score distributions; App. D The projection on the four controls and Fine-tuned LaTa; App. E Number of directions and Removed and retained subspaces (its frank "close to a tautology" is good, and the main text should agree with it: L-2); App. G; App. I; App. J; App. L. App. H's frank lines ("A 71-file pool is the weak link in this protocol"; "the 499 pairs are memorized and more epochs buy nothing") read as written by a person. Keep them.

---

## 5. For James (outside your sections, noted in passing)

- The Sec. 5 and 6 openers also use "This section asks" (X-1).
- Sec. 5's T5-base paragraph and the Sec. 6 paragraphs end on "therefore / thus / This extends" summary sentences (X-6).
- Sec. 5's bold headings never state the layout claim. A headings-only reader meets "T5 v1.1-layout" for the first time in the third heading. The Sec. 5 opener states it, so this is minor.
