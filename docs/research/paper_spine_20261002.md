# Paper spine (2 October 2026)

All the reframe experiments have landed (upstream `d2f7625`). The main text was about 20 pages and must be 8 by 12 October. This document is the claim chain, the section plan and the page budget the paper was rebuilt around.

**Status.** The rewrite in this pull request implements the spine: the main text is now about 8.4 pages with three figures and three tables, and token attribution is out of the paper. Sections 1 to 4 below describe the paper as it now stands, except for the points listed in Section 9. Section 6 records the decisions James took; tell us where you disagree. Sections 5 and 8 were written before the rewrite and are kept for the record.

Terms used below:
- **Projection** = ABTT fit on the training split (the paper keeps the name ABTT).
- **Ranking** = Task A, pairwise same-source AUROC. **Routing** = Task B, sending a witness to an existing source or to "new" (DirAcc@1).
- **Panel** = the six models with full results (LaTa, PhilTa, mT5-base, LaBSE, Qwen3-0.6B, KaLM-mini), 100 layers in all. **Controls** = LaBERTa, PhilBERTa, T5-base and T5-v1.1-base in the main text; Sentence-T5 and SPhilBERTa in the appendix only. Twelve models in all.
- **Collapsed layer** = baseline AUROC below 0.70. **Top-PC share** = share of variance on the first principal component of centred embeddings.
- **T5.1.1 layout** = gated-GELU feed-forward and untied embeddings, as in T5-v1.1, mT5, LaTa and PhilTa. The original T5-base has ReLU and tied embeddings.
- Bracketed ids such as [G1] point to claim blocks in a longer working file of all 141 candidate claims with their evidence, which James holds and can share.

## 1. The story in six sentences

1. People read intermediate layers of off-the-shelf encoders, check anisotropy with mean cosine, and correct it afterwards. We test that practice on medieval Latin, which offers labels from a scholarly edition and T5 and encoder-only models pretrained on the same data.
2. LaTa, PhilTa and mT5-base retrieve near chance through most of their depth, although their first and last layers work.
3. The signal is masked, not missing: a projection fit on training vectors alone restores ranking at every layer of the panel.
4. The usual warning sign is silent. Mean pairwise cosine does not flag the collapse; top-PC share is high at every collapsed layer.
5. The collapse tracks the T5.1.1 layout: encoder-only siblings and the original T5-base do not collapse, the low-rank geometry appears on English text too, and fine-tuning leaves the middle layers collapsed.
6. What masks the signal is about three directions that rank near chance. Zeroing the largest coordinates does not remove them. In LaTa three token types feed them; in mT5-base no set of up to 100 token types does.

The paper closes on consequences and limits: after the projection the six encoders route about equally well, and no frozen encoder beats a character n-gram reference.

**Contributions for the introduction (four):**
1. A per-layer retrieval profile of T5 encoder states, the first we know of, showing a mid-depth collapse that a train-only projection reverses (Sec. 4).
2. Controls showing which models collapse: in our panel the collapse tracks the T5.1.1 layout, not the unlabelled pretraining corpus, the input language or the lack of an embedding objective (Sec. 5).
3. A localization of the masking subspace: about three directions, not removed by zeroing coordinates, fed by three token types in LaTa and by no set of up to 100 token types in mT5-base (Sec. 6).
4. A measured account of what the projection buys downstream and where it stops, with a lexical reference stated plainly, and the released testbed (Sec. 7).

## 2. Curated claims, ranked by importance to the story

Novelty verdicts come from a recheck on 2 October against these sharper claims.

| # | Claim | Evidence | Novelty | Wording limits |
|---|---|---|---|---|
| 1 | **Mid-depth collapse.** In LaTa, PhilTa and mT5-base, 26 of 36 layers retrieve at AUROC 0.50–0.66. The next-lowest layer in the panel is 0.799. A cutoff on training AUROC picks the same 26 layers. | G1–G3 | Partly novel: no per-layer retrieval profile of T5 encoder states exists. | "The first per-layer retrieval profile of T5 encoder states in isolation that we know of." Never "T5 encoders collapse". |
| 2 | **Masked, not missing.** A train-only projection that removes the top D ≤ 10 principal components brings all 100 panel layers to AUROC 0.962–0.987. Reduced-rank whitening does the same on ranking. | R-1, W-2 | Partly novel: the idea is Timkey & van Schijndel's; layers at chance restored into one narrow band is new. | Ranking only (routing after the projection still varies by layer). Panel only. The projection is not a contribution. |
| 3 | **The collapse tracks the model's layout.** (a) LaTa and PhilTa collapse; their encoder-only siblings LaBERTa and PhilBERTa never fall below 0.826 and 0.883. (b) T5-v1.1-base collapses on the same Latin text (lowest 0.489); the original T5-base does not (lowest 0.816). All four collapsing models have the T5.1.1 layout. (c) The same low-rank middle layers appear when mT5-base, PhilTa and T5-v1.1-base read English legal text (geometry only). (d) In LaTa, contrastive fine-tuning lifts the last layer (0.938 to 0.984) and leaves layers 2–11 at 0.50–0.57. | P-1, P-5, P-6, GEN-1–3, FT-1, FT-2 | (a) Novel as a controlled comparison. (b) No prior report found; the most distinctive result. (c) A control. (d) A replication (Hämmerl 2023). | "In our panel the collapse tracks the T5.1.1 layout; we do not isolate which change is responsible." T5-base also saw supervised tasks in pretraining; this goes in Limitations (the run that would separate the two is deferred, see D2). |
| 4 | **Zeroing coordinates fails; removing three directions works.** The dominant direction sits mostly on about ten coordinates, yet zeroing the ten largest restores 0 of 26 collapsed layers (best 0.79): the rest of the direction still masks retrieval. Centering does nothing. Removing one component recovers a median 45% of the gain; removing three brings all 26 layers to 0.91 or above. At collapsed layers the removed directions rank near chance (median 0.489) and the retained ones at 0.977. | E1-1, E1-5, E1-12, E1-13, H1-1, H1-3, E3-2 | Partly novel. Hämmerl 2023 is a precedent for a projection beating zeroing; Parupudi 2026 reports a similar one-component share. The mT5-base case has no precedent we found. | An extension of the outlier-dimension work, not a contradiction of it. "Three suffice for 0.91; mT5-base keeps improving to ten." Not "no ranking signal" in general: at healthy layers the removed directions carry some. |
| 5 | **The usual diagnostic is silent.** Mean pairwise cosine is unreliable in either direction: embedding-trained models sit at 0.59–0.97 and retrieve well; the collapsed Latin T5 layers sit at 0.22–0.58; T5-v1.1-base collapses at 0.87–0.96. Top-PC share is at least 0.76 at every collapsed layer. | G4, G5, G6 | Partly novel: the concept is known (IsoScore, Timkey, Parupudi); the clean split by layer is new. | High top-PC share is necessary, not sufficient: one healthy panel layer (mT5-base layer 4) and Sentence-T5 also have it. It marks the regime and does not order layers inside it. One threshold everywhere. |
| 6 | **In LaTa, three token types feed the removed directions.** Pooling without the comma, the period and `</s>` (11% of tokens) restores all 10 collapsed layers (median 0.919); an equal-mass random drop stays at 0.500. In mT5-base no set of up to 100 token types restores a layer. The first component is not passage length. | E2-1, E2-8, E2-9 | Partly novel. Delimiter tokens carrying outliers is known (Bondarenko 2023; Fuster Baggetto & Fresno 2022). New: the mass-matched control, recovery from chance, the mT5-base negative. | A LaTa result plus an mT5-base negative; PhilTa in one sentence (it needs 30 types, a third of the tokens). The tokens are carriers, not a cause: LaBERTa pools the same three and does not collapse. Say that carriers were ranked on the top three training components. |
| 7 | **After the projection, encoders route about equally well.** At train-selected layers the DirAcc@1 spread across the six encoders falls from 39.3 to 3.3 points, and every routing gain excludes zero. For the embedding-trained models the gain appears with no threshold, and removing the mean alone does not reproduce it. | R-4, CI-1, PQ-1, PQ-2, PQ-4 | Partly novel; nearest precedent is out-of-distribution detection (Kamoi & Kobayashi 2020; Podolskiy 2021). | No cause is named. Not "offset", not "hubness". |
| 8 | **Limits.** No frozen encoder, at its train-selected layer, beats the character n-gram reference (0.987 AUROC, 89.9 DirAcc@1): the best repaired cells tie it and three are significantly below. Fine-tuned Qwen3 and KaLM-mini exceed it in point estimate. Fine-tuning plus the projection ties the zero-shot projection on LaTa and adds about 2 routing points on Qwen3 and KaLM-mini. | LEX-1, CI-5, FTC-2, FTC-3 | Known (Loci Similes 2026 headlines the same). A disclosure. | State in Sec. 3 that the task is lexically solvable by design, so 0.987 is the reference the projection approaches. One sentence in Limitations. |

**Where the novelty sits.** Claims 1 and 3 carry the paper. Two results have no prior report that we could find: the T5-base against T5-v1.1-base contrast (3b), and mT5-base resisting both coordinate zeroing and token dropping while a three-direction projection repairs it (4, 6). The rest are known ideas confirmed or bounded in a new model family. Nothing competing has appeared since 20 September. Findings remains the realistic target.

## 3. What stays out of the main text

Rule: a result may leave the main text if the claim as worded stays true with that result in view. Where a result contradicts a claim, we narrowed the claim instead. Everything listed stays in the appendix or the released files unless marked "drop".

| Left out of the main text | How the claim stays true |
|---|---|
| Sentence-T5 (top-PC share 0.66–0.94 at layers 2–11, with AUROC 0.80–0.89) | Claim 5 says "necessary, not sufficient" and names it in one clause. Full row in the appendix table of all twelve models. |
| PhilTa's token audit in detail; the failed token-mix rule | Claim 6 is a LaTa result with one honest PhilTa sentence. |
| mT5-base needing five to ten directions | Claim 4 says so in a clause. |
| Per-coordinate standardization (median 79% of the gain; 9 of 26 layers reach 0.90) | Claim 4 is about zeroing, and says a projection works where zeroing fails. Appendix. |
| Zeroing 50–300 coordinates (variance ranking) does repair LaTa and PhilTa | Claim 4 says "the ten largest". One clause, with the random-coordinate control. |
| The r-ratio prediction (holds at 18 of 26 layers; all 8 misses in PhilTa) | Drop. No claim uses it. |
| Within-regime correlations; the standardized gap; LaTa's negative raw gap | Appendix figures and table only. The main text does not discuss the raw gap. |
| "Two forms of anisotropy"; "the offset explains the routing gain"; hubness as a cause | Drop as claims. The first is known; the other two failed their checks. The hubness observation itself is an optional add-on (below). |
| SIF as a probe of the mechanism | SIF becomes an appendix baseline row. |
| "The projection hurts fine-tuned models *because* the top components carry task signal" | Drop the mechanism. The direction of the effect is an optional add-on (below). |
| Whitening routes 2–15 points below the projection | Claim 2 is scoped to ranking. Appendix. |
| Token attribution; cluster-geometry silhouettes | Attribution is removed (D3). Silhouettes on 2-D t-SNE: drop the numbers. |

**Optional add-ons** (one sentence each, if space allows; the first two in Sec. 7, the third in Sec. 6):
- *Hubness.* "The projection also lowers hubness." The paper must not say this causes the routing gain.
- *The projection after fine-tuning.* "After fine-tuning, the projection no longer helps ranking: AUROC falls slightly in all three fine-tuned models (by 0.002 to 0.014), in line with Rajaee & Pilehvar (2021)." Stated as a direction in point estimate, with no interval in the main text and no mechanism.
- *SIF.* Frequency-weighted pooling (SIF) down-weights the same frequent tokens, and its partial repair agrees with the token ablation.

## 4. Section plan and page budget

Model paper: Timkey & van Schijndel (EMNLP 2021), the closest accepted paper in shape. Two changes from the current draft: the projection result moves into Section 4, because the localization tests use the removed components; and the model controls come before the localization, so the reader who has just seen three T5 encoders collapse learns at once which models do and do not.

**Heading convention** (Timkey for sections, Ethayarajh 2019 for result paragraphs):
- Sections are Title Case noun phrases: no questions, no experiment ids (E1, H1, P2x2 leave the main text).
- No subsections in the main text.
- Setup paragraphs get a noun-phrase label ("Corpus and labels.").
- Each result paragraph is headed by its finding as a sentence, scoped to the models it holds for. Reading only the bold sentences gives the argument.

| § | Section and run-in paragraph headings | Pages | Floats |
|---|---|---|---|
| 1 | **Introduction.** The practice and the setting; the surprise; masked, not missing; which models and what masks it; the four contributions; one scope sentence (we localize and repair; we do not explain why the T5.1.1 layout produces the collapse). | 1.0 | |
| 2 | **Related Work.** Three grouped paragraphs: anisotropy statistics; outlier dimensions and the tokens that carry them; layerwise analysis and post-hoc correction. | 0.6 | |
| 3 | **Testbed and Protocol.** Corpus and labels. / Split and train-only rule. / Tasks and metrics. / Models. / Post-processing and lexical reference. | 0.8 | |
| 4 | **Collapse by Depth.** LaTa, PhilTa and mT5-base fall to near chance at mid-depth. / A train-only projection restores ranking at every layer of the six panel models. / Mean pairwise cosine gives no reliable warning. / Top-PC share is high at every collapsed layer. | 1.5 | F1 (full width): AUROC by layer, baseline and projected. F2: top-PC share and mean cosine against AUROC. |
| 5 | **Collapse Across Models.** Encoder-only siblings trained on the same data do not collapse. / T5-v1.1-base collapses; the original T5-base does not. / In three T5.1.1-layout models, the low-rank middle layers appear on English text too. / In LaTa, contrastive fine-tuning leaves the middle layers collapsed. | 0.9 | T1: every model in the paper, with lowest AUROC and peak top-PC share. |
| 6 | **Localizing the Collapse.** Zeroing the ten largest coordinates restores no collapsed layer. / Removing three directions restores every collapsed layer. / In LaTa, three token types feed the removed directions. / In mT5-base, no set of up to 100 token types does. | 1.6 | F3, two panels: coordinates zeroed (with random control) and components removed. T2: each test, its prediction and its outcome. |
| 7 | **Routing and Limits.** After the projection, routing accuracy converges across the six encoders. / For the embedding-trained encoders, the routing gain needs no threshold. / No frozen encoder at its train-selected layer beats a character n-gram reference. / Fine-tuning Qwen3 and KaLM-mini adds about two routing points over the projection. | 0.8 | T3: headline ranking and routing with intervals; baseline, projected, fine-tuned and n-gram rows. |
| 8 | **Discussion** (no run-in headings). Which statistic to check before reading a layer; why the T5.1.1 layout collapses is open; what the result means for the editor. | 0.5 | |
| | Slack for float rounding | 0.3 | |
| | **Total** | **8.0** | 3 figures, 3 tables |

Limitations and Ethics do not count toward the 8 pages. Limitations lists the uncontrolled factors as (i)–(iv): one labelled corpus; architecture, objective and tokenizer vary together in the sibling pairs; the T5.1.1 changes vary together; mT5-base has no token account.

Three habits from the model papers: open each analysis section with the question and its answer before any setup; state each test in Section 6 as a prediction that could fail, and show the failed ones plainly in T2; keep the abstract to about eight sentences and three numbers.

## 5. Proposed split of the writing (superseded: the rewrite is done; the split now suggests who reviews what)

Each paragraph goes to whoever ran its experiment. Ian keeps the paper's frame and the larger share.

| Part | Writer |
|---|---|
| Abstract; Sec. 1 Introduction; Sec. 2 Related Work | Ian (James supplies the outlier-dimension and token-carrier citations) |
| Sec. 3 Testbed and Protocol | Ian |
| Sec. 4: collapse and projection paragraphs; F1 | Ian |
| Sec. 4: the two diagnostic paragraphs; F2 | James |
| Sec. 5: sibling and T5-base paragraphs; T1 | James |
| Sec. 5: English and fine-tuning paragraphs | Ian |
| Sec. 6: coordinate and token paragraphs; F3 left panel; T2 | James |
| Sec. 6: the three-directions paragraph; F3 right panel | Ian |
| Sec. 7 Routing and Limits; T3 | Ian |
| Sec. 8 Discussion; Limitations; Ethics; appendix | Ian (James supplies appendix text for his three experiments) |
| Section openers for Sec. 5 and 6; review pass over the whole paper | James |

By page budget this is about 5.4 pages for Ian and 2.3 for James.

## 6. Decisions (James, 2 October)

- **D1. T5-base and T5-v1.1-base come into the main text as a within-family contrast, with all twelve models in one appendix table: yes.** Without it, the title and first contribution say "T5 encoders" while the best-known T5 checkpoint does not collapse, which a reviewer can check in minutes. With it, the same fact becomes the paper's most distinctive result.
- **D2. Two small extra runs: deferred, only if time allows.** (i) `google/t5-efficient-base` through `p2x2_panel.py`: it has the original layout but was pretrained on C4 only, so it would separate "layout" from "supervised pretraining" in claim 3(b). (ii) The projection on the four control models from their cached vectors. Until they run, claim 2 stays scoped to the six-model panel and claim 3(b) carries its limitation.
- **D3. Token attribution (appendices K–M, about 10 pages) is removed: yes.** Nothing in the main line depends on it, the results are mixed, and keeping it means disclosing a failed pre-registered threshold for something that is no longer a contribution. It could stand alone as a later workshop paper. LaTa layer 7 as the worked example needs a new reason (each model's lowest-AUROC layer).
- **D4. Title: "Masked, Not Missing: Mid-Depth Retrieval Collapse in T5-Family Encoders" (for now; your call).** "T5.1.1 Encoders" names a cause we have not isolated while D2(i) is deferred, and few readers know the version label. The abstract states the scope instead: four of the five T5-family encoders collapse, and all four share the T5 v1.1 layout.
- **D5. Cite Mikkelsen 2026 once, as a concurrent contrast, in Discussion: yes.** It reports a mid-depth collapse with final-layer recovery in retrieval-trained BERT encoders on clinical text.
- **D6. Section order and heading convention of Section 4: yes.**

## 7. Corrections found while auditing the claims

Checking every number against the result files turned up 6 sentences in the old draft that were wrong as written and 19 that were overstated. All are fixed in the rewrite. The ones a reviewer would have noticed first: "mean cosine is lower at collapsed layers" (not true of T5-v1.1-base); "one false alarm in 100"; two different top-PC thresholds (0.6 and 0.76); "two to five directions"; "the repair matches the n-gram reference".

- That LaTa and LaBERTa share authors and a source corpus rests on their Hugging Face model cards; the ACL 2023 paper does not mention either model. The paper now cites the cards and says "pretrained on the same corpora".
- Seventeen citations were added, each checked against its primary page.

## 8. Suggested calendar (written before the rewrite)

| Date | Step |
|---|---|
| 3 Oct | Agree the spine |
| 4–5 Oct | Attribution removal; twelve-model appendix table (the two runs in D2 only if time allows) |
| 4–7 Oct | Rewrite to the section plan; build F1–F3 and T1–T3 |
| 8–10 Oct | Review cycle, page fit, citation check |
| 11 Oct | Freeze. 12 Oct: submit |

## 9. Where the rewritten paper departs from this spine

| What | Spine | Paper now | Why |
|---|---|---|---|
| Scope phrase for the layout claim | "In our panel the collapse tracks the T5.1.1 layout" | "Across the ten models we test, the collapse tracks the T5 v1.1 layout" | "Panel" means the six models; the evidence comes from the controls. The paper says "T5 v1.1 layout" throughout. |
| Section 5, first heading | "…siblings trained on the same data do not collapse." | "…pretrained on the same corpora…" | What the model cards support. |
| Section 7, first heading | "After the projection, routing accuracy converges across the six encoders." | "At train-selected layers, routing after the projection converges across the six encoders." | Routing after the projection still varies by layer (75.6 to 90.2). |
| Section 7, second heading | "…the routing gain needs no threshold." | "…the routing gain does not depend on the threshold." | Clearer. |
| Fine-tuning paragraph | "adds about two routing points"; "LaTa ties at 87.7" | The same claim, stated as means over five reseedings, with the single-split cells of Table 3 named (+2.0, +3.1, LaTa −0.9) | Table 3 showed different numbers from the text. |
| Optional add-ons | hubness, the fine-tuned drop, SIF with the token ablation | Only the fine-tuned drop is in the main text | Sections 6 and 7 were over budget. Hubness numbers stay in the appendix. |
| Worked-example layer | LaTa layer 7 with a new reason | No worked-example layer in the main text; the appendix quotes LaTa layer 6 | Layer 6 is LaTa's lowest-AUROC layer and the layer the existing tables report. |
| Table 2 | each test, its prediction and its outcome | Columns Test / Expected / Outcome / Result; daggers mark what was written down before the run | Not every row was a prediction made in advance. |
| Table 3 | with intervals | Intervals for the six panel models; the fine-tuned block points to the appendix interval table | Space. |
| Top-PC threshold | 0.76 | 0.76 everywhere, scoped to pretrained checkpoints | Fine-tuned LaTa layer 2 is collapsed at 0.759. |
| Discussion | "why the layout collapses is open" | Adds one sentence naming the nearest precedents (gated feed-forward blocks amplify outliers in decoder models; the original T5 also carries large outliers) | A reviewer would ask for the precedent. No mechanism is claimed. |

**Needs Ian's machine** (inputs are not on James's):
1. `fig_gen_geometry.pdf` still draws its guide line at 0.6; the generator already draws 0.76. Re-render with `python scripts/paper/reframe/gen_ft_geometry.py`, then change the caption's "0.6" to "0.76, the lowest share of any collapsed layer of the pretrained models".
2. The `tab:gen_geometry` caption says "every collapsed Latin layer" reaches 0.76; add "of the pretrained models" in its generator and re-render.
3. Five tables were re-rendered from rows parsed out of the committed files, with bodies byte-identical and only names and captions changed: `finetune_ceiling`, `lexical_baselines`, `selected_layers`, `appendix_lasttok_comparison`, `ft_lata_layerwise`. A plain run of their generators should reproduce them.
4. The paper mirror holds one Overleaf edit that upstream lacks (a comma in a sentence of the old introduction that no longer exists), so `sync_paper_repo.sh push` needs a pull first or `SYNC_FORCE=1`.
