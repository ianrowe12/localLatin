# Do the highlights sit on Latin prefixes? (issue #211, part 2)

Issue #211, part of epic #93. From Prof. Firey's email of 2026-09-15: she and the new
evaluator see the highlighting disregard the distinctive words a human uses to confirm a
match, and suspect that frequent Latin prefixes (prae, sub, pro, per, ad, ab) create false
matches, at least in the highlighting. The models tokenise words into pieces, and the webapp
renders one highlight per piece.

This note answers two questions with numbers:

1. Does the highlight rest on frequent prefix pieces disproportionately?
2. Does aggregating pieces to whole words restore the words a reader uses?

The evidence is 89,896 scored pair sides of the deployed webapp artifacts (89,264 from the
unlabelled bulk artifacts, 632 from the gallery pairs) and 4,800 scored pair sides of the
paper's 200-positive-pair run. The per-pair CSVs hold a few rows more than that, 89,942 and
4,803, because each carries one `_corpus` row per model and one row per side skipped for a
non-finite attribution vector.

**Short answer. No to the first, yes to the second.** Prefix pieces are 6 to 8 percent of the
pieces in a text and receive 0.3 to 8.2 percent of the positive attribution mass, so they are at
or below their share in almost every deployed configuration. Taking the highest prefix lift in
any deployed cell, on either side of the pair, it is 1.13 over the bulk artifacts (Qwen3-0.6B,
`raw`, IG, candidate side) and 1.16 over the 20 gallery pairs per model (Qwen3-0.6B, `raw`, IG,
candidate side); on the query side that the tables below report, the same maxima are 1.09 and
1.14. Under the deployed `sif_abtt` variant the bulk query-side lift falls to 0.15 to 0.82
(0.09 in one 20-pair gallery cell, PhilTa). High-frequency pieces are likewise under-weighted,
sharply so under `sif_abtt`. What is true, and what most likely produced the impression, is that
**the highlighted unit is usually not a word**: 35 to 93 percent of the top five highlighted
pieces are word fragments, and only 2 to 56 percent of them are a whole word a reader would
quote. Summing each word's pieces fixes
that: 65 to 97 percent of the top five *words* are distinctive words. The fix belongs in the
display, which is exactly what part 1 of this issue proposes.

Reproduce everything below with one CPU-only job:

```
sbatch slurm/ig/prefix_attribution_analysis.sbatch
```

Summary CSVs (tracked, small): `docs/research/data/prefix_attribution_deployed.csv` and
`docs/research/data/prefix_attribution_pos200.csv`. The per-pair rows land under
`runs/active/ig_examples/prefix_attribution/` of whichever checkout the job ran in, and are
gitignored. Analysis code:
`scripts/ig/prefix_attribution.py` (the classifier and the aggregator) and
`scripts/ig/run_prefix_attribution_analysis.py` (the driver), with
`tests/test_prefix_attribution.py` covering both on synthetic pieces and a synthetic NPZ.

## Method

**Prefix pieces.** A piece counts as a prefix piece when its surface text, with the
tokenizer's word-boundary marker stripped and case and punctuation normalised, equals one of
`prae, pre, sub, pro, per, ad, ab, con, com, de, ex, in, re, dis, trans`. That is the set named
in the issue plus the rest of the productive Latin prefixes. Most of these are also standalone
prepositions, so every table separates two flavours:

* a **prefix whole word**, for example `_in` standing alone as the preposition;
* a **prefix fragment**, any prefix piece that does not span its whole word, for example
  `_prae` opening `praedestinatione`, which is the thing the issue actually describes.

Two properties of that second flag should be read with the numbers. It is **positional only in
the weak sense**: it fires wherever a prefix string appears inside a longer word, not only at
the start, so `con` in `diaconus`, `re` in `facere` and `in` in `hominum` all count. Measured
over the whole labelled corpus, the word-internal or word-final share of the flagged prefix
fragments is LaTa 9.0 percent, PhilTa 8.4 percent, mT5-base 20.8 percent, LaBSE 26.2 percent,
Qwen3-0.6B and KaLM-mini 23.3 percent each. Restricting the flag to word-opening pieces would
therefore shrink it by a fifth to a quarter for the four heavily fragmenting tokenizers, and by
under a tenth for the two Latin T5s. It is also a **lower bound**: a prefix fused into a larger
piece, such as Qwen's `Ġpra` + `ed` for `prae`, is not counted at all, because no single piece
equals a prefix string.

**High-frequency pieces.** Piece frequencies come from tokenising all 1,705 files of
`data/canon_labelled/` with each model's own tokenizer (a null-safe `os.walk`, because CCL
directory names contain newlines). The "top 1 percent" set is the top 1 percent of the piece
*types* observed in that corpus: 125 types for LaTa, 105 for PhilTa, 85 for mT5-base, 94 for
LaBSE, 58 for the shared Qwen and KaLM vocabulary. Those few types cover 31 to 46 percent of
all piece tokens, which is why the frequency baseline matters.

**The baseline that makes "disproportionate" mean something.** For every pair side the tables
report both the share of pieces that carry a flag and the share of attribution mass those
pieces receive. Each is averaged over the sides in the cell, and the **lift** is the ratio of
those two cell means, not the mean of the per-side ratios. Lift 1.0 means the pieces get
exactly their share; above 1.0 means disproportionate.

**Attribution mass.** The positive part of the per-token vector, normalised over the side. A
negative IG score argues against the match, so folding its magnitude into the denominator would
make the shares uninterpretable. MaRC masks are already in [0, 1], so the clip is a no-op there.

**Top five highlights.** Ranked by `|attribution|`, which is what
`web/services/token_map_svc.py` ranks the auto-highlights by, so the top-5 columns describe the
units a reviewer actually sees outlined.

**Word aggregation.** Pieces are grouped into words by the tokenizer's own convention
(SentencePiece `▁`, WordPiece `##`, byte-level BPE `Ġ`), and a word's attribution is the sum of
its pieces. A bare `▁` and a special token both close the current word, otherwise
`consonante` + `r` + `▁` + `capitulum` would read back as one word. Byte-level BPE pieces are
decoded out of the GPT-2 byte alphabet first, so `QuÃ¦` reads back as `Quæ`.

**Distinctive word.** A word whose normalised form is neither one of the prefix strings nor
among the top 1 percent most frequent word types of `data/canon_labelled/` (16,028 word types,
so the 160 commonest). This is a proxy for "a word a reader would quote to confirm the match",
not a lexicographic judgement.

**What was read.**

* *Deployed run*, `runs/active/ig_examples/`: for each model, every artifact at the layer that
  `scripts/resubmit/deployed_unlabelled_layers.json` currently serves for `raw` and for
  `sif_abtt`, which is 2,237 to 4,981 artifacts per model per variant, plus the 20 gallery
  pairs per model. Bulk artifacts carry IG only; the gallery pairs also carry the MaRC masks.
* *Paper run*, `runs/active/ig_examples_200pos_v1/`: all 200 positive pairs for LaTa, PhilTa and
  mT5-base, IG and MaRC, `raw` (called `baseline` in the artifacts) and `abtt`.

Query and candidate sides are reported separately; the tables below give the query side, which
is the text a reviewer is triaging. The candidate side tracks it closely on the mass shares
(the largest query-to-candidate gap in any deployed cell is 0.019) but the lift, being a ratio
of two small numbers, can diverge more: the largest lift gap is 0.24, on Qwen3-0.6B `raw` IG
over the 20 gallery pairs. Full candidate-side rows are in
`docs/research/data/prefix_attribution_deployed.csv`.

## Result 1: prefix pieces get at most their share of the mass

Deployed run, unlabelled bulk artifacts, IG, query side. `n` is artifacts read.

| Model | Layer | Variant | n | pieces that are prefixes | IG mass on prefixes | lift | pieces that are prefix fragments | IG mass on prefix fragments | lift |
|---|---|---|---|---|---|---|---|---|---|
| LaTa | 1 | raw | 4981 | 0.079 | 0.072 | 0.91 | 0.020 | 0.017 | 0.87 |
| LaTa | 1 | sif_abtt | 4981 | 0.079 | 0.016 | 0.21 | 0.020 | 0.008 | 0.38 |
| PhilTa | 1 | raw | 4903 | 0.074 | 0.008 | 0.11 | 0.020 | 0.003 | 0.14 |
| PhilTa | 1 | sif_abtt | 4903 | 0.074 | 0.011 | 0.15 | 0.020 | 0.006 | 0.30 |
| mT5-base | 6 | raw | 2237 | 0.073 | 0.079 | 1.09 | 0.033 | 0.041 | 1.24 |
| mT5-base | 1 | sif_abtt | 3420 | 0.072 | 0.041 | 0.57 | 0.032 | 0.020 | 0.63 |
| LaBSE | 11 | raw | 4177 | 0.070 | 0.064 | 0.92 | 0.028 | 0.025 | 0.90 |
| LaBSE | 11 | sif_abtt | 4177 | 0.070 | 0.033 | 0.47 | 0.028 | 0.015 | 0.54 |
| Qwen3-0.6B | 28 | raw | 2968 | 0.064 | 0.061 | 0.96 | 0.029 | 0.028 | 0.98 |
| Qwen3-0.6B | 7 | sif_abtt | 2237 | 0.063 | 0.048 | 0.76 | 0.029 | 0.027 | 0.96 |
| KaLM-mini | 22 | raw | 2237 | 0.063 | 0.068 | 1.07 | 0.029 | 0.043 | 1.51 |
| KaLM-mini | 1 | sif_abtt | 3411 | 0.063 | 0.052 | 0.82 | 0.028 | 0.027 | 0.93 |

The deployed variant is `sif_abtt`, the right column block of every model's second row. Under it
prefix pieces receive 1.1 to 5.2 percent of the mass while occupying 6.3 to 7.9 percent of the
pieces: they are down-weighted by a factor of 1.2 to 6.5. The uncorrected `raw` view is close to
neutral for four models, and the two cases that exceed their share on this side (mT5-base 1.09
overall and 1.24 on fragments, KaLM-mini 1.07 and 1.51 on fragments) are the two models whose
tokenizers fragment Latin hardest. Even there the absolute mass is under 8 percent. On the
candidate side a third model joins them, Qwen3-0.6B at 1.13 overall, though its fragment lift
stays at 0.99.

## Result 2: high-frequency pieces are under-weighted, and SIF is why

Same rows, frequency flags.

| Model | Variant | pieces in the top 1 pct | IG mass on them | lift |
|---|---|---|---|---|
| LaTa | raw | 0.456 | 0.390 | 0.86 |
| LaTa | sif_abtt | 0.456 | 0.114 | 0.25 |
| PhilTa | raw | 0.462 | 0.350 | 0.76 |
| PhilTa | sif_abtt | 0.462 | 0.099 | 0.22 |
| mT5-base | raw | 0.385 | 0.412 | 1.07 |
| mT5-base | sif_abtt | 0.385 | 0.133 | 0.34 |
| LaBSE | raw | 0.313 | 0.267 | 0.86 |
| LaBSE | sif_abtt | 0.313 | 0.083 | 0.27 |
| Qwen3-0.6B | raw | 0.307 | 0.177 | 0.58 |
| Qwen3-0.6B | sif_abtt | 0.308 | 0.058 | 0.19 |
| KaLM-mini | raw | 0.308 | 0.197 | 0.64 |
| KaLM-mini | sif_abtt | 0.307 | 0.071 | 0.23 |

**SIF down-weights frequent pieces before pooling, so the match and the highlight can differ.**
The `sif_abtt` embedding is a SIF-weighted pool, and a piece's SIF weight falls as its corpus
probability rises. Every deployed configuration therefore shows the frequent third to half of
the tokens carrying roughly 6 to 13 percent of the attribution: lift 0.19 to 0.34, against 0.58 to
1.07 for `raw`. This is worth stating plainly to Prof. Firey, because it cuts both ways. The
model that decides the match is deliberately almost deaf to `et`, `in`, `qui` and the common
inflectional endings, and the highlight inherits that. A reviewer comparing the highlight
against their own reading is comparing against a scorer with a different ear, not against a
malfunction.

## Result 3: the highlighted unit is usually not a word

This is where the observation is right. Deployed run, bulk artifacts, IG, query side, top five
highlights ranked by `|IG|`.

| Model | Variant | top5 that are prefixes | top5 that are prefix fragments | top5 that are word fragments | top5 that are a whole distinctive word | top5 WORDS that are prefixes | top5 WORDS that are distinctive | distinct words in the 5 slots |
|---|---|---|---|---|---|---|---|---|
| LaTa | raw | 0.013 | 0.003 | 0.351 | 0.476 | 0.003 | 0.897 | 4.97 |
| LaTa | sif_abtt | 0.003 | 0.003 | 0.413 | 0.564 | 0.000 | 0.973 | 4.94 |
| PhilTa | raw | 0.014 | 0.007 | 0.686 | 0.216 | 0.005 | 0.880 | 4.78 |
| PhilTa | sif_abtt | 0.004 | 0.004 | 0.591 | 0.377 | 0.000 | 0.961 | 4.93 |
| mT5-base | raw | 0.065 | 0.034 | 0.799 | 0.040 | 0.027 | 0.725 | 4.45 |
| mT5-base | sif_abtt | 0.016 | 0.011 | 0.873 | 0.080 | 0.003 | 0.898 | 4.63 |
| LaBSE | raw | 0.027 | 0.011 | 0.599 | 0.183 | 0.006 | 0.728 | 4.55 |
| LaBSE | sif_abtt | 0.013 | 0.007 | 0.630 | 0.285 | 0.005 | 0.884 | 4.65 |
| Qwen3-0.6B | raw | 0.068 | 0.022 | 0.840 | 0.036 | 0.042 | 0.721 | 3.86 |
| Qwen3-0.6B | sif_abtt | 0.040 | 0.020 | 0.877 | 0.051 | 0.016 | 0.849 | 4.44 |
| KaLM-mini | raw | 0.043 | 0.037 | 0.928 | 0.017 | 0.006 | 0.653 | 4.25 |
| KaLM-mini | sif_abtt | 0.045 | 0.024 | 0.858 | 0.066 | 0.012 | 0.874 | 4.76 |

Read the two middle columns against the two on the right. Under the deployed `sif_abtt` variant,
41 to 88 percent of the highlighted pieces are fragments of a longer word, and only 5 to 56
percent of them are a whole word that is not a corpus commonplace. After summing each word's
pieces, 85 to 97 percent of the top five words are distinctive words. Still under `sif_abtt`, at
most 4.5 percent of the highlighted pieces are prefixes of any kind and at most 2.4 percent are
the part-of-a-longer-word kind the issue describes; under `raw` those ceilings are 6.8 and 3.7
percent, both on the query side. So the prefixes are a symptom of fragmentation rather than its
cause: they are simply the most recognisable fragments when a reader sees one.

Fragmentation is a property of the tokenizer, not of the attribution. The share of all pieces
that are word fragments runs LaTa 0.35, PhilTa 0.48, LaBSE 0.72, mT5-base 0.79, Qwen3-0.6B 0.84,
KaLM-mini 0.84. A per-piece display cannot show words when, for the decoder models, five pieces
in six are not words.

One concrete pair, LaBSE gallery example 1, `Can.apost.7`, `sif_abtt`:

```
query   VII Episcopus aut presbiter aut diaconus nequaquam saeculares curas adsumat Sin aliter deiciatur
  top 5 pieces   VII | cura | ##scop | Epi | ##quam
  top 5 words    VII | Episcopus | curas | nequaquam | saeculares

candidate  VII Ut sacerdotes et ministri altaris secularibus curis abstineant Episcopus aut presbiter ...
  top 5 pieces   VII | sacerdotes | cura | ##scop | alit
  top 5 words    VII | sacerdotes | Episcopus | aliter | curas
```

Three of the five query slots go to pieces of two words (`Epi` + `##scop`, `cura`), so the
reviewer sees broken stems where the aggregated view names `Episcopus`, `curas`, `nequaquam`
and `saeculares`. The attribution is not wrong; the rendering is.

## Result 4: word-level mass confirms it

Same rows, after summing pieces within words.

| Model | Variant | words that are prefixes | word mass on prefix words | word mass on distinctive words | piece mass on whole words | piece mass on fragments |
|---|---|---|---|---|---|---|
| LaTa | raw | 0.074 | 0.055 | 0.683 | 0.674 | 0.326 |
| LaTa | sif_abtt | 0.074 | 0.009 | 0.879 | 0.637 | 0.363 |
| PhilTa | raw | 0.074 | 0.005 | 0.751 | 0.479 | 0.521 |
| PhilTa | sif_abtt | 0.074 | 0.006 | 0.881 | 0.469 | 0.531 |
| mT5-base | raw | 0.075 | 0.047 | 0.668 | 0.207 | 0.793 |
| mT5-base | sif_abtt | 0.074 | 0.021 | 0.797 | 0.166 | 0.834 |
| LaBSE | raw | 0.075 | 0.039 | 0.680 | 0.336 | 0.664 |
| LaBSE | sif_abtt | 0.075 | 0.019 | 0.797 | 0.317 | 0.683 |
| Qwen3-0.6B | raw | 0.075 | 0.036 | 0.716 | 0.148 | 0.852 |
| Qwen3-0.6B | sif_abtt | 0.074 | 0.022 | 0.794 | 0.131 | 0.869 |
| KaLM-mini | raw | 0.074 | 0.029 | 0.648 | 0.104 | 0.896 |
| KaLM-mini | sif_abtt | 0.074 | 0.027 | 0.780 | 0.147 | 0.853 |

Prefix words (the standalone prepositions `in`, `de`, `ad`, `ex`, `per`, `ab` and the rest) are
7.4 to 7.5 percent of the words and take 0.5 to 5.5 percent of the word-level mass. Under
`sif_abtt` between 78 and 88 percent of the word-level mass sits on distinctive words. Aggregation
does not invent that; it recovers mass that the piece view had scattered across stems.

## Result 5: MaRC, and the paper's operational layers

The 200-positive-pair run (issue #141, benchmark v1) carries both views and both `raw` and
`abtt` for the three models it covers. Query side, 200 pairs per cell.

| Model | Variant | View | pieces prefix | mass prefix | lift | lift on prefix fragments | pieces top 1 pct | mass top 1 pct | lift | top5 prefix | top5 fragment | top5 whole distinctive | top5 WORDS distinctive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LaTa | raw | ig | 0.077 | 0.054 | 0.70 | 0.88 | 0.476 | 0.400 | 0.84 | 0.010 | 0.310 | 0.377 | 0.690 |
| LaTa | raw | marc | 0.077 | 0.076 | 0.99 | 0.98 | 0.476 | 0.481 | 1.01 | 0.061 | 0.254 | 0.329 | 0.731 |
| LaTa | abtt | ig | 0.077 | 0.060 | 0.79 | 0.67 | 0.476 | 0.384 | 0.81 | 0.010 | 0.364 | 0.474 | 0.886 |
| LaTa | abtt | marc | 0.077 | 0.076 | 0.99 | 0.90 | 0.476 | 0.470 | 0.99 | 0.056 | 0.287 | 0.390 | 0.920 |
| PhilTa | raw | ig | 0.072 | 0.008 | 0.12 | 0.14 | 0.480 | 0.352 | 0.73 | 0.012 | 0.694 | 0.213 | 0.895 |
| PhilTa | raw | marc | 0.072 | 0.069 | 0.96 | 0.98 | 0.480 | 0.464 | 0.97 | 0.027 | 0.499 | 0.283 | 0.927 |
| PhilTa | abtt | ig | 0.072 | 0.036 | 0.49 | 0.54 | 0.480 | 0.342 | 0.71 | 0.003 | 0.601 | 0.326 | 0.931 |
| PhilTa | abtt | marc | 0.072 | 0.065 | 0.90 | 0.93 | 0.480 | 0.435 | 0.91 | 0.008 | 0.494 | 0.376 | 0.944 |
| mT5-base | raw | ig | 0.071 | 0.065 | 0.91 | 1.00 | 0.442 | 0.430 | 0.97 | 0.027 | 0.803 | 0.026 | 0.747 |
| mT5-base | raw | marc | 0.071 | 0.100 | **1.40** | 1.38 | 0.442 | 0.555 | 1.26 | **0.154** | 0.660 | 0.015 | 0.823 |
| mT5-base | abtt | ig | 0.071 | 0.074 | 1.04 | 1.08 | 0.442 | 0.445 | 1.01 | 0.067 | 0.769 | 0.036 | 0.823 |
| mT5-base | abtt | marc | 0.071 | 0.077 | 1.08 | 1.06 | 0.442 | 0.463 | 1.05 | 0.104 | 0.732 | 0.058 | 0.886 |

Within this run, MaRC is the view where the observation lands hardest, and only on the most
heavily fragmenting tokenizer: mT5-base under `raw` puts 10 percent of its mask mass on prefix
pieces against a 7.1 percent baseline, and 15.4 percent of its top five slots are prefix pieces.
MaRC optimises a soft mask rather than integrating a gradient, and the mask drifts towards
common short pieces when nothing is removed from the embedding first. ABTT pulls it back to
1.08. Every IG cell in this run stays at or under 1.04, though note that the deployed run's own
`raw` IG does exceed 1.0 on prefix fragments for two models (Result 1), so MaRC is not the only
place the effect appears. The deployed gallery pairs agree: LaTa MaRC lift 0.99,
PhilTa 1.14, LaBSE 0.59 (see `prefix_attribution_deployed.csv`, `stratum=gallery`).

Two data gaps worth recording. The deployed Qwen3-0.6B gallery artifacts carry **MaRC masks
riddled with NaN**: all 20 pairs, both sides, and both of the variants that carry a mask at all
(`baseline` and `abtt`; there is no `sif` or `sif_abtt` mask for any model). No vector is
entirely NaN, but 87 to 97 percent of the positions are, a mean of 92 percent, so the mask is
unusable and the analysis records those 40 sides as `n_nonfinite` in the deployed CSV rather
than averaging NaN. The mask optimisation evidently diverged for that model when those
artifacts were regenerated. Separately, the KaLM-mini and mT5-base deployed artifacts carry no
MaRC keys at all. Neither gap affects the webapp, which renders IG by default, but a MaRC panel
for Qwen would currently be blank and the divergence should be fixed separately (issue #216).

## Answers for Prof. Firey

**"The highlighting seems to fall on frequent prefixes."** Measured over 89,896 scored pair
sides, it does not. Prefix pieces hold 6 to 8 percent of the text and receive 0.3 to 8.2 percent
of the highlight weight, and under the variant the webapp serves they receive well under their
share on every model. Frequent pieces are down-weighted harder still, because the scoring method
deliberately discounts them.

The exceptions are all in the **uncorrected** views, and there are three worth naming. Two of
the six models put more than their share on word-internal prefix pieces under `raw` IG, which is
precisely the kind the issue describes: KaLM-mini at 1.51 times its share (4.3 percent of the
mass) and mT5-base at 1.24 times (4.1 percent). The third is MaRC on mT5-base under `raw`, at
1.40 times its share overall with 15.4 percent of the top five slots. The deployed `sif_abtt`
variant brings all three under 1.0 (KaLM-mini to 0.93, mT5-base to 0.63), and no IG cell under
the deployed variant exceeds 0.96 on fragments. So the intuition has a real home, in the
uncorrected view of the two tokenizers that cut Latin into the smallest pieces, and it does not
describe what the webapp serves.

**"The highlighting disregards the distinctive words."** This is right, and the cause is the
rendering rather than the model. The model reads a word as several pieces, and the display
outlines the pieces. Four to nine of every ten highlighted units are a fragment such as `Epi`,
`##scop` or `cura`, and a reader looking for `Episcopus` or `curas` sees the stem cut in half.
When the pieces of a word are added together and the word is outlined instead, 85 to 97 percent
of the five highlights become whole distinctive words, and the prefixes essentially vanish from
the highlight (0.0 to 1.6 percent of slots).

**Why the match and the highlight can differ.** The deployed scorer pools tokens with SIF
weights, which shrink as a token gets more common in the corpus, and then removes the leading
principal components. The embedding that decides the match is therefore built mostly from rare
words, and the highlight reflects that same weighting. A reviewer who confirms a match on a
shared commonplace phrase and then sees no highlight there is seeing the method working as
designed, not a bug.

## Recommendation for the display

1. **Aggregate to words and highlight words** (part 1 of this issue). Sum the piece attributions
   inside each word, using the word boundaries of the original text, and outline the word. Keep
   an expand control for the pieces. The numbers above are the case for this: it is the single
   change that moves the top five highlights from 5 to 56 percent whole distinctive words to 85
   to 97 percent, and it costs nothing at the artifact level because it is a display step.
   `scripts/ig/prefix_attribution.py` has the boundary logic, including the two traps
   (the bare SentencePiece `▁`, and the byte-level BPE alphabet) that a naive implementation
   will hit.
2. **Do not mask or dim by frequency.** A frequency threshold was the fallback the issue
   proposed if prefixes carried disproportionate mass. They do not, and the deployed variant
   already suppresses frequent pieces by a factor of three to five. Adding a display-side
   frequency filter on top would hide evidence twice and would misrepresent the scorer. If a
   control is wanted at all, make it an optional off-by-default toggle labelled as a reading
   aid, not a default.
3. **Fix the auto-highlight variant.** `token_map_svc.py` chooses the IG vector for the K = 5
   auto-highlights by a fixed preference order, `abtt` first, and ignores the variant the
   reviewer selected. On the artifacts that carry both, the `abtt` and `sif_abtt` top five
   agree on only 44 to 78 percent of slots (mT5-base 0.44, KaLM-mini 0.56, LaBSE 0.72, Qwen3-0.6B
   0.75, PhilTa 0.76, LaTa 0.78; 400 artifacts per model, 20 for Qwen3-0.6B, which has only the
   gallery pairs carrying both). A reviewer on the `sif_abtt` view is
   being shown ABTT's highlights. This is a small change and belongs with part 1.
4. **State the SIF caveat in the reviewer-facing help text.** One sentence: the system weights
   rare wording more heavily than common wording when it decides a match, so the highlight
   favours unusual words over shared formulae.

## Caveats

* Piece and word frequencies are fitted on `data/canon_labelled/`, while most queries in the
  deployed run come from `data/canon_unlabelled/`. The two are the same register and the same
  transcription pipeline, but the frequency flags are therefore an out-of-sample judgement on
  the query side.
* "Distinctive" is a frequency proxy, not a philological one. A word can be rare in this corpus
  and still useless for confirming a match (a scribal error, a damaged reading).
* Mass shares use the positive part of the attribution. A separate question, not asked here, is
  whether the pieces that argue *against* a match behave differently.
* The deployed bulk artifacts carry IG only, so the MaRC evidence rests on 20 gallery pairs per
  model plus the 200-pair run, and on three models rather than six.
