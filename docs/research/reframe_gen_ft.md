# Reframe GEN and FT: label-free generality across model and text, fine-tuned LaTa row

Issue #234, share of the analysis reframe (#229). Handoff rows GEN and FT in
`docs/research/reframe_handoff_20260927.md`. Run 2026-09-28 on the CPU partition.

## Results in brief

- **GEN: every cell shows the mid-depth low-rank profile.** mT5-base, PhilTa and
  T5-v1.1-base on a length-matched English sample, and T5-v1.1-base on the Latin corpus,
  reach a top-PC share of 0.82 or more with effective rank 1.0 to 3.1 at every layer from 4
  to 11, and all recover at the first and last layer. Neither fail branch in the handoff
  occurs: mT5-base does not stay high-rank on English (so the collapse does not depend on the
  input text, within the two texts tested), and T5-v1.1-base collapses on English as much as
  mT5-base does (so it does not depend on the pretraining data, within the three pretraining
  mixes tested).
- **T5-v1.1-base on Latin** (the P2x2 raw partner of Sentence-T5): peak top-PC share 0.975
  (layer 9), at least 0.958 at every layer from 2 to 11, effective rank 1.14 to 1.20.
- **Latin reproduction**: the new CPU pipeline reproduces the paper's mT5-base and PhilTa
  geometry to the six significant digits stored (vectors: cosine 1.00000000 to the cached ones,
  max relative L2 difference 3.2e-4).
- **FT: the handoff numbers are right.** Fine-tuned LaTa, layers 2 to 11: baseline AUROC
  0.499 to 0.569 (printed 0.50 to 0.57); last layer 0.938 to 0.984. New: its peak top-PC
  share is **0.945** (layer 4), against 0.952 (layer 4) before fine-tuning.
- Compute: 3.6 CPU core-hours (pilot 0.4, main job 3.2); no GPU.

## Provenance

| Item | Value |
|---|---|
| Scripts | `scripts/paper/reframe/gen_english_sample.py`, `gen_extract.py`, `gen_ft_geometry.py` |
| Job | `slurm/reframe/gen_ft_geometry.sbatch`, job 22535075 (cpu, 16 cores, 48 GB, 12:04 elapsed of 50:00 reserved); pilot job 22534999 (16 cores, 1:28) |
| Outputs (gitignored) | `runs/active/reframe/gen/`: `english_sample.{csv,json}`, `english_sample_lengths.csv`, `bases/<model_slug>/{latin,english}/hidden_layer{1..12}_embeddings.npy` + `meta.csv` + `config.json`, `gen_geometry.csv`, `gen_repro.csv`, `ft_layerwise.csv`, `gen_ft_facts.md` |
| Committed artefacts | `overleaf_drafts/tables/gen_geometry.tex` (`tab:gen_geometry`), `overleaf_drafts/tables/ft_lata_layerwise.tex` (`tab:ft_lata_layerwise`), `overleaf_drafts/figures/fig_gen_geometry.{pdf,png}` |
| Models (HF revision) | `google/mt5-base` 2eb15465c5dd, `bowphs/PhilTa` 8572ff520a1a, `google/t5-v1_1-base` b5fc947a416e (all Apache-2.0); fp32, torch 2.10, transformers 4.57.6 |
| Split | `runs/active/resubmit/data/phase_resubmit_split.csv` (847 train / 858 test) |
| Tests | `tests/test_reframe_gen_ft.py` (sampler logic on a toy counter, renderer determinism, byte-identical regeneration of the committed tables from `runs/`, skipped when absent) |

Rerun, from the checkout that holds the code (models and CAP shards must be in the HF cache;
compute nodes run offline):

```bash
CODE_ROOT=$PWD REPO_ROOT=/u/irowerojas/localLatin sbatch --export=ALL slurm/reframe/gen_ft_geometry.sbatch
# tables, figure and facts only (reads the CSVs, seconds):
python scripts/paper/reframe/gen_ft_geometry.py --stage render
```

`--stage render` regenerates both tables and the figure byte for byte (checked, PDF included).

### Representation

Identical to the paper's cached T5 vectors (`src/extract_hidden_cli.py`, `--pooling mean
--token_filter tokenizer_empty --max_length 512`): encoder of `AutoModelForSeq2SeqLM`,
`hidden_states[1..12]`, mean over attended tokens minus the tokenizer's empty tokens, EOS
kept. `gen_extract.py` imports `pool_hidden` and `build_token_keep_lookup` from `src/`, so
the pooling cannot drift. The only differences are operational: one forward pass per batch
for all layers, batches formed after sorting by length, and CPU instead of GPU.

### Geometry definitions and the fitting question

The paper's statistics come from `scripts/resubmit/run_layer_geometry_diagnostics.py` and are
read by `geometry_vs_retrieval.py`; `gen_ft_geometry.py` imports the same `pca_stats` and
`cosine_stats` functions unchanged.

- Top-PC share: first eigenvalue share of the centered (not L2-normalized) covariance.
- Effective rank: exp of the entropy of the eigenvalue shares.

Neither statistic has a fit/apply step: both are read directly on a set of rows, and the
paper reads them on the 847 training passages. We followed that exactly rather than fitting
on one half and reporting on the whole. For English, the primary subset is the 847 English
passages whose Latin partner is a training passage, so n and the length profile are identical
to the Latin cells. Every cell is also computed on all 1,705 rows (in `gen_ft_facts.md`): no
conclusion changes, and no peak moves by more than 0.001.

## English sample

- **Source**: US court opinions from the Caselaw Access Project (Harvard Law School Library
  Innovation Lab), in the public-domain raw redistribution of Common Pile v0.1
  (`common-pile/caselaw_access_project`, revision `3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90`),
  seven of its 173 shards: `cap_00000`, `00030`, `00060`, `00090`, `00120`, `00150`, `00170`.
- **Licence**: the CAP texts are public domain (CC0 at the source; Common Pile includes only
  public-domain CAP documents). Nothing is redistributed by this repo: the sample is rebuilt
  from the pinned revision.
- **Why this corpus**: legal prose is the nearest openly licensed English register to canon
  law. It is on the HF hub and ungated. (The TeraflopAI/free-law CAP mirrors are gated, and
  Pile of Law is CC BY-NC-SA.) The seven shards span federal and state reporters from the 1810s
  (Martin's Louisiana reports, Cowen's New York reports) to the 2010s. The sample draws on 45
  reporters, led by F.2d (316 passages), Ga. App. (208) and Ill. App. (103).
- **Selection** (`gen_english_sample.py`, seed 42): 1,500 random documents per shard; a
  document's prose is its body paragraphs (lines at column 0 with at least 12 words, of which at
  least 80 percent contain a letter; this drops the indented caption, docket, date and counsel
  lines and citation strings); 8,140 documents have prose. Latin passages are visited in a
  seeded order, and each takes the first unused document that can hold it. A sentence start is
  drawn at random, and whole words are added until the mT5 token count (EOS included) is as close
  to the Latin passage's as possible. A match is accepted within max(2, 3 percent) tokens. One
  document gives at most one passage (1,703 distinct opinions). The two empty Latin files get an
  empty English partner. Targets above 1,024 tokens (4 passages) are capped there, since the
  encoder truncates at 512.
- **Size**: 1,705 passages, one per labelled Latin passage, each inheriting its partner's
  split. Text sha256 (passages joined by newlines):
  `db09a4cc56528d59008072b1d011a55815e3cd9b5db4510aca50ba4d5b5c38b3`.
- **Length match** (mT5 tokenizer, the matching tokenizer): 1,086 of 1,705 exact, all within
  3 tokens. The distributions coincide:

| Tokenizer | Text | median | IQR | 95th pct | > 512 |
|---|---|---|---|---|---|
| mT5 | Latin | 98 | 62-155 | 334 | 16 |
| mT5 | English | 98 | 62-155 | 333 | 16 |
| PhilTa | Latin | 76 | 47-119 | 247 | 9 |
| PhilTa | English | 107 | 67-174 | 362 | 23 |
| T5-v1.1 | Latin | 138 | 87-218 | 469 | 66 |
| T5-v1.1 | English | 89 | 56-140 | 293 | 14 |

  The match is exact only under the mT5 tokenizer, the one multilingual model in the cross.
  Under PhilTa's tokenizer the English passages are about 40 percent longer. Under T5-v1.1's
  English-only tokenizer, Latin fragments into more pieces and the Latin passages are about
  55 percent longer. All cells collapse whatever the direction of the mismatch, so length does
  not drive the result. It still bounds how exactly Latin and English cells can be compared
  number by number.

## Latin reproduction check

mT5-base and PhilTa were re-extracted on the Latin corpus with the new pipeline and compared
with the paper's `runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv` (train, raw
view) and with the cached vectors in `runs/active/resubmit_bases/` (loaded through
`AlignmentResolver`; 17 rows are permuted there after the benchmark v1 label corrections, and
were realigned by filename).

- Top-PC share: identical to the 6 significant digits stored at all 24 model-layers.
- Effective rank: max absolute difference 1e-4, max relative 1.7e-6 (rounding).
- Vectors: min cosine 1.00000000; max relative L2 difference 3.2e-4. The max absolute
  difference is 0.28, at mT5-base layers 5 to 11, where the pooled vectors carry massive
  coordinates. That is float32 CPU versus GPU arithmetic on large values.

The published geometry therefore reproduces, and the English and T5-v1.1 cells are comparable
with the paper's numbers.

## GEN results

Train subset (847 passages per cell). Full per-layer values, including mean pairwise cosine,
are in `runs/active/reframe/gen/gen_ft_facts.md`.

| Model | Text | Peak top-PC share (layer) | Layers with share >= 0.6 | Share there | Eff. rank there | Share at layers 1 / 12 |
|---|---|---|---|---|---|---|
| mT5-base | Latin | 1.000 (5) | 4-11 | 0.893-1.000 | 1.00-2.15 | 0.229 / 0.217 |
| mT5-base | English | 1.000 (5) | 4-11 | 0.822-1.000 | 1.00-3.10 | 0.339 / 0.154 |
| PhilTa | Latin | 0.858 (6) | 3-11 | 0.764-0.858 | 1.81-4.34 | 0.060 / 0.077 |
| PhilTa | English | 0.894 (6) | 4-11 | 0.846-0.894 | 1.82-2.30 | 0.158 / 0.104 |
| T5-v1.1-base | Latin | 0.975 (9) | 2-11 | 0.958-0.975 | 1.14-1.20 | 0.403 / 0.235 |
| T5-v1.1-base | English | 0.950 (9) | 2-11 | 0.931-0.950 | 1.28-1.36 | 0.288 / 0.169 |

Per layer, top-PC share (Latin / English):

| Layer | mT5-base | PhilTa | T5-v1.1-base |
|---|---|---|---|
| 1 | 0.229 / 0.339 | 0.060 / 0.158 | 0.403 / 0.288 |
| 2 | 0.199 / 0.325 | 0.068 / 0.213 | 0.958 / 0.931 |
| 3 | 0.220 / 0.311 | 0.764 / 0.579 | 0.970 / 0.946 |
| 4 | 0.893 / 0.822 | 0.831 / 0.846 | 0.970 / 0.946 |
| 5-11 | 0.9997-1.000 / 0.999-1.000 | 0.829-0.858 / 0.868-0.894 | 0.970-0.975 / 0.942-0.950 |
| 12 | 0.217 / 0.154 | 0.077 / 0.104 | 0.235 / 0.169 |

## Reading

1. **Fail branch 1 (text) does not occur.** mT5-base's middle layers are rank one on English
   legal prose as on Latin: top-PC share at least 0.999 and effective rank 1.00 to 1.01 at
   layers 5 to 11, with the same onset (layer 4) and the same recovery at layer 12. PhilTa's
   plateau is slightly higher on English (0.85-0.89 against 0.83-0.86), with the same effective
   rank (1.8-2.3); its onset is one layer later on English (layer 3 is 0.58 against 0.76).
2. **Fail branch 2 (pretraining data) does not occur.** T5-v1.1-base was pretrained on English
   C4 only, mT5-base on 101-language mC4, and PhilTa on Greek, Latin and English. All three
   collapse on English. T5-v1.1-base collapses the earliest and most uniformly of the three
   (from layer 2), on its own pretraining language as well as on Latin.
3. **What this supports.** Within the models and texts tested, the low-rank middle layers are
   a property of the model, not of Latin input or of out-of-domain text. That is the reading
   the Section 6 paragraph states as the prediction. It also removes one alternative
   explanation for the raw T5 cells of P2x2: T5-v1.1-base's collapse on Latin is not a
   language-mismatch artefact, because it collapses the same way on English.
4. **What it does not support.**
   - It is geometry, not retrieval. On the Latin testbed a top-PC share of 0.6 or more marks
     all 26 collapsed layers with one false alarm. That false alarm is mT5-base layer 4
     (share 0.893, AUROC 0.80), and English mT5-base layer 4 sits at 0.822. So a high share on
     English is strong evidence of the same degenerate geometry, not proof that English
     retrieval would fail at every such layer.
   - One English register (court opinions) and one sample. The two texts differ in language,
     era and genre together, so "the input text" here means these two texts.
   - Three T5 encoders of one size (base). Nothing here separates T5 pretraining from the
     missing embedding objective: every model in the cross is raw. That remains P2x2's job.
5. **A side observation on mean cosine**, the usual anisotropy statistic. It moves a lot with
   the text while the variance statistics do not. On the collapsed layers of T5-v1.1-base it is
   0.87-0.96 on Latin and 0.34-0.44 on English. On mT5-base layers 5-11 it is 0.33-0.38 on Latin
   and 0.63-0.69 on English. This is consistent with the paper's Section 4 point that mean cosine
   measures the shared offset and gives no warning of collapse. It is an observation, not a
   test.

### Against the current text

- Section 6 predicts, if the collapse is a property of the model, "top-PC share above 0.8 and
  effective rank near 1 at mid-depth" in every cell. Top-PC share holds (>= 0.82 at layers 4 to
  11 in every cell). "Effective rank near 1" holds for mT5-base (1.00) and T5-v1.1-base
  (1.1-1.4), but PhilTa sits at 1.8-2.3 on both texts, as it already does on Latin (Section 4
  quotes collapsed effective rank 1.0-4.6). Suggested wording: "effective rank below 2.5".
- The expected result in `\pending{GEN}` ("a mid-depth low-rank profile in every cell") is
  what we find.
- The Limitations first sentence ("one labeled corpus in one language") stays true for
  retrieval. An optional addition is below.

## FT: fine-tuned LaTa layerwise row

Sources: `runs/active/resubmit/results/finetune/finetune_lata_layer_results.csv` (baseline and
ABTT-optimal AUROC per layer, fine-tuned), `runs/active/resubmit/results/phase_resubmit_results.csv`
(pre-trained), and the fine-tuned vectors in
`runs/active/resubmit_finetune_bases/phase9_bases/bowphs_LaTa-ft/hidden_mean_tokempty/`, where
top-PC share and effective rank were recomputed with the same code on the 847 training
passages. The pre-trained LaTa geometry recomputed the same way matches the paper (peak 0.952
at layer 4, `z1_numbers.md`).

| Layer | AUROC PT | AUROC FT | Top-PC share PT | Top-PC share FT | Eff. rank PT | Eff. rank FT |
|---|---|---|---|---|---|---|
| 1 | 0.934 | 0.936 | 0.050 | 0.050 | 134.33 | 134.44 |
| 2 | 0.563 | 0.569 | 0.771 | 0.759 | 4.61 | 4.92 |
| 3 | 0.513 | 0.512 | 0.935 | 0.929 | 1.59 | 1.64 |
| 4 | 0.500 | 0.499 | 0.952 | 0.945 | 1.38 | 1.43 |
| 5 | 0.496 | 0.500 | 0.943 | 0.931 | 1.41 | 1.48 |
| 6 | 0.496 | 0.500 | 0.934 | 0.921 | 1.44 | 1.51 |
| 7 | 0.498 | 0.501 | 0.937 | 0.923 | 1.43 | 1.50 |
| 8 | 0.502 | 0.505 | 0.950 | 0.940 | 1.36 | 1.42 |
| 9 | 0.503 | 0.507 | 0.943 | 0.931 | 1.40 | 1.48 |
| 10 | 0.505 | 0.510 | 0.942 | 0.930 | 1.43 | 1.50 |
| 11 | 0.508 | 0.512 | 0.939 | 0.927 | 1.47 | 1.55 |
| 12 | 0.938 | 0.984 | 0.081 | 0.069 | 132.22 | 137.68 |

Verification of the handoff:

- "layers 2-11 at 0.50-0.57": correct (0.499 at layer 4 to 0.569 at layer 2). The minimum
  over all 12 layers is also 0.499 (layer 4), so the cell reads the same as a strict minimum
  to two decimals.
- "last layer 0.938 -> 0.984": correct (0.9376 -> 0.9839). "0.997 is KaLM-mini's": not
  touched here.
- ABTT-optimal on the fine-tuned model spans 0.9625-0.9751 over layers 1-12 (D = 10 at every
  layer), as `finetune_ceiling.md` says.
- Geometry, new: fine-tuning leaves the collapsed geometry in place. Top-PC share at layers
  2-11 is 0.759-0.945 fine-tuned against 0.771-0.952 pre-trained, and effective rank at
  layers 3-11 is 1.42-1.64 against 1.36-1.59. The last layer, where the loss attaches, moves
  from 0.081 to 0.069.

### Values for James's `tab:panel_2x2`

| Cell | Models | AUROC_min | PC1_max |
|---|---|---|---|
| T5, raw+FT | LaTa (fine-tuned) | 0.50--0.57 (unchanged; verified) | **0.945** |
| T5, raw | T5-v1.1-base | P2x2 (retrieval not run here) | **0.975** (geometry from GEN, layer 9) |

The T5-v1.1-base AUROC_min cell needs Task A scoring, which is P2x2's. The Latin vectors are
already extracted in the paper's format at
`runs/active/reframe/gen/bases/google_t5-v1_1-base/latin/` (split order, `meta.csv` beside
them), so they can be scored on CPU with no GPU re-extraction, and a GPU re-extraction can be
checked against them.

## Replacement LaTeX

`acl_latex.tex` is filled in one integration pass; nothing here edits it.

**1. Section 6, `\pending{GEN: ...}` (line 589).** Replace the whole sentence
"We find \pending{GEN: ...}." with:

```latex
Every cell shows the same profile (Table~\ref{tab:gen_geometry}, Figure~\ref{fig:gen_geometry}).
The English sample is 1,705 passages of US court opinions from the Caselaw Access Project \citep{kandpal2025commonpile}, one per Latin passage and matched to it in mT5 token length; as on Latin, we read the statistics on the 847 passages paired with training passages.
On English, mT5-base reaches a top-PC share of at least 0.999 with effective rank 1.0 at layers 5 to 11, as on Latin, and PhilTa reaches 0.85 to 0.89 at layers 4 to 11 (0.83 to 0.86 on Latin), with effective rank 1.8 to 2.3 on both texts.
T5-v1.1-base, pretrained on English alone, collapses from layer 2 to layer 11 on both texts, at top-PC share 0.93 to 0.95 on English and 0.96 to 0.98 on Latin.
In every cell the first and last layers stay high-rank.
Neither fail condition occurs: in these cells the low-rank middle layers follow the model rather than the input text or the pretraining language.
```

and, if the prediction sentence stays, change "effective rank near 1" to "effective rank below
2.5" (PhilTa is 1.8 to 2.3 on both texts).

**2. `tab:panel_2x2` caption, `\pending{FT: import layerwise fine-tuned LaTa row}` (line 570).**
Delete the `\pending{...}` (the caption already defines the fine-tuned AUROC range). In the
table body (James's region): `\pendingnum{FT}` -> `0.945`, and the T5-v1.1-base PC1$_{\max}$
`\pendingnum{P2x2}` -> `0.975`. The `% PLACEHOLDER TABLE` comment can drop "and FT (fine-tuned
LaTa top-PC share)".

**3. Optional, the fine-tuning paragraph of Section 6**, after "Its layers 2 to 11 still collapse,
at AUROC 0.50 to 0.57.":

```latex
Their geometry barely moves: the peak top-PC share is 0.945 after fine-tuning against 0.952 before (Appendix~\ref{app:reference_systems}, Table~\ref{tab:ft_lata_layerwise}).
```

**4. Appendix floats** (suggested home: the reference-systems appendix for the FT table, a
short GEN paragraph in the layer-diagnostics appendix for the rest):

```latex
\input{tables/gen_geometry}
\input{tables/ft_lata_layerwise}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_gen_geometry.pdf}
\caption{Label-free layer geometry of three T5 encoders on the Latin corpus (solid, filled markers) and on a length-matched English sample of US court opinions (dashed, hollow markers), read on 847 passages per cell. (a)~Top-PC share, the share of centered variance on the first principal component; the dotted line is the 0.6 threshold that separates collapsed from healthy layers on the Latin testbed. (b)~Entropy effective rank, log scale. Every model is low-rank from layer 2 to 4 through layer 11 on both texts and recovers at layer 12.}
\label{fig:gen_geometry}
\end{figure*}
```

**5. Optional, Limitations**, after the first sentence:

```latex
A label-free check extends the geometry, but not the retrieval result, to one English register (Section~\ref{sec:objective}).
```

**Bibliography** (`custom.bib` is not edited on this branch; add with the integration pass):

```bibtex
@article{kandpal2025commonpile,
    title = {The Common Pile v0.1: An 8{TB} Dataset of Public Domain and Openly Licensed Text},
    author = {Kandpal, Nikhil and Lester, Brian and Raffel, Colin and Majstorovic, Sebastian and Biderman, Stella and Abbasi, Baber and Soldaini, Luca and Shippole, Enrico and Cooper, A. Feder and Skowron, Aviya and others},
    journal = {arXiv preprint arXiv:2506.05209},
    year = {2025}
}
@misc{cap2024,
    title = {Caselaw Access Project},
    author = {{The President and Fellows of Harvard University}},
    howpublished = {\url{https://case.law/}},
    year = {2024}
}
```

Cite `cap2024` alongside `kandpal2025commonpile` if the venue expects the original source;
the arXiv id is the one the dataset card links (2506.05209).
