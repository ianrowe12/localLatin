# Reframe D2: t5-efficient-base control and the projection on the four controls

The two deferred runs of `docs/research/paper_spine_20261002.md`, section 6, D2. Approved by
Ian on 2026-10-08; run the same day on the CPU partition, no GPU.

## Results in brief

- **(i) T5-efficient-base does not collapse.** It has T5-base's layout (ReLU feed-forward with
  inner width 3072, tied embeddings, 12 layers). On the Latin corpus its lowest AUROC at layers
  2-11 is **0.736** (layer 11) and its top-PC share peaks at **0.627** (layer 11). No layer meets
  either half of the spine's test (AUROC below 0.70, top-PC share at least 0.76). **An
  original-layout checkpoint pretrained on C4 alone, without supervised tasks or dropout, does
  not collapse. It saw about 30 times fewer pretraining tokens than T5-base and T5-v1.1-base
  (524,288 steps of 65,536 tokens, against about one million steps of 1,048,576 tokens), so it
  does not show that the supervised mix plays no part at T5-base's budget.** The layout account
  survives this control at this budget. T5-efficient-base is weaker than T5-base at every
  layer, by 0.016 to 0.080 AUROC, and its layer 11 has the low-rank shape of a mild dip
  (effective rank 8.1).
- **Comparability.** The same run re-extracted T5-base, T5-v1.1-base, LaBERTa and PhilBERTa on
  CPU and matched the committed P2x2 rows (GPU, #248) at every layer to at most 1.8e-7 AUROC,
  1.9e-7 top-PC share, 7.5e-7 relative effective rank and 1.1e-7 mean cosine. All P2x2
  reproduction gates (a-d) passed on the same run.
- **(ii) The projection does not bring every control layer into the panel's band.** With the
  panel's recipe (ABTT fit on training vectors, D chosen on training DirAcc@1 from 1, 2, 3,
  5, 7, 10), LaBERTa (0.963-0.977) and PhilBERTa (0.974-0.980) land inside 0.962-0.987 at
  every layer. T5-base misses at one layer (layer 11, 0.959). T5-v1.1-base reaches only
  0.927-0.970, with layers 6-12 below 0.962. Its collapsed layers 2-11 still go from
  0.489-0.536 to 0.927-0.969. **Claim 2 cannot be widened to all ten models as worded.**
  A weaker claim holds for all ten: every one of the 148 layers reaches AUROC 0.927 or above.
  D = 10, the top of the grid, is selected at 46 of the 48 control layers; all four gate
  models (LaTa, PhilTa, mT5-base, LaBSE) select D = 10 at every layer too.
- Compute: 11.5 CPU core-hours charged (main job 10.6, pilots 0.9); 5.5 core-hours used.

## Provenance

| Item | Value |
|---|---|
| Scripts | `scripts/paper/reframe/d2_controls.py` (new driver: stages `config`, `extract`, `panel`, `projection`, `render`); `scripts/paper/reframe/p2x2_panel.py` (new opt-in `EXTRA_MODELS` list holding T5-efficient-base, scored only when named in `--models`; the default ten-model run, its gates and its tables are unchanged) |
| Job | `slurm/reframe/d2_controls.sbatch`, job **22753661** (cpu, 16 cores, 32 GB): 39:44 elapsed of 1:10:00 reserved, 10.6 core-hours charged, TotalCPU 5:27:26, MaxRSS 5.4 GB. Stage times: config 30 s, extraction 14 min, panel 3 min, tokenizer sensitivity 5 min, projection 19 min, render 1 s |
| Pilots | job 22753558 (`PILOT=1`, 2:30, 0.67 core-hours: config stage, 64-passage extractions, gated projection at layers 6 and 12 of the four gate models, about 10 s per layer for the three methods); job 22753532 (0:52, 0.23 core-hours, failed: T5-v1.1-base's model card was not in the offline HF cache) |
| Models (HF revision, pinned in `D2_MODELS`; the extractor refuses any other snapshot) | `google/t5-efficient-base` `1f496af48641`, `google-t5/t5-base` `a9723ea7f1b3`, `google/t5-v1_1-base` `b5fc947a416e` (the GEN pin), `bowphs/LaBerta` `94fab85783dc`, `bowphs/PhilBerta` `acfe65d43b93`; the HF `main` commits on 2026-10-08. The #248 extraction did not record revisions; the 1e-7 agreement below indicates the same weights. fp32, torch 2.10.0, transformers 4.57.6 |
| Split | `runs/active/resubmit/data/phase_resubmit_split.csv` (847 train / 858 test) |
| Committed outputs | `runs/active/reframe/d2/`: `d2_config_check.csv`, `p2x2_layers.csv` + `p2x2_repro.csv` (the panel script's own outputs for LaTa and the five D2 models), `d2_repro.csv` (comparability), `d2_projection.csv` (324 rows: 9 models x 12 layers x baseline / ABTT D=10 / ABTT D on train), `d2_facts.md`, `t5tok/p2x2_layers.csv` (tokenizer sensitivity); tables `overleaf_drafts/tables/d2_t5_efficient.tex` (`tab:d2_t5_efficient`) and `d2_controls_abtt.tex` (`tab:d2_controls_abtt`), not `\input` anywhere |
| Not committed | the vectors, `runs/active/reframe/d2/bases/` (301 MB) and `bases_t5tok/` (61 MB), gitignored; copied byte for byte (`cp -a -n`, checked with `diff -r`) to the main checkout's `/u/irowerojas/localLatin/runs/active/reframe/d2/` |
| Tests | `tests/test_d2_controls.py`: the opt-in model list, gin parsing and the layout check, the collapse read-out, the band read at three decimals, the comparability check, the published-cell gate (all synthetic, in CI); the committed outputs (layout confirmed, every comparability cell and gate cell passes, the projection baseline equals the panel AUROC, byte-identical re-render of the facts and both tables), skipped when `runs/active/reframe/d2/` is absent |

Rerun (pinned snapshots must be in the HF cache; the job runs offline):

```bash
sbatch --export=ALL,CODE_ROOT=$PWD,REPO_ROOT=/u/irowerojas/localLatin slurm/reframe/d2_controls.sbatch
python scripts/paper/reframe/d2_controls.py --stage render   # facts and tables from the CSVs, seconds
```

### What "exactly as the other controls were run" means here

- **Representation.** Mean over attended tokens of encoder `hidden_states[1..12]`, minus the
  tokens the `tokenizer_empty` filter drops, `max_length` 512, batch 8, fp32. T5 checkpoints:
  the encoder of `AutoModelForSeq2SeqLM` pooled with `extract_hidden_cli.pool_hidden`, as
  `src/extract_hidden_cli.py` and `gen_extract.py` do; encoder-only checkpoints: `AutoModel`
  pooled with `extract_encoder_cli.pool_embeddings`, as `src/extract_encoder_cli.py` does. The
  pooling functions are imported, not copied. The differences are operational: one forward pass
  per batch for all 12 layers, batches sorted by length, CPU instead of GPU. The comparability
  check shows these change nothing at the precision the paper prints.
- **Scoring.** `p2x2_panel.py --stage compute` itself, on LaTa (so gates a, b and d run) plus the
  five D2 caches: Task A test AUROC through the paper's metric block, train top-PC share,
  effective rank and mean pairwise cosine through `gen_ft_geometry.layer_stats`, rows aligned by
  filename through each cache's `meta.csv`.
- **Projection.** `run_resubmit_evaluate.evaluate_single`, the function that produced
  `phase_resubmit_results.csv`, with `abtt_fixed` (D = 10) and `abtt_optimal` (D chosen on
  training DirAcc@1 over 1, 2, 3, 5, 7, 10, the grid of `slurm/resubmit/*evaluate*.sbatch`).
  Before the controls, the stage recomputes LaTa, PhilTa, mT5-base and LaBSE at layers 1-12 for
  all three methods and requires the published cells: 144 cells equal on AUROC, test DirAcc@1
  and training DirAcc@1 within 1e-6, and D exactly. Gate passed.

## Architecture check (stage `config`)

From the pinned `config.json`, the stored weights, the model card and, for T5-efficient-base,
the released Mesh TensorFlow `operative_config.gin`:

| | T5-efficient-base | T5-base | T5-v1.1-base |
|---|---|---|---|
| `feed_forward_proj` | relu | relu | gated-gelu |
| `tie_word_embeddings` | True (default; key absent) | True (default; key absent) | False |
| stored `lm_head` distinct from the input embedding | no (stored copy equals `shared`) | no (not stored) | yes |
| layers / d_model / d_ff | 12 / 768 / 3072 | 12 / 768 / 3072 | 12 / 768 / 2048 |
| pretraining mixture (released gin `MIXTURE_NAME`) | `c4_v220_unsupervised` (C4 only) | `all_mix` (C4 mixed with supervised tasks) | `c4_v020_unsupervised` (C4 only) |
| model card `datasets` | c4 | c4 | c4 |
| pretraining steps | 524,288 (gin `run.train_steps`) | about 1M (released checkpoint `model.ckpt-999900`; gin `train_steps` is the open-ended 1e9) | 1,000,000 (gin; checkpoint `model.ckpt-1000000`) |
| tokens per batch (gin `run.batch_size`) | 65,536 | 1,048,576 | 1,048,576 |
| pretraining tokens | about 34.4B | about 1.05T | about 1.05T |
| pretraining dropout | 0.0 (gin) | 0.1 (gin; Raffel et al. 2020) | off, by the T5.1.1 release notes (the released pretraining gin carries `dropout_rate = 0.1`; the two sources disagree) |
| gin `shared_embedding` / encoder activation | True / relu | True / relu (`shared_embedding_and_softmax_weights = True`) | True / gelu, linear (`shared_embedding_and_softmax_weights = False`) |

The model card calls it "a *pretrained-only* checkpoint" trained "on the Colossal, Cleaned
version of Common Crawl (C4) for 524288 steps using the span-based masked language modeling
(MLM) objective", released with Tay et al. (2022), *Scale Efficiently* (ICLR 2022,
arXiv:2109.10686). It is the original layout pretrained on C4 only, so the run went ahead.

The T5-base and T5-v1.1-base rows come from the released Mesh TF configurations and checkpoint
indices, read on 2026-10-08:
`https://storage.googleapis.com/t5-data/pretrained_models/base/operative_config.gin` and
`.../base/checkpoint`, `.../t5.1.1.base/operative_config.gin` and `.../t5.1.1.base/checkpoint`.
The token ratio is 999,900 x 1,048,576 / (524,288 x 65,536) = 30.5 for T5-base, and 30.5 for
T5-v1.1-base at 1,000,000 steps. The review round quoted 32, which would hold for 2^20 steps;
the released checkpoints sit at 999,900 and 1,000,000 steps, so this memo says "about 30
times".

**Tokenizer.** T5-efficient-base ships its own `tokenizer.json`. It differs from T5-base's in
one respect: on 471 passages it emits exactly one extra token (in the passages inspected, a
lone `▁` after trailing whitespace, before `</s>`). The pooling filter drops that token, so the
pooled token ids are identical to T5-base's on all 1,705 passages (0 differ). The token still enters attention. Reading
T5-efficient-base with T5-base's tokenizer (`--tokenizer_from T5-base`) changes AUROC by at
most 0.0013 and top-PC share by at most 0.061 (layer 6: 0.304 against 0.365). The peak share
is 0.615 (layer 11) and the lowest mid-depth AUROC 0.736, so the reading does not change.

## Comparability check

| model | AUROC max abs diff | top-PC share max abs diff | eff. rank max rel diff | mean cos max abs diff |
|---|---|---|---|---|
| T5-base | 1.2e-7 | 2.0e-8 | 7.7e-8 | 3.5e-9 |
| T5-v1.1-base | 1.8e-7 | 1.0e-7 | 3.3e-7 | 1.0e-8 |
| LaBERTa | 8.0e-8 | 7.2e-8 | 2.4e-7 | 6.2e-8 |
| PhilBERTa | 1.8e-7 | 1.9e-7 | 7.5e-7 | 1.1e-7 |

12 layers each, tolerance 1e-4 (the tolerance `p2x2_panel.py` uses across hardware, gate c).
The panel script's own gates on the same run: a (LaTa AUROC against the published cells) 1.1e-16;
b (LaTa train geometry) 1.1e-16 / 8.5e-16 relative; c (T5-v1.1-base against the #244 CPU
extraction) 6.2e-8; d (printed cells: LaTa, T5-v1.1-base, LaBERTa, PhilBERTa) 14 of 14.

## Results

### (i) T5-efficient-base on the Latin corpus

Unmodified mean-pooled vectors. AUROC: Task A test. PC1, rank, cos: 847 training passages.

| Layer | AUROC | PC1 | Rank | Mean cos | ABTT AUROC | T5-base AUROC | T5-base PC1 |
|---|---|---|---|---|---|---|---|
| 1 | 0.820 | 0.164 | 49.85 | 0.994 | 0.975 | 0.855 | 0.228 |
| 2 | 0.822 | 0.137 | 54.66 | 0.995 | 0.968 | 0.864 | 0.250 |
| 3 | 0.812 | 0.190 | 48.51 | 0.994 | 0.973 | 0.864 | 0.234 |
| 4 | 0.798 | 0.193 | 47.78 | 0.994 | 0.974 | 0.849 | 0.305 |
| 5 | 0.794 | 0.231 | 45.57 | 0.994 | 0.971 | 0.848 | 0.412 |
| 6 | 0.803 | 0.304 | 35.04 | 0.993 | 0.967 | 0.843 | 0.489 |
| 7 | 0.809 | 0.263 | 37.85 | 0.992 | 0.969 | 0.849 | 0.526 |
| 8 | 0.812 | 0.327 | 32.52 | 0.991 | 0.970 | 0.847 | 0.545 |
| 9 | 0.794 | 0.260 | 36.81 | 0.990 | 0.964 | 0.836 | 0.534 |
| 10 | 0.782 | 0.276 | 34.43 | 0.987 | 0.962 | 0.826 | 0.513 |
| 11 | **0.736** | **0.627** | **8.10** | 0.973 | 0.959 | 0.816 | 0.494 |
| 12 | 0.827 | 0.213 | 66.67 | 0.891 | 0.964 | 0.843 | 0.121 |

Read-out against the spine's test (layers 2-11):

| model | layers with AUROC < 0.70 | of those, PC1 >= 0.76 | lowest AUROC (layer) | peak PC1 (layer) | collapses |
|---|---|---|---|---|---|
| T5-efficient-base | 0 | 0 | 0.736 (11) | 0.627 (11) | no |
| T5-base | 0 | 0 | 0.816 (11) | 0.545 (8) | no |
| T5-v1.1-base | 10 | 10 | 0.489 (2) | 0.975 (9) | yes |
| LaTa (reference) | 10 | 10 | 0.496 (6) | 0.952 (4) | yes |

Two side observations, neither needed for the read-out:
- T5-efficient-base sits at mean pairwise cosine 0.97-0.995 at layers 1-11 and still ranks at
  0.74-0.82. That is another case where mean cosine flags a model that does not collapse
  (claim 5), and a stronger one than T5-v1.1-base's 0.87-0.96.
- It ranks below T5-base at every layer, by 0.016 (layer 12) to 0.080 (layer 11). Its layer 11
  combines the lowest AUROC with a top-PC share of 0.627 and effective rank 8.1, a milder form
  of the collapsed geometry (collapsed layers: share at least 0.76, effective rank about 1 to 4.3). We
  cannot say from one checkpoint whether this gap comes from the pretraining mix, dropout, or
  run-to-run variation, so it should not be read as a partial effect of supervised pretraining.

### (ii) The projection on the four controls

Test AUROC after ABTT, D chosen on training DirAcc@1 (the panel's `abtt_optimal`; D in brackets
where it is not 10). Base: unmodified vectors.

| Layer | LaBERTa base | ABTT | PhilBERTa base | ABTT | T5-base base | ABTT | T5-v1.1-base base | ABTT |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.826 | 0.974 | 0.893 | 0.976 | 0.855 | 0.981 | 0.848 | 0.970 (7) |
| 2 | 0.832 | 0.977 | 0.905 | 0.977 | 0.864 | 0.981 | 0.489 | 0.969 |
| 3 | 0.839 | 0.972 | 0.913 | 0.978 | 0.864 | 0.974 | 0.493 | 0.965 |
| 4 | 0.830 | 0.969 | 0.903 | 0.974 | 0.849 | 0.971 | 0.505 | 0.963 |
| 5 | 0.828 | 0.967 | 0.885 | 0.976 | 0.848 | 0.969 | 0.515 | 0.963 |
| 6 | 0.834 | 0.963 | 0.883 | 0.974 | 0.843 | 0.967 | 0.523 | 0.959 |
| 7 | 0.836 | 0.964 | 0.896 | 0.974 | 0.849 | 0.962 | 0.533 | 0.952 |
| 8 | 0.844 | 0.963 | 0.905 | 0.975 | 0.847 | 0.966 | 0.536 | 0.945 |
| 9 | 0.857 | 0.965 | 0.911 | 0.979 (7) | 0.836 | 0.969 | 0.533 | 0.934 |
| 10 | 0.873 | 0.968 | 0.919 | 0.978 | 0.826 | 0.962 | 0.531 | 0.930 |
| 11 | 0.892 | 0.966 | 0.913 | 0.980 | 0.816 | 0.959 | 0.522 | 0.927 |
| 12 | 0.910 | 0.967 | 0.923 | 0.976 | 0.843 | 0.973 | 0.772 | 0.959 |

| set | layers | AUROC after ABTT | below 0.962 (three decimals) | above 0.987 |
|---|---|---|---|---|
| six-model panel (published) | 100 | 0.962-0.987 | 0 | 0 |
| four controls | 48 | 0.927-0.981 | 8 (T5-v1.1-base 6-12, T5-base 11) | 0 |
| all ten models | 148 | 0.927-0.987 | 8 | 0 |
| ten + T5-efficient-base | 160 | 0.927-0.987 | 9 (adds T5-efficient-base 11, 0.959) | 0 |

Fixing D = 10 gives the same ranges: min 0.927, max 0.987, the same eight layers below the
band. T5-v1.1-base's fixed-D cells differ from the selected ones only at layer 1 (0.978 at D = 10
against 0.970 at the selected D = 7).

Routing after ABTT, for the record only (claim 2 is about ranking): test DirAcc@1 spans 74.6-80.4
(LaBERTa), 75.2-83.2 (PhilBERTa), 76.3-88.6 (T5-base) and 57.9-87.4 (T5-v1.1-base) over layers.

## Reading

**(i).** Under the test fixed before the run, T5-efficient-base does not collapse. The spine
predicted that "the layout account survives this control" in that case, and it does, with a
budget caveat. Of the two differences between T5-base and T5-v1.1-base that the paper could not
separate, pretraining mix and layout, this checkpoint takes T5-base's layout and T5-v1.1-base's
pretraining data (C4 only), without dropout, at about 1/30 of their pretraining
tokens, and stays healthy. So a model with the original layout can avoid the collapse without
supervised tasks or pretraining dropout, at least at this budget; longer C4-only pretraining of
the original layout is untested. The supervised-mixture explanation does not become the leading
account, but this run does not rule out a part for it at T5-base's budget. Limits that stay:

- One checkpoint per cell, from a separate and much shorter pretraining run: about 34B tokens
  against about 1T for T5-base and T5-v1.1-base (released gin files), on a different C4 release
  (c4_v220 against T5-v1.1-base's c4_v020), with a different seed.
- The T5 v1.1 changes, gated-GELU feed-forward and untied embeddings, still vary together. This
  run does not say which of them matters, and nothing here makes the layout a cause.
- T5-efficient-base is weaker than T5-base at every layer and dips to 0.736 at layer 11 with
  some low-rank geometry. The paper should report this and not stretch "does not collapse" into
  "is as healthy as T5-base". With the budgets 30-fold apart, this gap cannot be assigned to the
  supervised mix either.

**(ii).** Claim 2's band does not carry over. LaBERTa and PhilBERTa, which never collapse, land
inside it. T5-base misses by 0.003 at its weakest layer. T5-v1.1-base, the one control that
collapses, is restored from near chance (0.489-0.536) to 0.927-0.969, but at seven layers it
stays below the band and falls with depth. "Masked, not missing" holds for T5-v1.1-base too:
ten layers at chance rise above 0.92 with a projection fit on training vectors alone. But
"brings every layer into 0.962-0.987" is a panel fact, and the widened statement has to give
the lower floor, 0.927. D = 10 is the top of the selection grid and is selected at 46 of the 48
control layers, as at every layer of the four gate models, so a larger D might close some of the
gap. We did not test that: the
recipe is fixed by the panel.

## Proposed paper text

Sentences are quoted from `overleaf_drafts/acl_latex.tex` at `d011691`; the eight-page cut in
progress may have moved or merged them. All replacements keep "tracks", make no causal claim
about the layout, state the pretraining budget, and keep claim 2's band scoped to the panel.
Items 1 to 6 are required edits; item 7 applies only if T5-efficient-base joins the main-text
model table. The token ratio is written "about 30 times" (30.5, see the architecture check),
correcting the review round's "32".

**Integration.** Both appendix tables must be `\input`. Put `\input{tables/d2_t5_efficient}` in
the models appendix (`app:models`), right after the "T5-base and T5-v1.1-base" paragraph that
item 6 extends. Put `\input{tables/d2_controls_abtt}` in the same appendix, after the
T5-efficient-base sentences, or beside the per-layer tables (`app:per_layer`) if that keeps the
models appendix shorter. Items 1, 4 and 6 cite `app:models`, `tab:d2_controls_abtt` and
`tab:d2_t5_efficient`, so all three labels must resolve.

**Bib entry** (`overleaf_drafts/custom.bib`; the ten authors in the order of arXiv:2109.10686;
venue from its arXiv comment, "ICLR 2022"):

```bibtex
@inproceedings{tay2022scale,
    title = {Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers},
    author = {Tay, Yi and Dehghani, Mostafa and Rao, Jinfeng and Fedus, William and Abnar, Samira and Chung, Hyung Won and Narang, Sharan and Yogatama, Dani and Vaswani, Ashish and Metzler, Donald},
    booktitle = {International Conference on Learning Representations},
    year = {2022},
    url = {https://arxiv.org/abs/2109.10686}
}
```

**1. Section 5, T5 paragraph.**
Current:
> Across the ten models we test, the collapse tracks the T5 v1.1 layout; we do not isolate which change is responsible.
> T5-base also saw supervised tasks in pretraining \citep{raffel2020t5}, so this contrast does not separate layout from pretraining mix.

Proposed:
> T5-base also saw supervised tasks in pretraining \citep{raffel2020t5}, so we add T5-efficient-base \citep{tay2022scale}, which has T5-base's layout but, like T5-v1.1-base, was pretrained on C4 alone; it was also pretrained without dropout and on about 30 times fewer tokens than either.
> It does not collapse either: its lowest AUROC is 0.736 and its top-PC share peaks at 0.627 (Appendix~\ref{app:models}).
> Across the ten models in Table~\ref{tab:models} and T5-efficient-base, the collapse tracks the T5 v1.1 layout; we do not isolate which of its changes is responsible.

**2. Introduction, the same two sentences.**
Current:
> Across the ten models we test, the collapse tracks the T5 v1.1 layout; we do not isolate which change is responsible.
> T5-base also saw supervised tasks in pretraining, so this contrast does not separate layout from pretraining mix.

Proposed:
> Across the ten models in Table~\ref{tab:models} and T5-efficient-base, the collapse tracks the T5 v1.1 layout; we do not isolate which change is responsible.
> T5-efficient-base, with T5-base's layout but pretrained on C4 alone and far more briefly, does not collapse either.

**3. Counts in the abstract and the introduction (required whether or not the model table gains a row).**
Abstract, current:
> Four of the five raw T5-family encoders we test, these three and T5-v1.1-base, collapse, and all four share the T5 v1.1 layout; encoder-only siblings pretrained on the same corpora and the original T5-base do not.

Proposed:
> Four of the six raw T5-family encoders we test, these three and T5-v1.1-base, collapse, and all four share the T5 v1.1 layout; encoder-only siblings pretrained on the same corpora and two checkpoints with the original T5 layout do not.

Introduction, current:
> Four of the five raw T5-family encoders we test collapse, and all four share the T5 v1.1 layout: a gated-GELU feed-forward block \citep{shazeer2020glu} and untied input and output embeddings \citep{t5v11release}.

Proposed:
> Four of the six raw T5-family encoders we test collapse, and all four share the T5 v1.1 layout: a gated-GELU feed-forward block \citep{shazeer2020glu} and untied input and output embeddings \citep{t5v11release}.

**4. Section 5, the projection sentence.**
Current: "We did not apply the projection to the controls (Limitations)."
Proposed:
> On the four controls ABTT lifts every layer to AUROC 0.927 or above, the ten collapsed layers of T5-v1.1-base included (0.927 to 0.969), but eight of their 48 layers, seven of them in T5-v1.1-base, stay below the panel's band (0.962 to 0.987; Table~\ref{tab:d2_controls_abtt}).

**5. Limitations (iii) and the scope paragraph.**
(iii), current:
> (iii)~The T5 v1.1 changes (gated-GELU feed-forward, untied embeddings) vary together, and T5-base also saw supervised tasks in pretraining, so our models do not separate layout from pretraining mix.

Proposed:
> (iii)~The T5 v1.1 changes (gated-GELU feed-forward, untied embeddings) vary together, so we do not say which of them matters; T5-efficient-base has the original layout without T5-base's supervised pretraining, but it is a single checkpoint pretrained on about 30 times fewer tokens, so it separates layout from pretraining mix only at that budget.

Scope paragraph, current: "We claim the projection result for the six-model panel only, since
we did not apply the projection to the four controls."
Proposed:
> We claim the narrow band for the six-model panel only: on the four controls ABTT lifts every layer to AUROC 0.927 or above, but eight of their 48 layers, seven of them in T5-v1.1-base, stay below the panel's band (0.962 to 0.987).

**6. Appendix, "T5-base and T5-v1.1-base" paragraph: append.**
> T5-efficient-base \citep{tay2022scale} has T5-base's ReLU feed-forward with inner width 3072, tied embeddings, vocabulary, depth and width, and T5-v1.1-base's C4-only pretraining data; it was pretrained without dropout and on a much shorter run: 524,288 steps of 65,536 tokens, about 1/30 of the tokens of either, by the released training configurations.
> On the Latin corpus it never falls below 0.736 (layer 11), and its top-PC share stays at or below 0.627 (Table~\ref{tab:d2_t5_efficient}).
> It ranks below T5-base at every layer, by 0.016 to 0.080, and its layer 11 has effective rank 8.1, a mild form of the low-rank profile.
> Its tokenizer adds one whitespace token to 471 passages, which the pooling filter drops; reading it with T5-base's tokenizer changes no AUROC by more than 0.002 and no top-PC share by more than 0.07 (peak 0.615).
> Its mean pairwise cosine is 0.97 to 0.995 at layers 1 to 11, so a high mean cosine also flags a model that does not collapse.

**7. Only if T5-efficient-base joins the main-text model table.** In the Section 5 opener, "Of the
ten models in Table~\ref{tab:models}, four collapse" and "none of the other six collapses"
become eleven and seven. "Across the ten models in Table~\ref{tab:models} and
T5-efficient-base" in items 1 and 2 becomes "Across the eleven models in
Table~\ref{tab:models}". The appendix title "All Twelve Models" becomes "All Thirteen Models"
if its table gains the row (`spine_tables.py` would then need to read
`runs/active/reframe/d2/p2x2_layers.csv`).

**Spine bookkeeping.** Claim 3's wording-limits cell ("the run that would separate the two is
deferred, see D2") and D2 itself are now done. The D4 title argument ("T5.1.1 Encoders" names a
cause we have not isolated) still holds: this run separates the layout from the pretraining
mix at a budget about 30 times smaller than T5-base's, not one layout change from the other.

**Had T5-efficient-base collapsed (not the outcome).** Item 1 would have read: "T5-efficient-base,
with T5-base's layout but pretrained on C4 alone, collapses as well, so the collapse does not
track the T5 v1.1 layout alone, and T5-base's supervised pretraining is the leading account of
its health." The abstract's "and all four share the T5 v1.1 layout" and the title discussion
would have had to drop the layout framing.
