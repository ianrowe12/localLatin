# Anisotropy in T5 and other encoder-decoder models: literature check

Issue #125 (epic #109). Search run 2026-09-06. Sources of record: ACL Anthology, arXiv API,
Europe PMC, PMLR. Every entry below was fetched from one of those; nothing is cited from memory.
Papers I could not verify are listed in the "Unverified" section and are not cited in the paper.

## 1. The claim under test

The draft says (abstract, Section 1, Section 5, Discussion, Limitations):

> Anisotropy is universal on this corpus, but catastrophic mid-depth collapse is not. LaTa,
> PhilTa and mT5-base fall to Task A AUROC 0.497, 0.539 and 0.654 in their middle layers, while
> LaBSE, Qwen3-0.6B and KaLM-mini improve close to monotonically with depth. ABTT repairs all six.

Two separable questions for the literature:

1. Is T5-family anisotropy known? (geometry)
2. Is a mid-depth **retrieval** collapse in T5-family **encoder** layers known? (task behaviour)

## 2. Verified findings

### 2.1 Papers that measure T5-family geometry

| Paper | Models | What was measured | Layers | Relevance |
|---|---|---|---|---|
| Godey, de la Clergerie, Sagot (2024), EACL 2024, pp. 35-48, `2024.eacl-long.3`, doi 10.18653/v1/2024.eacl-long.3 | BERT, RoBERTa, GPT-2, **T5-base**, CharacterBERT, CANINE-s/c, MANTa-LM, **ByT5**, plus speech and vision models | Average cosine similarity between hidden representations, per layer | All layers, but for T5-base "we concatenate encoder and decoder results" | The only verified paper that puts a T5 model on a layerwise anisotropy plot. Geometry only, no retrieval or similarity task. The encoder profile is not reported separately from the decoder. |
| Godey, de la Clergerie, Sagot (2023), ACL SRW 2023 poster, arXiv:2306.07656 | Same programme, earlier version | Same | Same | Precursor of the EACL paper. Cite the EACL version. |
| Ni, Hernandez Abrego, Constant, Ma, Hall, Cer, Yang (2022), Findings of ACL 2022, pp. 1864-1874, `2022.findings-acl.146`, doi 10.18653/v1/2022.findings-acl.146 | T5-base to T5-11B encoders and encoder-decoders | STS Spearman and SentEval transfer with mean pooling, first token, and encoder-decoder pooling | Final encoder layer only | Verbatim: "mean pooling of T5 embeddings performs poorly on STS, achieving an average correlation of 55.97", and "We believe that the pre-training corruption task of T5 does not require models to avoid anisotropy." This is the strongest published statement linking span corruption to T5 anisotropy. It is an inference about the last layer, with no geometric measurement and no layer sweep. |

Nothing verified measures per-layer geometry of the **T5 encoder alone**, and nothing measures
mT5 encoder geometry at all.

### 2.2 Layerwise geometry in encoder-only and decoder-only models

| Paper | Models | Metric | Finding |
|---|---|---|---|
| Razzhigaev, Mikhalchuk, Goncharova, Oseledets, Dimitrov, Kuznetsov (2024), Findings of EACL 2024, pp. 868-874, `2024.findings-eacl.58`, doi 10.18653/v1/2024.findings-eacl.58 | Encoders: BERT, RoBERTa, ALBERT. Decoders: OPT 125M-13B, Llama-2 7B-13B, GPT-2, GPT-J, Falcon-7B, Bloom, Pythia-2.8B, TinyLlama | anisotropy(X) = sigma_1^2 / sum_i sigma_i^2, the same top-PC variance share we report | Decoders show "a distinct bell-shaped curve, with the highest anisotropy concentrations in the middle layers"; encoders are "more uniformly distributed". **No encoder-decoder model is analysed**, and no downstream retrieval or similarity task is run. |
| Machina and Mercer (2024), NAACL 2024, pp. 4892-4907, `2024.naacl-long.274`, doi 10.18653/v1/2024.naacl-long.274 | Pythia suite, contrasted with previously studied anisotropic models | Isotropy of embedding spaces, training dynamics | Large Pythia models are isotropic. Anisotropy is therefore not inherent to Transformers, and is traced to the final LayerNorm rather than to the LM objective. Supports reporting anisotropy as family-level rather than universal. |
| Skean et al. (2025), ICML 2025, PMLR 267:55854-55875, arXiv:2502.02013 (already cited) | Pythia, Llama3, Mamba, BERT, LLM2Vec | 32 MTEB tasks per layer, plus entropy, curvature, LiDAR | Intermediate layers beat final layers by up to 16%. Explicitly a mid-depth **peak**, not a dip. No encoder-decoder model in the panel. |
| Timkey and van Schijndel (2021), EMNLP 2021 (already cited) | GPT-2, BERT, RoBERTa, XLNet | Contribution of single dimensions to cosine | Rogue dimensions dominate cosine and obscure representational quality. No T5. |
| Rajaee and Pilehvar (2022), Findings of ACL 2022 (already cited) | mBERT, XLM-R | Isotropy, outlier dimensions, six languages | Multilingual spaces are massively anisotropic; mBERT has no outlier dimensions while XLM-R does. Encoder-only. |
| Hämmerl, Fastowski, Libovický, Fraser (2023), Findings of ACL 2023, pp. 7023-7037, `2023.findings-acl.439`, doi 10.18653/v1/2023.findings-acl.439 | XLM-R, mBERT, multilingual sentence transformers | Outlier dimensions and anisotropy against Tatoeba, multilingual STS, BUCC | Zeroing outliers and isotropy-enhancing transforms improve cross-lingual similarity for pre-trained models; a fine-tuned sentence transformer is already isotropic and gains little. No T5 or mT5 in the panel (checked by full-text search of the PDF). |

### 2.3 Does geometry predict task behaviour?

| Paper | Models / task | Finding |
|---|---|---|
| Ait-Saada and Nadif (2023), ACL 2023 short, pp. 1194-1203, `2023.acl-short.103`, doi 10.18653/v1/2023.acl-short.103 | BERT and RoBERTa, text clustering on classic3, classic4, BBC, DBPedia, AG-news | Anisotropy has "a limited impact on the expressiveness of sentence representations"; high anisotropy can coexist with good clustering. Direct support for our separation of the geometry claim from the retrieval claim. |
| Jung, Park, Choi, Kim, Rhee, "Isotropic Representation Can Improve Dense Retrieval", arXiv:2209.00218 (2022, v2 2023) | ColBERT, RepBERT on MS-MARCO and Robust04 | Normalizing flow and whitening on BERT-based retrievers improve NDCG@10 by 5.2 to 22.8 percent. Repair-for-retrieval precedent, encoder-only, final layer. |
| Mikkelsen (2026), "Effects of Model Choice, Corpus Context, and Post Hoc Correction on Layer-Level Embedding Degradation in Clinical Document Retrieval", JMIR Medical Informatics 14, e99639, doi 10.2196/99639, PMID 42455615 | 13 configurations: non-retrieval-trained encoders (BERT-family), retrieval-trained encoders, decoder LLMs; 3 clinical corpora | Closest published work in method: layerwise MRR@10 and recall@10 alongside per-layer participation ratio, average pairwise cosine and anisotropy, plus corpus-only ZCA whitening as a post-hoc repair. Reports **monotone degradation with depth** in three non-retrieval-trained encoders and net improvement with depth in the other ten, not a mid-depth dip, and the panel contains no encoder-decoder model. Abstract verified through Europe PMC; the article is not open access, so the model list beyond the abstract was not re-read. |

### 2.4 Mechanism background (not needed for the claim)

- Dong, Cordonnier, Loukas (2021), "Attention is Not All You Need: Pure Attention Loses Rank
  Doubly Exponentially with Depth", ICML 2021, PMLR 139, arXiv:2103.03404. Pure attention
  converges to a rank-1 matrix with depth; skip connections and MLPs counteract it. A theoretical
  reason to expect depth-driven rank loss, not an observation about T5.
- Gao et al. (2019) representation degeneration and Mu and Viswanath (2018) ABTT are already in
  `custom.bib` and unchanged by this review.

### 2.5 Recent preprints, verified but not cited

Both are real arXiv records (fetched from the arXiv API) but unrefereed, so they stay out of the
paper. Recheck before camera-ready in case they appear at a venue.

- Parupudi, "Anisotropy Decides Cosine vs. Rank Metrics for Text Embeddings", arXiv:2606.29571
  (2026-06-28). Nineteen encoders, seven datasets. The fraction of variance in the single most
  dominant dimension predicts when cosine fails (rank correlation 0.86), and projecting out the
  dominant directions restores cosine. Same mechanism as our PC1-share diagnostic, on modern
  sentence encoders and LLMs rather than encoder-decoders.
- Bernas, Jourdan, Poché, Hudelot, "Revisiting Anisotropy in Language Transformers: The Geometry
  of Learning Dynamics", arXiv:2604.08764 (2026-04-09). Encoder-style and decoder-style models,
  gradient-geometry account of anisotropy during training.

## 3. Searched and not found (the gap)

Searches on ACL Anthology, arXiv, Europe PMC and general web, in several phrasings, returned no
paper that reports any of the following:

1. A per-layer retrieval or similarity profile of a **T5 encoder** (T5, mT5, ByT5, LaTa, PhilTa)
   in which mid-depth layers fall to chance and recover at the ends.
2. Any isotropy or anisotropy analysis of **mT5** at all.
3. Any layerwise study of **BART** geometry.
4. Any post-hoc repair (ABTT, whitening, SIF) applied per layer to an encoder-decoder encoder.
5. Any anisotropy analysis of Latin or historical-language models (Latin BERT, LaTa, PhilTa,
   SPhilBERTa).

Negative results from a search are weak evidence, so the paper wording below hedges accordingly.

## 4. Unverified, not cited

Web search surfaced several plausible-looking titles whose metadata I could not confirm on
arXiv or the ACL Anthology (mostly aggregator or topic-summary pages, e.g. emergentmind topic
pages on "anisotropy in embedding representations" and "middle-layer hidden states"). None is
cited, and none contradicted the gap above. Semantic Scholar's API refused every request
(HTTP 429, no API key), so it was not used as a source of record.

## 5. Verdict

**Partially known, and the retrieval-level result is new as far as this review can establish.**
The geometric ingredient is documented: T5-base and ByT5 appear on a cross-architecture layerwise
anisotropy plot, where the T5 and ByT5 decoders stand out as extremely anisotropic, though encoder
and decoder states are pooled into a single curve (Godey et al., 2024); and mean-pooled T5 encoder
embeddings are known to be weak for semantic similarity, which the ST5 authors attribute to span
corruption not penalising anisotropy (Ni et al., 2022), a statement about the final layer only.
What is not documented is the task-level phenomenon we report. The one paper that profiles
anisotropy layer by layer across architecture types finds the mid-depth peak in decoder-only
models and a flat profile in encoders, and includes no encoder-decoder model at all (Razzhigaev
et al., 2024); the one paper that pairs per-layer retrieval with per-layer geometry finds monotone
degradation with depth in encoder-only models and again has no encoder-decoder in its panel
(Mikkelsen, 2026); and the layerwise embedding-quality literature reports a mid-depth *peak*
rather than a dip (Skean et al., 2025). Two further results argue against stating the collapse as
a general law: anisotropy is not inherent to all Transformers (Machina and Mercer, 2024) and is
not always harmful downstream (Ait-Saada and Nadif, 2023), which is exactly the geometry-versus-task
split our six-model panel makes. The safe claim for the paper is therefore: T5-family anisotropy is
known, the mid-depth retrieval collapse of T5 **encoder** layers and its repair by ABTT are, to our
knowledge, not previously reported, and we attribute the collapse to the three T5 encoders we test
rather than to encoder-decoder architectures in general.
