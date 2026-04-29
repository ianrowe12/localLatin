# Run 1 Agent 2: Attribution Metric and Method Provenance

Date: 2026-04-29

Purpose: provide patch-ready support for the attribution section without
rewriting the paper. The main paper should be precise about which components
come from prior work, which are adapted, and which are ours.

## Recommended Framing

Use this division consistently:

| Component | Provenance | Paper wording |
|---|---|---|
| Integrated Gradients | Borrowed method from Sundararajan et al. (2017) | "We use integrated gradients as a local attribution method for a scalar retrieval score." |
| MaRC / RetrievalMarK | Adapted from Brinner and Zarriess (2023) | "We adapt MaRC's input-mask optimization from classification to pairwise retrieval." |
| Sufficiency and comprehensiveness | ERASER-family rationale-evaluation metrics, adapted to cosine | "We report retrieval-adapted sufficiency and comprehensiveness." |
| Minimum rationale fraction | Short-rationale / sparsity desideratum, not MaRC compactness | "We report the smallest token fraction needed to recover 80% of the full cosine." |
| LOO rank correlation | Perturbation faithfulness adapted from attention and saliency evaluation | "We report Spearman correlation between attribution magnitude and leave-one-out cosine drop." |
| ABTT | Borrowed post-processing method from Mu and Viswanath (2018) | "We apply ABTT after pooling, fitting principal components on the training split only." |
| Retrieval-cosine target | Ours | "The target scalar is the bi-encoder cosine, optionally after ABTT." |

Avoid claiming that ABTT generally improves attribution quality unless the
final metrics support that exact statement. The current safe claim is narrower:
ABTT changes the scalar decision function being explained, and the attribution
metrics should be read as retrieval-space diagnostics rather than as evidence
that any one attribution method is globally superior.

## Patch-Ready Paragraphs

### Methods: Integrated Gradients

```latex
For gradient-based explanations, we use integrated gradients
\citep{sundararajan2017axiomatic}. Integrated gradients assigns input-feature
attribution to any differentiable scalar function by integrating gradients
along a path from a baseline input to the observed input. In our setting the
scalar is not a class logit. It is the retrieval score
$S_v(q,c)=\cos(E_v(q),E_v(c))$, where $v$ denotes the uncorrected or
ABTT-corrected embedding variant. We therefore use integrated gradients as a
standard attribution method, while the choice of retrieval cosine as the
explained scalar is specific to this paper.
```

### Methods: Retrieval-Adapted MaRC / RetrievalMarK

```latex
Our learned-mask attribution is adapted from MaRC
\citep{brinnerzarriess2023marc}, which optimizes a soft input mask for a
frozen classifier so that the masked input preserves the target class score
while regularizers encourage sparse and smooth rationales. We keep the
per-instance mask-optimization idea, but replace the classification target
with pairwise retrieval cosine. Given a query-candidate pair, the mask is
optimized so that the masked query representation preserves
$\cos(E_v(q),E_v(c))$ with the candidate representation held fixed; the
candidate-side attribution is obtained by swapping the two sides. This is an
adaptation of MaRC's objective to bi-encoder retrieval, not a new training
procedure for the encoder.
```

### Methods: ABTT

```latex
We use All-but-the-Top (ABTT) post-processing
\citep{mu2018allbutthetop} as a train-free geometric correction after
pooling. For each model and layer, we center the pooled training embeddings,
estimate the leading principal components on the training split, and remove
the selected components from both train and test embeddings. ABTT itself is
borrowed from prior work; our use of it is the retrieval-specific application
to Medieval Latin semantic matching and to the scalar functions used by the
attribution methods.
```

### Metrics: Sufficiency and Comprehensiveness

```latex
To quantify faithfulness, we adapt rationale-evaluation metrics from the
ERASER line of work \citep{deyoung2020eraser} to retrieval. For an attribution
ranking over query tokens, sufficiency keeps the top-ranked tokens and measures
how much of the full pair cosine remains. Comprehensiveness removes the
top-ranked tokens and measures the resulting drop in pair cosine. In both
cases the class probability or class logit used in classification settings is
replaced by $S_v(q,c)=\cos(E_v(q),E_v(c))$.
```

### Metrics: Minimum Rationale Fraction

```latex
We also report a compactness-style sparsity diagnostic: the smallest fraction
of query tokens whose retention recovers at least $80\%$ of the full pair
cosine. This metric should be described as a minimum rationale fraction, or
as sparsity at a sufficiency threshold. It should not be attributed to MaRC's
compactness term, which is a smoothness/contiguity regularizer on the learned
mask rather than a rank-based recovery threshold.
```

### Metrics: Leave-One-Out Rank Correlation

```latex
As a perturbation-based faithfulness check, we compute the Spearman rank
correlation between token attribution magnitudes and leave-one-out drops in
the retrieval score. For each query token $i$, the perturbation score is
$\Delta_i = S_v(q,c)-S_v(q_{\setminus i},c)$. The metric is adapted from
rank-correlation uses of leave-one-out faithfulness in attention and saliency
evaluation \citep{jain2019attention}; it replaces the classification output
with the same retrieval cosine used by the other metrics.
```

### Limitations / Reviewer-Safe Wording

```latex
The attribution methods and metrics are local diagnostics for a fixed retrieved
pair. They do not establish a global explanation of model behavior, and they
should not be read as showing that the learned-mask method is uniformly more
faithful than integrated gradients or alignment baselines. Their role is to
make the retrieval score inspectable and to compare attribution rankings under
the same scalar function used for retrieval.
```

## Bibliography Checklist

Entries already present in `overleaf_drafts/custom.bib`:

- `sundararajan2017axiomatic`: Integrated Gradients. Current entry matches
  PMLR metadata: ICML 2017, PMLR 70:3319-3328.
- `brinnerzarriess2023marc`: MaRC. The entry should add DOI
  `10.18653/v1/2023.findings-acl.867` and URL
  `https://aclanthology.org/2023.findings-acl.867/`.
- `mu2018allbutthetop`: ABTT. Current OpenReview URL is appropriate.

Entries to add if the final rewrite includes the attribution-metrics
paragraphs:

```bibtex
@inproceedings{deyoung2020eraser,
  title = {{ERASER}: A Benchmark to Evaluate Rationalized {NLP} Models},
  author = {DeYoung, Jay and Jain, Sarthak and Rajani, Nazneen Fatema and Lehman, Eric and Xiong, Caiming and Socher, Richard and Wallace, Byron C.},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year = {2020},
  publisher = {Association for Computational Linguistics},
  pages = {4443--4458},
  doi = {10.18653/v1/2020.acl-main.408},
  url = {https://aclanthology.org/2020.acl-main.408/},
}

@inproceedings{jain2019attention,
  title = {Attention is not Explanation},
  author = {Jain, Sarthak and Wallace, Byron C.},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2019},
  publisher = {Association for Computational Linguistics},
  pages = {3543--3556},
  doi = {10.18653/v1/N19-1357},
  url = {https://aclanthology.org/N19-1357/},
}

@inproceedings{lei2016rationalizing,
  title = {Rationalizing Neural Predictions},
  author = {Lei, Tao and Barzilay, Regina and Jaakkola, Tommi},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  year = {2016},
  publisher = {Association for Computational Linguistics},
  pages = {107--117},
  doi = {10.18653/v1/D16-1011},
  url = {https://aclanthology.org/D16-1011/},
}
```

Optional entry:

```bibtex
@inproceedings{atanasova2020diagnostic,
  title = {A Diagnostic Study of Explainability Techniques for Text Classification},
  author = {Atanasova, Pepa and Simonsen, Jakob Grue and Lioma, Christina and Augenstein, Isabelle},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year = {2020},
  publisher = {Association for Computational Linguistics},
  pages = {3256--3274},
  doi = {10.18653/v1/2020.emnlp-main.263},
  url = {https://aclanthology.org/2020.emnlp-main.263/},
}
```

Use Atanasova et al. only for the broader diagnostic-evaluation family. Do not
make it the primary source for the specific Spearman leave-one-out correlation
metric, since the audit found that citation too loose.

## Reviewer-Risk Notes

1. **Compactness naming.** The current code and tables use `Compact@0.8` for
   "smallest fraction of tokens that recovers 80% of the cosine." This is
   useful, but it is not MaRC's compactness term. Safer table labels are
   `MinFrac@0.8`, `MinRationale@0.8`, or `Sparsity@0.8`.
2. **MaRC adaptation claim.** Say "adapted from MaRC" rather than "we use
   MaRC" if the loss differs from the original classifier objective and if the
   smoothness regularizer is total variation rather than MaRC's Gaussian
   bandwidth construction.
3. **IG scope.** IG can explain any differentiable scalar. The borrowed part is
   the attribution method. The retrieval-specific part is choosing pair cosine,
   after optional ABTT, as that scalar.
4. **ABTT scope.** ABTT is not ours. The contribution is diagnosing the
   retrieval geometry and applying train-split ABTT as the repair in this Latin
   semantic-matching pipeline.
5. **Attribution quality claims.** Do not say "ABTT improves attribution
   quality" unless the final table supports that across the named metrics and
   models. A safer claim is "ABTT changes the retrieval score being explained;
   we evaluate the resulting attribution rankings under the same score."
6. **Metric denominators.** Ratio metrics become unstable when the full cosine
   is close to zero. Keep the `FULL_COS_FLOOR` policy visible in table captions
   if those metrics are reported.
7. **Local explanation.** The heatmaps and masks explain one query-candidate
   score, not the model globally and not the human scholar's final decision.

## Source Checks

Primary sources verified while preparing this memo:

- Sundararajan, Taly, and Yan 2017, PMLR:
  https://proceedings.mlr.press/v70/sundararajan17a
- Brinner and Zarriess 2023, ACL Anthology:
  https://aclanthology.org/2023.findings-acl.867/
- DeYoung et al. 2020, ACL Anthology:
  https://aclanthology.org/2020.acl-main.408/
- Jain and Wallace 2019, ACL Anthology:
  https://aclanthology.org/N19-1357/
- Lei, Barzilay, and Jaakkola 2016, ACL Anthology:
  https://aclanthology.org/D16-1011/
- Mu and Viswanath 2018, OpenReview:
  https://openreview.net/forum?id=HkuGJ3kCb
