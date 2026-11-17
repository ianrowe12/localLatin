# Dataset 2: scoring and paper parity

Dataset 2 contains reviewed unlabelled queries. All generated results, rankings
and paper assets stay in the gitignored `runs/active/dataset2/` directory.
Never commit snapshots, CCL text or reviewer identities. The commands report
aggregate counts and model identifiers, not reviewer identities.

## Delta run after merge

From the existing data checkout on Delta, after updating its code to `main`:

```bash
cd /u/irowerojas/localLatin
mkdir -p slurm/logs
CODE_ROOT=/u/irowerojas/localLatin \
SNAPSHOT=runs/active/dataset2/snapshot_20261115_run43762601690 \
sbatch /u/irowerojas/localLatin/slurm/resubmit/dataset2_score.sbatch
```

This requests the `cpu` partition, `beto-delta-cpu` account, 16 CPUs, 64 GB
and eight hours, with no GPUs. It checks the Qwen3-8B manifests, selects that
model's layers on benchmark-v1 train, runs the unchanged `score` command,
and runs `compare`, `layers` and `render`. `CODE_ROOT` can point at a code
checkout distinct from `DATA_ROOT`; the working directory remains `DATA_ROOT`
so cache and source paths resolve there. `SKIP_QWEN8B=1` skips manifest and
selection preparation only. `N_BOOT` and `SEED` default to 2000 and 289 and
are passed identically to `score` and `compare`.

The real run is not part of local validation. Tests use synthetic fixtures.

## Commands and outputs

Individual commands, from the data checkout:

```bash
python scripts/resubmit/score_dataset2.py score --snapshot "$SNAPSHOT"
python scripts/resubmit/score_dataset2.py compare --snapshot "$SNAPSHOT"
python scripts/resubmit/score_dataset2.py layers --snapshot "$SNAPSHOT"
python scripts/resubmit/score_dataset2.py render
```

`compare` requires the existing `score` outputs from the same snapshot, seed,
replicate count and cache metadata. It validates their metrics and intervals
while replaying the old draw schedule. Neither it nor the other new commands
rewrites any original `score` artifact.

| Output under `runs/active/dataset2/` | Contents |
| --- | --- |
| `results/reference_accuracy_long.csv` | Character n-gram acc@1/5/10 and MRR with 95% CIs, exclusions and draw provenance, every existing label set and seen-model group |
| `results/reference_per_query_ranks.csv` | Reference ranks against final and reviewer-only labels, pseudonymous query IDs only |
| `rankings/tfidf_char35__reference.npz` | Full reference directory ranking with exclusion masks |
| `results/best_methods_train.csv` | Best non-baseline method per model, chosen only by benchmark-v1 train DirAcc@1 |
| `results/paired_differences.csv` | Every model/method minus n-grams, and each train-best method minus its model's baseline, acc@1/5/10 and MRR, all label sets and seen-model groups |
| `results/comparison_info.json` | Seed, replicate count, bootstrap unit, selection rule and final/reviewer status counts, including conflicts |
| `results/per_layer.csv` | Agreed acc@1 and MRR, D and its train provenance, exclusions and status for every method and unlabelled cache layer |
| `paper/headline_main.tex` | Agreed main table: baseline and train-best method, acc@1 with CI, MRR and sample sizes, plus n-grams |
| `paper/headline_ci_diffs.tex` | Agreed paired differences for the train-best methods |
| `paper/fig_depth.pdf`, `paper/fig_depth.png`, `paper/fig_depth.tex` | All-method depth plot, both metrics, and a LaTeX figure wrapper |

`compare` and `layers` accept the original cache, split, selection and output
path overrides plus `--data_root` (default `data`). `render` reads result CSVs
and aggregate metadata only, not embeddings or the snapshot.

## Protocol

The reference reuses `lexical_baselines.py` exactly: shared text normalization,
`char_wb` TF-IDF with character 3--5-grams and L2 normalization, vocabulary and
IDF fitted on benchmark-v1 training witnesses only. Test witnesses and dataset-2
queries are transformed, never fitted. Directory scores are maximum cosine
over usable witnesses, with stable directory-order tie breaking. Blank and
out-of-vocabulary reference vectors are excluded, not arbitrary zero-score wins.

All four existing label sets are retained: `agreed`, `agreed_plus_conflict`,
`conflict`, `reviewers_agreed`. The last uses the separate reviewer-only
assignment file, including its own keys. Conflicts take the best rank among
their mapped keys. Unmapped keyed labels remain misses, not exclusions.
`no_key` has no known correct directory, so is counted but not assigned a
retrieval accuracy; the existing served-list diagnostic is unchanged.

Reference CIs use the exact legacy `score` draws when a cell has the same
non-excluded query cohort. If no such cell exists, an explicitly identified
named reference stream is necessary because the cohort differs. Paired CIs
resample the common non-excluded queries of both terms and take each
replicate's difference, not a difference of marginal interval endpoints.
These comparisons have their own named streams, independent of every legacy
stream and stable across model order. Infinite ranks are misses and remain
in the sample; only excluded (`NaN`) ranks leave it. The bootstrap unit is a
reviewed query, not dataset 1's test-directory unit.

Layers are always those selected on benchmark-v1 train. Among the selected
non-baseline methods, "best" maximizes train DirAcc@1, with ties in the
scorer's declared method order. It is not the best observed dataset-2 result.
For the depth sweep, an optimal method uses the recorded train D for that
specific layer; a missing record is retuned on benchmark-v1 train only.
Every unlabelled cache layer is reported even when its labelled counterpart
is missing (`missing_cache`). Whitening retains
`not_applicable_rank_deficient` when `n_train <= d`, with no numeric score.
Unavailable cells are gaps in the figure, not zeros. The figure describes
depth profiles, not an AUROC-based collapse threshold on dataset 2.
