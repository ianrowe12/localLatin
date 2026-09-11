# localLatin Webapp

Scholar review interface for the [localLatin](https://github.com/ianrowe12/localLatin) manuscript retrieval research project. Enables reviewers to manually verify model predictions for Latin manuscript text assignment.

## Architecture

- **Backend**: FastAPI (Python 3.10+) with in-memory data store and SQLite feedback DB
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **Data**: Reads pre-computed predictions and corpus texts from the research repo (no ML dependencies)

## Setup

### 1. As a subtree inside the research repo (recommended)

The webapp is embedded at `web/` in the research repo via git subtree:

```bash
cd localLatin
cp web/config.yaml.example web/config.yaml
# Edit web/config.yaml: set data_root to ".." (default)
pip install -r web/requirements.txt
python -m web
```

Frontend development:
```bash
cd localLatin/web/frontend
npm install
npm run dev        # with backend running
npm run dev:mock   # standalone with mock data
```

### 2. Standalone (separate clone)

```bash
git clone https://github.com/ianrowe12/localLatin-webapp.git
cd localLatin-webapp
pip install -e .

# Configure data path
cp config.yaml.example config.yaml
# Edit config.yaml: set data_root to path of your localLatin research repo

python -m web
```

Or use the environment variable:
```bash
LOCALLATIN_DATA_ROOT=/path/to/localLatin python -m web
```

### Run on HPC login node

```bash
module load miniforge3-python
conda activate localLatin
python -m web
# Access via SSH tunnel: ssh -L 8000:localhost:8000 user@delta.ncsa.illinois.edu
```

## Data Requirements

The webapp reads these files from `data_root` at startup:

| Path | Description |
|------|-------------|
| `canon_unlabelled/` | 2,238 query .txt files |
| `canon_labelled/` | 859 directories of candidate .txt files |
| `runs/active/resubmit/unlabelled/unlabelled_predictions_<variant>.csv` | Model predictions per post-processing variant (`raw`, `abtt`, `sif`, `sif_abtt`) |
| `runs/active/ig_examples/phase12f_examples.csv` | IG example index |
| `runs/active/ig_examples/artifacts/` | Per-model NPZ files for token map visualization (6 model slugs x 20 pair examples) |
| `runs/active/resubmit/webapp/feedback.db` | Auto-created SQLite feedback storage |

All paths are configurable in `config.yaml` or via `LOCALLATIN_DATA_ROOT` env var.

### Token map artifacts

`/api/token_map/{example_id}` returns `pair_matrices[method][variant]`. Methods
are `ig`, `bertscore`, `ot`, `attention_weighted`, `dla`,
`attention_standalone` and `retrieval_mark`; the four variants are:

| Variant | Hidden states | Token aggregation |
|---------|---------------|-------------------|
| `baseline` | raw | unweighted |
| `abtt` | top-D principal components removed | unweighted |
| `sif` | raw | SIF weights `a / (a + p(t))`, `a = 1e-3` |
| `sif_abtt` | top-D principal components removed | SIF weights `a / (a + p(t))`, `a = 1e-3` |

`available_variants` on the response (and `variants_available` on each grouped
example card) lists what a given artifact actually carries, so the UI can drive
its selector from the data rather than a hardcoded list. Older two-variant
artifacts keep working unchanged.

Token text comes from the `query_token_strings` / `candidate_token_strings`
arrays stored in the NPZ; the HuggingFace tokenizer is only a fallback for
artifacts generated before those arrays existed. `query_sif_weights` /
`candidate_sif_weights` expose the mean-1 normalised SIF weight per token.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/queries` | List queries with pagination, search, filtering |
| GET | `/api/query/{file_id}` | Query detail with tokenized text |
| GET | `/api/query/{file_id}/predictions` | Top-10 predictions per model |
| GET | `/api/query/{file_id}/predictions/{rank}/candidates` | Full candidate directory text |
| GET | `/api/token_map_examples` | IG example summaries |
| GET | `/api/token_map/{example_id}` | Token-level IG visualization data |
| POST | `/api/feedback` | Submit reviewer feedback |
| GET | `/api/feedback/export` | Export feedback as CSV |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/models` | Available model metadata |

### Reviewer directory creation and recovery

Approved, signed-in reviewers can create a group with
`POST /api/reviewer_dirs {query_file_id, label?}`. Creation immediately saves
the directory and its seed membership, independently of feedback submission.
It returns 201, rejects an existing seed with 409, and rejects an account at
`MAX_REVIEWER_DIRS_PER_ACCOUNT` with 429. A duplicate takes precedence over the
account cap. Model, variant, query and guard-exclusion validation are unchanged.

`FeedbackDB.create_reviewer_dir` opens a dedicated SQLite connection and takes
write ownership with `BEGIN IMMEDIATE` before checking the seed and account
count. It inserts both rows and commits on that connection. SQLite serializes
competing creators even across accounts, connections or application processes.
Auth/session and feedback commits use the shared connection and cannot commit
or roll back this transaction. The creation connection closes on failure or
cancellation, rolling back uncommitted changes. A cancellation or lost response
during commit does not establish whether the group saved.

Recover saved identity with `GET /api/reviewer_dirs?seed_query_id=N`, not by
automatically repeating the POST. This existing endpoint returns every group
for that seed in insertion order, oldest database `id` first. The 409 detail
also names the oldest group; its response shape has not changed. A failed GET
is an unresolved outcome, not evidence that no group exists.

Historical duplicate seeds are preserved. The seed index remains non-unique:
adding a unique index would fail on existing duplicates, and deleting, merging
or relabeling groups to make it succeed would erase reviewer assertions.
Migration does not change any directory, membership or feedback row for this
safeguard. Every historical group stays visible, with its original members,
and counts toward its creator's cap. The serialized storage operation rejects
any additional group for an existing seed, including already-duplicate seeds,
without disabling protection for unused seeds. There is no automatic cleanup,
rename or removal workflow.

## Subtree Workflow (for research repo maintainers)

```bash
# Pull latest webapp changes into the research repo
git subtree pull --prefix=web webapp main --squash

# Push research-side web/ changes to the webapp repo
git subtree push --prefix=web webapp main
```

## Optional Dependencies

- `transformers`: Enables decoded token labels in token map visualization. Without it, tokens display as numeric IDs. Install with `pip install locallatin-webapp[tokenizers]`.
