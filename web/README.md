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

### Reviewer directories, the CCL key, and the ranked list (issue #196)

Reviewer-created directories are **not ranked candidates**. `GET
/api/query/{file_id}/predictions` serves them in their own field,
`reviewer_dir_candidates`, best first and without a `rank`; `predictions` holds
the retrieval run's own ten and nothing else. They used to be appended at ranks
anchored on `MAX_MODEL_RANK + 1`, which evaluators read as the model predicting
them, and whose numbering changed as memberships changed. The PDF packets have
always listed them unranked (`services/pdf_packets.py`); the web list now agrees.

A document is filed into a reviewer directory by **naming its CCL key**, not by
pressing a rank. `POST /api/feedback` accepts an optional `ccl_key` alongside
`outcome: none_of_top_k` (and only there: a rank already names a directory). The
server normalises it — trimmed, inner whitespace collapsed, stored as typed,
matched case-folded (`services/ccl_keys.py`) — and takes exactly one of four
branches, recorded on the feedback row as `ccl_key_action` with the resolved
directory in `ccl_key_dir`:

| Branch | Meaning | Writes |
|--------|---------|--------|
| `matched_labelled_dir` | the key names a labelled corpus directory; `ccl_key_rank` records where it stood in this ranking, or null if it was not offered | assessment only |
| `joined_reviewer_dir` | the key names an existing reviewer directory | assessment + membership |
| `already_joined` | ... and this query was already a member | assessment only |
| `created_reviewer_dir` | nothing carries the key | assessment + directory + seed membership |
| `seed_taken` | nothing carries the key, but this query already seeds a directory. `ccl_key_dir` is **null**: the directory that blocked the write is not the one the key names | assessment only |

A directory is reached by its **key first, then its label**, both case-folded
(`_reviewer_dir_for_key`). The label is a second handle and never a name: a
directory created through the retired form has an empty `ccl_key`, and with the
rank route gone a key-only lookup would leave it permanently unjoinable while
its card kept showing its label, so typing that label would mint a duplicate.

An **identical repeat** -- same document, model, variant, account, key and note
as the caller's own newest row -- returns that row with **200** and appends
nothing. The log stays append-only: nothing is updated or removed, and a revised
note, a different key or another reviewer's submission is a new assertion and is
appended. A creation is refused before anything is written with **429**
(`MAX_REVIEWER_DIRS_PER_ACCOUNT`) or **422** (`UNSCORABLE_SEED`, the same guard
`create_reviewer_dir` makes for a degenerate query).

`FeedbackDB.insert_none_of_top_k` does all of it in ONE transaction on a
dedicated connection opened with `BEGIN IMMEDIATE`, so an assessment citing a
key cannot outlive the directory write it refers to. `correct_dir` is untouched
by this path and keeps its meaning: the directory a rank resolved to.

`reviewer_dirs.ccl_key` is the join key. The migration adds the column and fills
it only where the answer is already written down — a label that IS a CCL key by
the `scripts/data/label_taxonomy.py` rule. No label is ever rewritten, and a
non-key label keeps `ccl_key = ''`, which matches nothing.

### Reviewer directory creation and recovery

`POST /api/reviewer_dirs {query_file_id, label?}` still exists and is unchanged;
the reviewer UI no longer calls it, since the retired "New directory" button was
its only caller. Creation immediately saves
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
