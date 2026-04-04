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
| `runs/phase_resubmit/unlabelled/unlabelled_predictions.csv` | Combined model predictions |
| `runs/phase12f_examples/phase12f_examples.csv` | IG example index |
| `runs/phase12f_examples/artifacts/` | 40 NPZ files for token map visualization |
| `runs/phase_resubmit/webapp/feedback.db` | Auto-created SQLite feedback storage |

All paths are configurable in `config.yaml` or via `LOCALLATIN_DATA_ROOT` env var.

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

## Subtree Workflow (for research repo maintainers)

```bash
# Pull latest webapp changes into the research repo
git subtree pull --prefix=web webapp main --squash

# Push research-side web/ changes to the webapp repo
git subtree push --prefix=web webapp main
```

## Optional Dependencies

- `transformers`: Enables decoded token labels in token map visualization. Without it, tokens display as numeric IDs. Install with `pip install locallatin-webapp[tokenizers]`.
