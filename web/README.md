# localLatin Scholar Review — Backend

FastAPI backend for reviewing model predictions on unlabelled Latin manuscript fragments.

## Setup

### Install dependencies

```bash
conda activate localLatin
pip install -r web/requirements.txt
```

### Configure

```bash
cp web/config.yaml.example web/config.yaml
# Edit paths if needed (defaults match the NCSA Delta layout)
```

### Run (development)

```bash
cd /projects/beto/irowerojas/localLatin
python -m web
```

The server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Run with custom config

```bash
LOCALLATIN_CONFIG=/path/to/config.yaml python -m web
```

### Run on HPC login node

```bash
module load miniforge3-python
conda activate localLatin
python -m web
# Access via SSH tunnel: ssh -L 8000:localhost:8000 user@delta.ncsa.illinois.edu
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/queries` | Paginated query list (search, filter by status) |
| GET | `/api/query/{file_id}` | Query text + tokenized view |
| GET | `/api/query/{file_id}/predictions?model=` | Top-10 predictions with candidate text |
| GET | `/api/query/{file_id}/predictions/{rank}/candidates?model=` | Full candidate directory text |
| GET | `/api/token_map_examples` | List available IG token map examples |
| GET | `/api/token_map/{example_id}` | Token-to-token similarity + IG weights |
| POST | `/api/feedback` | Save scholar review feedback |
| GET | `/api/feedback/export` | CSV export of all feedback |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/models` | Available models |

## Architecture

- **Startup**: Loads all text files (~20MB) and prediction CSVs into memory for O(1) lookups
- **Feedback**: SQLite database at `runs/phase_resubmit/webapp/feedback.db`
- **Token maps**: Precomputed IG artifacts (LaTa + PhilTa only) served as reference demos
- **No GPU required**: All computation uses cached data + numpy
