# Read-only reviewer readiness

`scripts/webapp/check_reviewer_readiness.py` checks two pinned manuscripts under
all six reviewer models using `sif_abtt`. It uses Python's standard library, needs
no GPU or model downloads, and does not import or start the application.

This is separate from `smoke_reviewer_pilot.py`. The smoke script signs in,
registers an account, and approves/rejects account requests even without
`--write-check`. Do not use it for read-only incident diagnosis.

## Authorized execution

An owner must explicitly provide an **existing authorized session** and the
trusted service base URL. There is no sign-in, password, account provisioning,
session renewal, or login fallback. Missing, expired, unapproved, or
password-gated authentication cannot pass. This implementation work did not
collect production credentials or execute live acceptance for #155.

The input is only the value of the existing `locallatin_session` cookie, not a
`Cookie:` header, `Set-Cookie` line, browser export, password, or URL. Keep it in
an owner-controlled file with mode `0600`, outside the repository. Do not put the
value on the command line, in shell history, in reports, or in issue comments.
Do not create a new session to satisfy this tool.

With an already prepared private file:

```bash
python3 scripts/webapp/check_reviewer_readiness.py \
  --base-url https://your-authorized-service.example \
  --session-file /private/path/existing-session
```

Alternatively, read the same file through stdin without placing the token in
argv:

```bash
python3 scripts/webapp/check_reviewer_readiness.py \
  --base-url https://your-authorized-service.example \
  --session-stdin < /private/path/existing-session
```

These are owner-operated examples, not instructions to seek credentials.
The file is read, never rewritten. The CLI does not accept inline cookie or
password arguments, and argument errors do not echo their values.

HTTPS is required except for exact loopback hosts `127.0.0.1`, `localhost`, and
`::1`. A loopback URL checks the backend, not the public proxy/TLS path. An
optional base path such as `/review` is supported. Credentials, query strings,
fragments, and encoded base paths are rejected. No redirects are followed,
including same-origin redirects to login. Ambient HTTP proxies are disabled.
The explicitly supplied cookie goes only to the selected origin; response
cookies are ignored and never saved.

## What it reads

The only service routes requested are:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/auth/me` | Confirm an existing approved session without changing a password |
| GET | `/api/models` | Assert the expected six models and the fixed default pipeline |
| GET | `/api/queries` | Resolve each pinned filename through all search-result pages |
| GET | `/api/query/{id}` | Confirm filename/ID and query-text availability |
| GET | `/api/query/{id}/predictions?model=...&variant=sif_abtt&top_k=10` | Validate each model's actual ranked evidence |

It makes no POST, PUT, PATCH, DELETE, feedback, account-list, directory-creation,
configuration, deployment, or workflow-dispatch request. It creates no accounts,
sessions, reviewer directories, or feedback rows.

**Authenticated GET is not literally zero database writes.** The existing
`get_account_by_session` code updates `last_seen_at`. Normal session bookkeeping
is allowed by the owner decision in #155. The diagnostic neither opens the
feedback database nor promises that no database page changes.

## Pinned cases and acceptance

The ordinary case is `BAV1341.16r.7.txt`, the documented PhilTa backup example in
[`docs/research/abigail_demo_script.md`](../docs/research/abigail_demo_script.md),
section 5. The incident case is `BAV1341.22v.9.txt` from #155. Both are expected
to be evaluable. Historical/local numeric query IDs are never used.

Search is substring-based in the service. The diagnostic checks every page,
requires exactly one case-sensitive full-filename match, rejects duplicate IDs
and inconsistent pagination, then confirms the detail response identity.
The page limit is 10,000 search results; an incomplete or excessive result
fails rather than selecting a convenient match.

The expected models are fixed independently of the discovered catalog:

| Display name | Requested slug |
|---|---|
| LaTa | `bowphs_LaTa` |
| PhilTa | `bowphs_PhilTa` |
| LaBSE | `sentence-transformers_LaBSE` |
| mT5-base | `google_mt5-base` |
| KaLM-mini | `KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5` |
| Qwen3-0.6B | `Qwen_Qwen3-Embedding-0.6B` |

The research table contains Qwen3-8B instead of mT5-base and is not this contract.
The catalog must contain exactly these six unique entries with `sif_abtt`
available and default. A missing catalog entry cannot silently remove its
model from the prediction checks.

Each response must echo the resolved file ID, pinned filename, requested model
and variant. A usable model ranking has actual ranks 1 through 10 in order,
ten distinct nonblank directory identities, finite descending cosine scores,
and nonblank text for every listed candidate file at every model rank. Candidate
filenames must match `dir_files` without duplication. This is deliberately a
strict full-ranking check for these two pinned cases, not a new general API rule
for shorter rankings or `top_k` requests.

Reviewer-created candidates are counted separately. Their labels, notes,
attribution, or text never fill a gap in model evidence. Missing PhilTa,
a PhilTa 500, or blank PhilTa candidate text fails even if KaLM succeeds.
`prediction_count` in the catalog is a row count and is not used as evidence
that any ranking is usable.

## Status and output contract

The tool consumes #156's additive `PredictionResponse.status` field, preserved
from the per-query/model/variant CSV row:

| Source status | Interpretation |
|---|---|
| `ok` | Validate the actual model ranking and text |
| `excluded_blank_source` | Explicit blank-source exclusion, non-evaluable |
| `excluded_zero_norm` | Explicit zero-norm exclusion, non-evaluable |
| Absent or `null` | Legacy response; usable ranks can pass, an empty model list fails |
| Anything else | Unknown contract, fail |

Explicit exclusions are `EXCLUDED`, not failed HTTP requests or human
non-matches. An exclusion that also carries model ranks is contradictory and
fails. Since both pinned manuscripts are expected to be evaluable, an exclusion
prevents overall PASS even when the other eleven cases succeed.

One JSON report goes to stdout. Its allowlisted fields contain fixed status
codes, UTC start/end times, approved filenames/model slugs, validated numeric IDs
and model ranks, structural counts/booleans, and separately labelled provenance.
It never emits response bodies, manuscript text, previews, candidate filenames
or directory names, reviewer labels/notes, account identity, cookies, headers,
URLs, or raw exception messages. This also applies to malformed JSON, redirects,
authentication errors and `gh` failures.

| Exit | Overall status | Meaning |
|---|---|---|
| 0 | `PASS` | Both pinned cases have usable evidence for every required model |
| 1 | `FAIL` | A request, identity, catalog, ranking, text or response-format check failed |
| 2 | `BLOCKED` | Authentication, a refused redirect or invalid invocation prevents diagnosis |
| 3 | `NOT_READY` | No failures/blockers, but an expected manuscript is explicitly excluded |

FAIL takes precedence over BLOCKED, which takes precedence over exclusions.
Individual query/model results remain visible so a blocker or exclusion is not
lost in the aggregate. Requests use a 15-second socket timeout and an 8 MiB
response limit. There is no automatic retry.

## Provenance and optional GitHub reads

The existing service endpoints do not expose a running source SHA or loaded
data-release digest. Those `observed_service` fields are **unknown**, even on
a successful retrieval check. Response identities describe what the service
served; they do not identify its loaded CSV bytes.

`local_diagnostic_head` is the checkout containing this script, not the target
service's source. It also does not describe uncommitted local edits. Target disk
state is not inspected. User-supplied deployment labels are not collected or
treated as observations.

Optionally add `--deployment-run RUN_ID`. This uses `gh api --method GET` against
`github.com`, removes injected `GH_TOKEN`/`GITHUB_TOKEN`, disables prompts and
confirms that the saved login is `ianrowe12`. It reads the requested workflow run,
all job pages for that run attempt, and the current `DATA_RELEASE_TAG` repository
variable. It does not search for credentials, change the saved login, or make
GitHub writes.

Only the uniquely named **Deploy production** job with a matching workflow
head can report `COMPLETED`. `SKIPPED`, `FAILED`, `PENDING`, `UNKNOWN`, and
`BLOCKED` remain distinct. A green top-level workflow is not evidence that
deployment ran. Run `34079707571` is historical evidence from 2026-09-07,
not a default or a claim about today's service.

The configured release is current repository configuration, not the installed
or loaded release and not necessarily the value used by that workflow. Only
the conventional `data-YYYYMMDD` identifier is emitted; other/unreadable values
remain unknown. The deploy script can pull a newer `main` than its triggering
run, so even a completed job's head SHA does not prove running bytes.

Deployment evidence does **not** change the retrieval exit code. A report can
have usable retrievals and a skipped deploy. Neither is production-deployment
acceptance. Consumers needing deployment proof must inspect the separate
deployment status and still obtain observed running app/data identifiers.
Without `--deployment-run`, no GitHub command runs and deployment evidence is
`NOT_REQUESTED`.
