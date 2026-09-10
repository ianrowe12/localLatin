# Feedback write contract

`POST /api/feedback` appends an assessment for a real query, served model and
variant. The deployment's default variant still applies when omitted.

- `matched_rank` and `none_of_top_k` require at least one usable model candidate.
  A usable candidate has a directory identity, a finite score and at least one
  nonblank candidate text. Every selected candidate must also be usable.
  An unusable candidate does not block readable positive choices.
- `none_of_top_k` also requires every offered model candidate to be usable.
  A partial ranking with missing or blank candidate text cannot support None.
  Reviewer extras neither replace model evidence nor make None unavailable when
  all offered model candidates are usable. Confidence thresholds do not gate
  evaluation.
- A source status beginning with `excluded` rejects evaluation even if candidates
  remain in that row. Missing source status on an older artifact does not reject
  an otherwise usable ranking.
- `skipped` requires a nonblank note, trimmed before storage. It does not require
  a query ranking, but unknown queries, models and unavailable variants still
  reject the request. Skip retains the existing query-level workflow effect; it
  is not a negative evaluation.
- New null outcomes, implicit note-only saves and `legacy_unresolved` writes are
  rejected. Older clients may omit `outcome` when they send a deliberate rank,
  `correct_rank: 0`, or `selected_ranks`. Rank values must be JSON integers, not
  booleans, strings or fractions.

## Directory identity precondition

New clients send `expected_candidate_dirs`, an object mapping every selected rank
to the directory identity displayed when that choice was made:

```json
{
  "query_id": 1,
  "model_slug": "bowphs_LaTa",
  "variant": "sif_abtt",
  "outcome": "matched_rank",
  "selected_ranks": [12, 1],
  "expected_candidate_dirs": {
    "12": "reviewer-dir-example",
    "1": "candidate-a"
  }
}
```

The Python type is `dict[int, str] | None`, with keys restricted to ranks 1-15.
JSON object keys are strings. If supplied, the field must be non-null, cover
exactly the selected ranks and contain nonblank identities. For a single choice,
it covers `correct_rank`. Omit it for None and Skip.

The server resolves all choices against one candidate snapshot, using the same
ranking assembly as the predictions route. It compares every expected identity
before appending feedback or membership. On a usable model ranking, a changed or
vanished choice returns:

```json
{
  "error": {
    "code": "CANDIDATE_IDENTITY_CHANGED",
    "message": "Candidate at rank 12 has changed. Refresh the ranking and review your selections before saving."
  }
}
```

That response has HTTP status 409. Keep the draft and request a fresh ranking;
do not silently accept the new directory at that rank or blindly retry.

Source-level ineligibility returns HTTP 422 with the same error envelope and code
`RANKING_NOT_EVALUABLE`, including None on a partially usable model ranking.
An unusable selected candidate returns
`CANDIDATE_NOT_EVALUABLE`. Invalid request fields return FastAPI's usual 422
`detail` list. A missing selected rank without a precondition retains a 422
string `detail`. A missing query prediction row returns 404. Source-level
eligibility checks run before identity comparisons.

## Preserved behavior and limits

The first selected rank remains canonical, even if it is numerically larger.
The server ignores client `correct_dir` as assignment authority. It records all
`selected_ranks`, but adds membership only for the canonical reviewer directory.
Ranks 11-15 remain anchored independently of the model candidate count.

Historical records, migrations and reads are unchanged. Latest feedback still
combines the caller's own decision with the newest shared nonblank note and its
original attribution.

Older clients that omit the precondition still resolve ranks at save time and
can therefore save a directory different from the one they saw. The precondition
does not prove a reviewer saw evidence, bind text or score revisions, provide
exactly-once delivery, or make feedback and membership inserts transactional.
Repeated accepted writes still append feedback. No feedback UPDATE or DELETE is
introduced.
