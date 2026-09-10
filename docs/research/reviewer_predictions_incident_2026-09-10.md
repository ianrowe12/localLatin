# Missing reviewer predictions: incident handoff, 2026-09-10

**Blocked; live root cause unknown.** This draft supports [#155][incident] under
[#153][parent]; it does not resolve either issue. Abigail reported that only
mT5-base worked. The reported screenshot shows PhilTa selected for
`BAV1341.22v.9.txt`, query text present, a blank candidate panel and "No predictions
available". No current authenticated incident response has been captured.

This handoff used source, issue/comments and GitHub workflow metadata only.
No production requests, sign-ins, service actions or reviewer-record changes
were made. The independently identified safety defects in #156, #157 and #158
do not establish this incident's cause.

## Verified history and limits

| Evidence | What it establishes |
| --- | --- |
| [Deployment run 34079707571][run], head `1ccf4fb62bef0287071a730c1b54a7c8feca7551` | The actual **Deploy production** job succeeded at `2026-09-07T03:31:25Z`. This is historical deployment evidence, not today's service health or loaded source/data identity. |
| Local Git object comparison with investigation base `5b058e310aae6c677185ce20c18ae6acb9378402` | Both commits have the same `web/` tree, `2a6409bc828791f94356cb805af4d76fd922a887`; `deploy/deploy.sh` and the deployment workflow also have no diff between them. |
| [Recorded investigation][incident] and [access update][access] | The prior research-data/backend replay mapped the filename to local query 59 and returned ten ranks with nonblank candidate-file placeholders backed by existing files for all six models. Target q-q rows were finite and not excluded. This worker did not repeat that replay; placeholders and file existence do not prove production candidate-text delivery. The reported unauthenticated 401 was an access blocker, not reproduction of Abigail's failure. |

The [deployment workflow][workflow] can skip deployment despite a green run,
and pulls `main` rather than pinning the run's head SHA. Neither a run SHA nor a
local tree comparison identifies the currently running bytes. The
[data installer][installer] can retain existing disk data when no release is
configured. Its installed-state digest hashes checksum lines, not the archive
itself. A configured release tag or disk marker alone does not identify data
already loaded into the service. The inspected [smoke utility][smoke] selects
only the first model/query for prediction checks, not all six incident paths.

## Authorized next capture

The [owner's access requirements][access] remain unmet: an owner-operated
existing app session, plus an explicitly approved service-user shell target or
a reviewed diagnostic-only workflow with separate approval for later dispatch
if service logs are needed. Runner availability is not access approval.
Do not guess SSH targets or credentials.

1. In that existing session, resolve `BAV1341.22v.9.txt` through the running
   service's query search. Require one exact filename match across all pages;
   [search is a paginated substring match][queries]. **Never hardcode production
   query 59.** Record the resolved ID and capture time in UTC.
2. Compare PhilTa and mT5-base for that same ID through the public-app path,
   explicitly using `sif_abtt` and `top_k=10`. Check response identity, actual
   model ranks, finite scores and nonblank candidate-text availability at each
   rank, not just HTTP success or the catalog's CSV-row `prediction_count`.
   Reviewer-directory extras must not substitute for missing model evidence.
3. If a request fails, obtain only its short, matching service-log window
   through the separately approved path. Keep sanitized exception type and
   source frames, not raw log lines or exception messages. If responses warrant
   a source/data comparison, distinguish observed running identity, on-disk
   identity, configured labels and historical workflow metadata. Mark anything
   unobserved as unknown.
4. Complete the same incident-file comparison for all six expected web models:
   LaTa, PhilTa, LaBSE, mT5-base, KaLM-mini and Qwen3-0.6B. Assert this set
   independently of the returned catalog. Use [#159][diagnostic] and its
   [proposed read-only diagnostic guide][guide] for the pinned ordinary control
   manuscript and diagnostic procedure. That guide is pending #159, not present
   at this handoff's base; this document does not implement or authorize it.

Persist only allowlisted fields: UTC window, capture scope, approved query
filename and resolved ID, requested/echoed model and variant, identity-match
booleans, HTTP status, sanitized error category, explicit source status, model
rank IDs/counts, finite-score booleans, per-rank candidate-file/nonblank-text
counts, and provenance identifiers with their evidence source. Keep response
bodies private. Do not attach raw HAR, previews, manuscript bodies, reviewer
notes or labels, attribution, cookies, credentials, headers or session URLs.

Existing-session GETs [update `last_seen_at`][session]; the owner permits this normal
bookkeeping, not new sessions or reviewer-record writes. Do not sign in,
dispatch existing workflows, restart services, regenerate artifacts or run
`smoke_reviewer_pilot.py`: it creates and approves/rejects accounts even without
`--write-check`. Do not start another app or call `FeedbackDB.connect` against
production for a snapshot; its startup runs schema/migration writes. Any
separately approved read-only snapshot remains distinct from public-app evidence.

## Classification and outstanding acceptance

The [source export][export] distinguishes `excluded_blank_source` from
`excluded_zero_norm`. These are explicit, non-evaluable source exclusions, not
request failures or human non-matches. At the inspected base the
[loader][loader] drops that status; #156 owns its additive transport to the API.
An empty legacy response has **unknown cause**, not an inferred exclusion.
Auth failures mean BLOCKED access; 5xx, network/malformed responses and
unexplained empty success remain technical failures. Even an explicit exclusion
cannot count as successful retrieval for the ordinary incident manuscript.
Never record unavailable evidence as `none_of_top_k`.

All incident acceptance remains outstanding:

- Record the exact production filename/ID mapping and failing authenticated
  response, without sensitive content.
- Identify the actual defect and reproduce it in a regression before fixing it.
- Demonstrate usable rankings and candidate text for the incident manuscript
  under all six models with `sif_abtt` in an authorized public-app session.
- Distinguish intentional exclusions from request errors and human non-matches.
- Complete diagnosis without creating reviewer accounts, directories or feedback.

Keep the PR draft and reference #155 without a closing keyword. A document,
source replay, future #159 result or frontend safety fix alone cannot satisfy
the missing live capture, root-cause regression and verified incident resolution.

[incident]: https://github.com/ianrowe12/localLatin/issues/155
[parent]: https://github.com/ianrowe12/localLatin/issues/153
[access]: https://github.com/ianrowe12/localLatin/issues/155#issuecomment-5624827631
[run]: https://github.com/ianrowe12/localLatin/actions/runs/34079707571/job/101612936016
[workflow]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/.github/workflows/deploy.yml#L55-L107
[installer]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/deploy/deploy.sh#L60-L146
[queries]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/web/routers/queries.py#L21-L75
[diagnostic]: https://github.com/ianrowe12/localLatin/issues/159
[guide]: ../../deploy/READ_ONLY_REVIEWER_READINESS.md
[export]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/scripts/resubmit/run_resubmit_unlabelled_retrieval.py#L120-L134
[loader]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/web/services/data_store.py#L194-L211
[smoke]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/scripts/webapp/smoke_reviewer_pilot.py#L626-L729
[session]: https://github.com/ianrowe12/localLatin/blob/5b058e310aae6c677185ce20c18ae6acb9378402/web/services/feedback_db.py#L824-L850
