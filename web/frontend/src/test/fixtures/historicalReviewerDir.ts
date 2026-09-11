import type { ReviewerDir } from '../../api/reviewerDirs'

/**
 * A reviewer directory that the server preserves without its seed membership
 * (issue #161, review finding 11).
 *
 * NOT hand-written. This is the row an independent reviewer produced by running
 * the pre-#160 `create_reviewer_dir` against a disposable FeedbackDB, pausing it
 * between its `reviewer_dirs` INSERT and its membership INSERT, letting an
 * ordinary authenticated request commit the shared connection in between, and
 * then failing the creation. The directory is permanent; the membership row
 * never existed. Reopening that database through the current `create_app` and
 * signing in, `GET /api/reviewer_dirs?seed_query_id=7` and
 * `GET /api/query/7/predictions` both answer 200 with exactly this object, and
 * `POST /api/reviewer_dirs` for the same seed answers 409 with exactly
 * `HISTORICAL_CONFLICT_DETAIL`.
 *
 * Issue #160 made creation atomic, so no new row can look like this. Nothing in
 * the application can repair or remove the ones already written, which is why
 * the client has to be able to read them: calling this unreadable hides a
 * grouping that exists and refuses the reviewer any way past it, since creating
 * a second one is refused too.
 */
export const HISTORICAL_SEEDLESS_DIR: ReviewerDir = Object.freeze({
  dir_id: 'reviewer-dir-f12cec1516c0',
  label: 'Historical partial grouping',
  status: 'awaiting_match',
  seed_query_id: 7,
  member_query_ids: [],
  created_at: '2026-09-11 01:52:29',
  created_by: 'Historical reviewer',
  model_slug: 'bowphs_LaTa',
  variant: 'sif_abtt',
  best_match_score: 0.0,
  has_potential_match: false,
})

/** The query that seeds it, as the captured database recorded it. */
export const HISTORICAL_SEED_QUERY_ID = 7

/** The 409 body the current server returns for a second create on that seed. */
export const HISTORICAL_CONFLICT_DETAIL =
  "Query 7 already seeds reviewer directory 'reviewer-dir-f12cec1516c0' (Historical partial grouping)."
