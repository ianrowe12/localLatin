import { apiFetch } from './client'
import type { PredictionVariant } from './variants'

/**
 * Reviewer-created directories (issue #95).
 *
 * A reviewer looking at a query that matches nothing in the labelled corpus
 * can declare a new directory seeded by that document. From then on the
 * directory is scored against every other query from the query-query cosine
 * matrix and appears in their candidate lists like any other option.
 *
 * Issue #94 shipped an earlier version of this client that feature-detected the
 * endpoint (404/405/501 -> a quiet "coming with the next update" state),
 * because the CTA landed before the backend did. This PR *is* that backend, so
 * the endpoint is never absent from a deployment that has this file and the
 * degradation branch is dead code; it is deleted rather than left to rot into a
 * state nothing can reach. Ordinary `apiFetch` error handling covers the
 * failures that remain real (401, 409, 422, 429).
 */

/**
 * `matched` means a reviewer filed a second document into the directory.
 * Similarity alone never produces it.
 */
export type ReviewerDirStatus = 'awaiting_match' | 'matched'

// Mirrors backend models.py ReviewerDir.
export interface ReviewerDir {
  dir_id: string
  label: string
  status: ReviewerDirStatus
  seed_query_id: number
  member_query_ids: number[]
  created_at: string
  created_by: string
  model_slug: string
  variant: PredictionVariant | null
  // Informational only. Status is decided by human confirmation, never by this
  // number -- see web/services/reviewer_dirs.py.
  best_match_score: number | null
  // best_match_score crosses the no-match band: the model sees something
  // related that nobody has confirmed yet.
  has_potential_match: boolean
  /**
   * CLIENT-SIDE PROVENANCE, never a wire field (issue #161).
   *
   * A directory can reach this app two ways. A directory response is parsed
   * from raw JSON here, so every value in it is something the server said. A
   * directory inside a ranking is parsed by the shared prediction validator
   * (issue #156), which tolerates an older backend omitting
   * `member_query_ids`, `created_at`, `created_by` or `model_slug` and fills in
   * `[]` and `''` on its behalf. Those substitutes are type-valid and therefore
   * invisible: `[]` from a server that said "no members" and `[]` from a field
   * that was never on the wire are the same value.
   *
   * That distinction matters exactly once, at the durable boundary, where the
   * difference is between a stored fact and a guess. So the normalizer names
   * the fields it supplied itself, and `isCompleteReviewerDir` declines the
   * row rather than the whole distinction being lost. Present only when
   * something really was defaulted, so a complete row is byte-identical to the
   * response it came from.
   */
  defaulted_fields?: readonly ReviewerDirOptionalField[]
}

/**
 * The fields `web/models.py` gives defaults, and which an older backend may
 * therefore omit. Named here because both the normalizer that substitutes them
 * and the boundary that refuses the substitutes need the same list.
 */
export type ReviewerDirOptionalField =
  | 'member_query_ids'
  | 'created_at'
  | 'created_by'
  | 'model_slug'

export interface CreateReviewerDirPayload {
  query_file_id: number
  label?: string
  model_slug?: string
  variant?: PredictionVariant
}

/**
 * Broadcast after a directory is created, so cached prediction responses are
 * dropped: the new directory is a candidate for every other query, and a stale
 * cache would keep it off screen until a reload.
 */
export const REVIEWER_DIRS_UPDATED_EVENT = 'locallatin:reviewer-dirs-updated'

/** Fire the refresh broadcast on its own, see `CreateReviewerDirOptions.notify`. */
export function notifyReviewerDirsUpdated(): void {
  window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
}

export interface CreateReviewerDirOptions {
  /**
   * Broadcast REVIEWER_DIRS_UPDATED_EVENT as soon as the POST resolves
   * (default). The refresh it triggers unmounts whatever rendered the call, so
   * a caller that has to *record* the saved directory first passes `false` and
   * calls `notifyReviewerDirsUpdated()` once the identity is committed. Issue
   * #161: a refresh that then fails must not be able to lose the only record
   * of a permanent write.
   */
  notify?: boolean
}

export async function createReviewerDir(
  payload: CreateReviewerDirPayload,
  options: CreateReviewerDirOptions = {},
): Promise<ReviewerDir> {
  const body = await apiFetch<unknown>('/api/reviewer_dirs', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  // Checked for the same reason the list is: this row becomes the
  // acknowledgement a reviewer reads. An unreadable 201 is treated as a failed
  // write, which sends the caller to the seed-filtered lookup and recovers the
  // real row instead of displaying a half-built one.
  //
  // This is the ONE place the atomic-creation guarantee applies. Issue #160
  // made creation insert the directory and its seed membership in a single
  // dedicated transaction, so a directory this request just created reports
  // its own seed as a member. A 201 that does not is the half-write #160
  // fixed, and the right answer is to treat the write as unconfirmed and
  // reconcile it against the database -- not to acknowledge a row whose
  // membership nobody can vouch for. The same demand must NOT be made of rows
  // that are merely being read back: see `isCompleteReviewerDir`.
  const created = assertReviewerDir(body, payload.query_file_id, {
    requireSeedMembership: true,
  })
  if (options.notify !== false) notifyReviewerDirsUpdated()
  return created
}

export interface ReviewerDirQuery {
  /**
   * Model slug whose q-q matrix supplies `best_match_score`. Optional, and
   * deliberately so for identity lookups: which directories exist is a
   * property of the database, not of the model a reviewer happens to have
   * selected, and the server falls back to its first served model.
   */
  model?: string
  /**
   * Restrict the answer to directories seeded by this query
   * (`GET /api/reviewer_dirs?seed_query_id=N`).
   *
   * This is the reload/recovery path for issue #161: the seed of a directory
   * is the query that created it, so one authenticated GET answers "does this
   * document already have a grouping?" after a full page reload, after a
   * failed prediction refresh, or when a create response was lost in flight.
   * It returns EVERY directory recorded for that seed, including historical
   * duplicates, rather than picking one and hiding the rest.
   */
  seedQueryId?: number
  signal?: AbortSignal
}

export async function fetchReviewerDirs(
  query: ReviewerDirQuery = {},
): Promise<ReviewerDir[]> {
  const params = new URLSearchParams()
  if (query.model) params.set('model', query.model)
  if (query.seedQueryId !== undefined) {
    params.set('seed_query_id', String(query.seedQueryId))
  }
  const search = params.toString()
  const body = await apiFetch<unknown>(
    `/api/reviewer_dirs${search ? `?${search}` : ''}`,
    query.signal ? { signal: query.signal } : undefined,
  )
  return assertReviewerDirList(body, query.seedQueryId)
}

/**
 * A 200 this client cannot read is a FAILURE, not an empty answer (issue #161).
 *
 * The one question this endpoint answers is "does this document already have a
 * directory?", and the two answers are treated very differently: an empty list
 * is permission to create a permanent, unremovable record. So a body that does
 * not parse -- a proxy's HTML error page, a truncated response, a future shape
 * -- must not be rounded down to "no". Throwing puts it on the failure path,
 * where the store leaves the question unresolved and the reviewer is offered a
 * retry rather than a Create button.
 *
 * The check covers EVERY field the acknowledgement actually consumes, not just
 * the ones needed to keep TypeScript quiet. A row with no `created_by` renders
 * "Created by undefined"; a row with no `status` or `member_query_ids` silently
 * downgrades a matched grouping (a human filed a second witness into it) to an
 * awaiting one; a row with no `created_at` breaks the oldest-first ordering
 * that decides which of several historical groups is named. Half a row is not
 * evidence that a permanent record exists in a particular shape.
 *
 * `seedQueryId`, when the request was filtered, is checked too: a row for
 * another document would mark this one saved and suppress its creation
 * controls. A mismatch is a broken response, so it is rejected whole rather
 * than filtered away -- filtering it would turn a server fault into "no
 * directory here", which is the permission this function exists to withhold.
 */
export function assertReviewerDirList(
  body: unknown,
  seedQueryId?: number,
): ReviewerDir[] {
  if (!Array.isArray(body)) {
    throw new Error(
      'The reviewer directory list could not be read: the server sent an unexpected response.',
    )
  }
  for (const entry of body) {
    assertReviewerDir(entry, seedQueryId)
  }
  return body as ReviewerDir[]
}

const DIR_STATUSES: readonly string[] = ['awaiting_match', 'matched']

/**
 * One directory, checked field by field. Also used for the 201 body: a create
 * response that cannot be read is treated as a failed write and reconciled
 * against the database, which recovers the row rather than inventing one.
 */
export function assertReviewerDir(
  entry: unknown,
  seedQueryId?: number,
  options: { requireSeedMembership?: boolean } = {},
): ReviewerDir {
  // Seed identity is checked FIRST, so a row for another document is reported
  // as that rather than as a generic unreadable one. The two faults want
  // different diagnostics: one names a server that answered about the wrong
  // document, the other a record this client cannot read at all.
  const seed = (entry as ReviewerDir | null)?.seed_query_id
  if (seedQueryId !== undefined && typeof seed === 'number' && seed !== seedQueryId) {
    throw new Error(
      `A reviewer directory could not be read: the server answered for document ${seedQueryId} with a record seeded by ${seed}.`,
    )
  }
  if (!isCompleteReviewerDir(entry)) {
    throw new Error(
      'A reviewer directory could not be read: the server sent an incomplete record.',
    )
  }
  if (seedQueryId !== undefined && entry.seed_query_id !== seedQueryId) {
    throw new Error(
      `A reviewer directory could not be read: the server answered for document ${seedQueryId} with a record seeded by ${entry.seed_query_id}.`,
    )
  }
  if (
    options.requireSeedMembership &&
    !entry.member_query_ids.includes(entry.seed_query_id)
  ) {
    throw new Error(
      'The new directory could not be confirmed: the server reported it without the document that seeds it.',
    )
  }
  return entry
}

/**
 * The completeness half of the check above, as a question rather than a throw.
 *
 * There are two readers of a directory row and they need the same answer in
 * different shapes. A response body is validated once, at the edge, where
 * throwing is right because the alternative is treating an unreadable answer as
 * "no directory here". Evidence arriving through an already-parsed value --
 * `PredictionResponse.seeded_dirs` -- is judged by the durable record, which
 * cannot throw at a reviewer mid-render and must simply decline to treat a
 * half-row as proof.
 *
 * Sharing this predicate is what keeps those two answers the same.
 *
 * WHAT THIS ASKS is whether every field the acknowledgement consumes was
 * really said by the server. It deliberately does NOT ask whether the
 * directory looks like one today's backend would create. Those are different
 * questions, and conflating them rejected real data: before issue #160,
 * creation wrote the directory and its seed membership on a connection shared
 * with everything else the request touched, so an unrelated commit in between
 * -- the session's `last_seen_at`, for one -- could make the directory
 * permanent while the membership insert was still to come. A cancelled or
 * failed creation after that point left a directory with no members at all,
 * and nothing in this application can remove it. The current server preserves
 * those rows, serves them on both the seed-filtered list and the ranking, and
 * refuses a second directory for the same seed with a 409. A client that calls
 * them unreadable hides a grouping that demonstrably exists and cannot offer
 * any way to move past it. The atomic guarantee is asserted where it is
 * actually promised: on the 201 of a write this client just issued.
 */
export function isCompleteReviewerDir(entry: unknown): entry is ReviewerDir {
  const dir = entry as ReviewerDir | null
  return (
    dir !== null &&
    typeof dir === 'object' &&
    !(dir.defaulted_fields && dir.defaulted_fields.length > 0) &&
    // Nothing here was filled in on the server's behalf. `[]` and `''` are
    // legitimate stored values AND are what a permissive parser writes where a
    // field was missing, so the two are indistinguishable by inspection: an
    // empty member list is a real historical record, and a substituted one is
    // a guess that would silently downgrade a matched grouping. The shared
    // prediction validator (issue #156) names what it supplied instead of
    // leaving this boundary to tell them apart, which it cannot.
    typeof dir.dir_id === 'string' &&
    dir.dir_id.length > 0 &&
    typeof dir.label === 'string' &&
    typeof dir.seed_query_id === 'number' &&
    // Non-EMPTY, not merely present. The acknowledgement renders both of these
    // at the reviewer -- a blank creator reads as "Created by ." -- and the
    // timestamp also orders the groups, deciding which of several is named as
    // the one this document is filed under.
    typeof dir.created_at === 'string' &&
    dir.created_at.length > 0 &&
    typeof dir.created_by === 'string' &&
    dir.created_by.length > 0 &&
    DIR_STATUSES.includes(dir.status) &&
    Array.isArray(dir.member_query_ids) &&
    dir.member_query_ids.every((id) => typeof id === 'number') &&
    (dir.best_match_score === null || typeof dir.best_match_score === 'number') &&
    typeof dir.has_potential_match === 'boolean'
  )
}

/**
 * Does the server's record of this directory list the document that seeds it?
 *
 * Normally yes, and since issue #160 always yes for anything newly created.
 * Where it is false the row is a preserved partial write (see above): the
 * grouping is stored and permanent, but the membership row that would make
 * this document a member of it never landed and nothing can add it now. The
 * acknowledgement says so rather than quietly implying the document is filed
 * there, because membership is what `matched` is derived from and what a
 * reviewer would reasonably read "seeded with this document" to mean.
 */
export function listsItsSeedAsMember(dir: ReviewerDir): boolean {
  return dir.member_query_ids.includes(dir.seed_query_id)
}
