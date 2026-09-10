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
}

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
  const created = assertReviewerDir(body, payload.query_file_id)
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
export function assertReviewerDir(entry: unknown, seedQueryId?: number): ReviewerDir {
  const dir = entry as ReviewerDir | null
  const wellFormed =
    dir !== null &&
    typeof dir === 'object' &&
    typeof dir.dir_id === 'string' &&
    dir.dir_id.length > 0 &&
    typeof dir.label === 'string' &&
    typeof dir.seed_query_id === 'number' &&
    typeof dir.created_at === 'string' &&
    typeof dir.created_by === 'string' &&
    DIR_STATUSES.includes(dir.status) &&
    Array.isArray(dir.member_query_ids) &&
    dir.member_query_ids.every((id) => typeof id === 'number') &&
    (dir.best_match_score === null || typeof dir.best_match_score === 'number') &&
    typeof dir.has_potential_match === 'boolean'
  if (!wellFormed) {
    throw new Error(
      'A reviewer directory could not be read: the server sent an incomplete record.',
    )
  }
  if (seedQueryId !== undefined && dir.seed_query_id !== seedQueryId) {
    throw new Error(
      `A reviewer directory could not be read: the server answered for document ${seedQueryId} with a record seeded by ${dir.seed_query_id}.`,
    )
  }
  return dir
}
