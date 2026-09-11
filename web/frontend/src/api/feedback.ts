import { ApiError, apiFetch, apiUrl } from './client'
import { DEFAULT_VARIANT, type PredictionVariant } from './variants'

export const FEEDBACK_UPDATED_EVENT = 'locallatin:feedback-updated'

export type FeedbackOutcome =
  | 'matched_rank'
  | 'none_of_top_k'
  | 'skipped'
  | 'legacy_unresolved'

export interface FeedbackPayload {
  query_id: number
  model_slug: string
  variant?: PredictionVariant
  outcome?: FeedbackOutcome
  correct_rank: number | null
  /**
   * Never an assignment authority: `web/routers/feedback.py` resolves the
   * directory from the rank against its own snapshot and ignores this field.
   * New saves send null and let `expected_candidate_dirs` carry what the
   * reviewer actually saw.
   */
  correct_dir: string | null
  selected_ranks?: number[] | null
  /**
   * The directory displayed at every selected rank when the choice was made,
   * keyed by rank (issue #157). The server compares each one against its own
   * resolution and answers 409 `CANDIDATE_IDENTITY_CHANGED` before appending
   * anything if one has moved, so a reviewer-created directory that shifted
   * rank cannot silently receive the assignment.
   *
   * Required for every selected rank when sent at all, and only sent with
   * `matched_rank`. It is an identity precondition, not proof that a reviewer
   * saw the evidence, and not a ranking revision token.
   */
  expected_candidate_dirs?: Record<string, string>
  notes: string
}

// Mirrors backend models.py FeedbackEntry.
export interface FeedbackEntry {
  id: number
  query_id: number
  timestamp: string
  model_slug: string
  variant: PredictionVariant | null
  outcome: FeedbackOutcome
  correct_rank: number | null
  correct_dir: string | null
  selected_ranks: number[] | null
  notes: string
  reviewer: string
  reviewer_account_id: number | null
  /** Login name of the author; null for pre-account or deleted-account rows. */
  reviewer_username: string | null
  schema_version: number
}

/** Slug shape the API answers in, so an HF id and its slug compare equal. */
function normalizeSlug(slug: string): string {
  return slug.replace(/\//g, '_')
}

function isFeedbackOutcome(value: unknown): value is FeedbackOutcome {
  return (
    value === 'matched_rank' ||
    value === 'none_of_top_k' ||
    value === 'skipped' ||
    value === 'legacy_unresolved'
  )
}

/**
 * The saved row, if the answer really is one.
 *
 * `POST /api/feedback` answers 201 with the appended `FeedbackEntry`. That row
 * is the only receipt this client ever gets, so it is checked rather than cast:
 * an answer from a proxy, a cached page, a rewritten route or a future server
 * that stops returning the row is not evidence that anything was written, and
 * discarding a reviewer's draft on the strength of it would destroy unsent work
 * for a save that may never have happened (issue #158).
 *
 * Returns null for anything that is not a readable row.
 */
export function readFeedbackEntry(value: unknown): FeedbackEntry | null {
  if (typeof value !== 'object' || value === null) return null
  const record = value as Record<string, unknown>
  if (typeof record.id !== 'number' || !Number.isInteger(record.id)) return null
  if (typeof record.query_id !== 'number' || !Number.isInteger(record.query_id)) {
    return null
  }
  if (typeof record.model_slug !== 'string' || record.model_slug === '') return null
  if (!isFeedbackOutcome(record.outcome)) return null
  return record as unknown as FeedbackEntry
}

/**
 * Save an assessment and return the appended row.
 *
 * Rejects with an `ApiError` when the request fails, and also when a 2xx body
 * is not the row that was asked for. `kind: 'malformed'` is deliberate: the
 * request may well have been committed, so `classifySaveFailure` reads this as
 * uncertain, keeps the draft and warns instead of promising an empty log.
 */
export async function submitFeedback(payload: FeedbackPayload): Promise<FeedbackEntry> {
  const body = await apiFetch<unknown>('/api/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  const entry = readFeedbackEntry(body)
  if (entry === null) {
    throw new ApiError({
      kind: 'malformed',
      status: null,
      code: 'FEEDBACK_RECEIPT_UNREADABLE',
      message: 'The server did not return a readable record of this assessment.',
    })
  }
  if (
    entry.query_id !== payload.query_id ||
    normalizeSlug(entry.model_slug) !== normalizeSlug(payload.model_slug) ||
    (payload.outcome !== undefined && entry.outcome !== payload.outcome)
  ) {
    throw new ApiError({
      kind: 'malformed',
      status: null,
      code: 'FEEDBACK_RECEIPT_MISMATCHED',
      message:
        'The server returned a record for a different assessment, so this save could not be confirmed.',
    })
  }
  return entry
}

// What to prefill for this query/model/variant, or null if nobody has reviewed
// it. Not a verbatim DB row: the server merges the team's newest non-empty NOTE
// (issue #96 -- hence `reviewer_username` / `reviewer`, which often name
// somebody else) with the CALLER's own newest decision, so `correct_rank` /
// `selected_ranks` are never another reviewer's answer.
export async function fetchLatestFeedback(
  queryId: number,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): Promise<FeedbackEntry | null> {
  const params = new URLSearchParams({
    query_id: String(queryId),
    model,
    variant,
  })
  return apiFetch<FeedbackEntry | null>(`/api/feedback/latest?${params.toString()}`)
}

export function exportFeedbackCsv(): void {
  window.open(apiUrl('/api/feedback/export'), '_blank')
}
