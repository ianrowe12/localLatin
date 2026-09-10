import { apiFetch, apiUrl } from './client'
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

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await apiFetch<void>('/api/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
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
