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
  correct_dir: string | null
  selected_ranks?: number[] | null
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
