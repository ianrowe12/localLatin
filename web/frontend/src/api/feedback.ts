import { apiFetch } from './client'

export const FEEDBACK_UPDATED_EVENT = 'locallatin:feedback-updated'

export type FeedbackOutcome = 'matched_rank' | 'none_of_top_k' | 'skipped'

export interface FeedbackPayload {
  query_id: number
  model_slug: string
  outcome?: FeedbackOutcome
  correct_rank: number | null
  correct_dir: string | null
  notes: string
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  await apiFetch<void>('/api/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function exportFeedbackCsv(): void {
  window.open('/api/feedback/export', '_blank')
}
