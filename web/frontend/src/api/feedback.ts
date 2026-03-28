import { apiFetch } from './client'

export interface FeedbackPayload {
  query_id: number
  model_slug: string
  correct_rank: number | null
  correct_dir: string | null
  notes: string
  reviewer: string
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
