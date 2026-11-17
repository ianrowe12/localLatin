import { apiFetch } from './client'

export type QueueFilter = 'all' | 'conflict' | 'needs_check' | 'with_notes' | 'reviewed_once' | 'decided'

export const QUEUE_FILTERS: { value: QueueFilter; label: string }[] = [
  { value: 'conflict', label: 'Disagreements' },
  { value: 'needs_check', label: 'Needs check' },
  { value: 'with_notes', label: 'With notes' },
  { value: 'reviewed_once', label: 'Reviewed once' },
  { value: 'decided', label: 'Decided' },
  { value: 'all', label: 'All reviewed' },
]

export const REVIEW_STATUS_LABELS: Record<string, string> = {
  conflict: 'Disagreement',
  needs_check: 'Needs check',
  agreed: 'Agreed',
  no_key: 'No key',
  unreviewed: 'Unreviewed',
}

export interface CountedAnswer {
  reviewer: string
  model_slug: string
  variant: string | null
  keys: string[]
  none_of_these: boolean
  unresolved: boolean
  notes: string
  timestamp: string
}

export interface DecidingAnswer {
  id: number
  xml_id: string
  timestamp: string
  reviewer_account_id: number
  reviewer: string
  ccl_key: string | null
  notes: string
}

export interface AdminReviewItem {
  xml_id: string
  query_id: number
  status: string
  reviewer_status: string
  reviewer_count: number
  answers: CountedAnswer[]
  deciding_answer: DecidingAnswer | null
}

export interface AdminReviewQueue {
  items: AdminReviewItem[]
  counts: Record<QueueFilter, number>
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export interface DecidingAnswerPayload {
  ccl_key: string | null
  none_of_these: boolean
  notes: string
}

export function fetchReviewQueue(filter: QueueFilter, page: number): Promise<AdminReviewQueue> {
  return apiFetch(`/api/feedback/review_queue?${new URLSearchParams({
    filter, page: String(page), page_size: '20',
  })}`)
}

export function fetchAdminReviewItem(queryId: number): Promise<AdminReviewItem> {
  return apiFetch(`/api/feedback/review_queue/${queryId}`)
}

export function saveDecidingAnswer(queryId: number, body: DecidingAnswerPayload): Promise<DecidingAnswer> {
  return apiFetch(`/api/feedback/decisions/${queryId}`, {
    method: 'POST', body: JSON.stringify(body),
  })
}
