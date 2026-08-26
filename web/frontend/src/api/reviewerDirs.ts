import { apiFetch } from './client'
import type { PredictionVariant } from './variants'

/**
 * Reviewer-created directories (issue #95).
 *
 * A reviewer looking at a query that matches nothing in the labelled corpus
 * can declare a new directory seeded by that document. From then on the
 * directory is scored against every other query from the query-query cosine
 * matrix and appears in their candidate lists like any other option.
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
  best_match_score: number | null
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

export async function createReviewerDir(
  payload: CreateReviewerDirPayload,
): Promise<ReviewerDir> {
  const created = await apiFetch<ReviewerDir>('/api/reviewer_dirs', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
  return created
}

export async function fetchReviewerDirs(model?: string): Promise<ReviewerDir[]> {
  const params = new URLSearchParams()
  if (model) params.set('model', model)
  const query = params.toString()
  return apiFetch<ReviewerDir[]>(`/api/reviewer_dirs${query ? `?${query}` : ''}`)
}
