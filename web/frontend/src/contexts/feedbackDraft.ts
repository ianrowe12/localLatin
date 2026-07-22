import type { FeedbackDraft } from './FeedbackContext'

export function isFeedbackDraftEmpty(draft: FeedbackDraft | undefined): boolean {
  if (!draft) return true
  return (
    draft.correctRank === null &&
    (!draft.selectedRanks || draft.selectedRanks.length === 0) &&
    draft.notes.trim() === ''
  )
}

export function seedDraftMapIfEmpty(
  drafts: Map<string, FeedbackDraft>,
  key: string,
  seed: FeedbackDraft,
): Map<string, FeedbackDraft> {
  if (!isFeedbackDraftEmpty(drafts.get(key))) return drafts
  const next = new Map(drafts)
  next.set(key, seed)
  return next
}
