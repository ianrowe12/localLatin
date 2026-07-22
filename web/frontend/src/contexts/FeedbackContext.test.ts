import { describe, expect, it } from 'vitest'
import type { FeedbackDraft } from './FeedbackContext'
import { seedDraftMapIfEmpty } from './feedbackDraft'

const key = '0-KaLM'
const saved: FeedbackDraft = {
  correctRank: 1,
  selectedRanks: [1, 3],
  notes: 'saved server note',
}

describe('seedDraftMapIfEmpty', () => {
  it('hydrates saved feedback over a semantically empty persisted draft', () => {
    const emptyPersistedDraft: FeedbackDraft = {
      correctRank: null,
      notes: '   ',
    }
    const drafts = new Map([[key, emptyPersistedDraft]])

    const result = seedDraftMapIfEmpty(drafts, key, saved)

    expect(result.get(key)).toEqual(saved)
  })

  it('preserves a real unsaved local edit', () => {
    const localEdit: FeedbackDraft = {
      correctRank: 2,
      notes: 'not submitted yet',
    }
    const drafts = new Map([[key, localEdit]])

    const result = seedDraftMapIfEmpty(drafts, key, saved)

    expect(result).toBe(drafts)
    expect(result.get(key)).toEqual(localEdit)
  })
})
