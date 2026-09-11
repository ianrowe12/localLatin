import { describe, expect, it } from 'vitest'
import type { FeedbackDraft } from './FeedbackContext'
import {
  deserializeDrafts,
  draftSelections,
  draftWithNone,
  draftWithSelections,
  legacyDraftKey,
  makeDraftKey,
  sanitizeDraft,
  seedDraftMapIfEmpty,
  serializeDrafts,
} from './feedbackDraft'

const key = makeDraftKey(0, 'KaLM', 'sif_abtt', 5)
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

  it('treats a selection-only draft as a real edit', () => {
    const drafts = new Map([
      [
        key,
        {
          correctRank: 4,
          selections: [{ rank: 4, dirName: 'Conc.Aurel.531', source: 'model' as const }],
          notes: '',
        },
      ],
    ])

    expect(seedDraftMapIfEmpty(drafts, key, saved)).toBe(drafts)
  })
})

describe('draft keys', () => {
  it('separates reviewers, queries, models and variants', () => {
    const mine = makeDraftKey(7, 'LaTa', 'sif_abtt', 2)
    expect(makeDraftKey(7, 'LaTa', 'sif_abtt', 3)).not.toBe(mine)
    expect(makeDraftKey(8, 'LaTa', 'sif_abtt', 2)).not.toBe(mine)
    expect(makeDraftKey(7, 'PhilTa', 'sif_abtt', 2)).not.toBe(mine)
    expect(makeDraftKey(7, 'LaTa', 'raw', 2)).not.toBe(mine)
  })

  it('never collides with the unowned pre-#157 key it has to read alongside', () => {
    expect(makeDraftKey(7, 'LaTa', 'sif_abtt', 2)).not.toBe(
      legacyDraftKey(7, 'LaTa', 'sif_abtt'),
    )
    // A signed-out browser is its own scope, not the legacy one: an anonymous
    // draft must not be handed to the next account either.
    expect(makeDraftKey(7, 'LaTa', 'sif_abtt', null)).not.toBe(
      legacyDraftKey(7, 'LaTa', 'sif_abtt'),
    )
  })
})

describe('draft selections', () => {
  it('reads an old rank-only draft as an unidentified choice', () => {
    expect(draftSelections({ correctRank: 3, notes: '' })).toEqual([
      { rank: 3, dirName: null, source: null },
    ])
    expect(
      draftSelections({ correctRank: 3, selectedRanks: [3, 9], notes: '' }),
    ).toEqual([
      { rank: 3, dirName: null, source: null },
      { rank: 9, dirName: null, source: null },
    ])
  })

  it('reports no positive choice for a None draft', () => {
    expect(draftSelections({ correctRank: 0, notes: '' })).toEqual([])
  })

  it('makes the first choice canonical and keeps the notes', () => {
    const draft = draftWithSelections({ correctRank: null, notes: 'keep me' }, [
      { rank: 11, dirName: 'reviewer-dir', source: 'reviewer' },
      { rank: 2, dirName: 'candidate-2', source: 'model' },
    ])
    expect(draft).toEqual({
      correctRank: 11,
      selectedRanks: [11, 2],
      selections: [
        { rank: 11, dirName: 'reviewer-dir', source: 'reviewer' },
        { rank: 2, dirName: 'candidate-2', source: 'model' },
      ],
      notes: 'keep me',
    })
    expect(draftWithSelections(draft, [])).toEqual({ correctRank: null, notes: 'keep me' })
    expect(draftWithNone(draft, true)).toEqual({ correctRank: 0, notes: 'keep me' })
  })
})

describe('stored drafts', () => {
  it('round-trips identity-bearing selections through localStorage', () => {
    const drafts = new Map<string, FeedbackDraft>([
      [
        key,
        {
          correctRank: 11,
          selectedRanks: [11, 1],
          selections: [
            { rank: 11, dirName: 'reviewer-dir', source: 'reviewer' },
            { rank: 1, dirName: 'candidate-1', source: 'model' },
          ],
          notes: 'both readings',
        },
      ],
    ])

    expect(deserializeDrafts(serializeDrafts(drafts))).toEqual(drafts)
  })

  it('reads anything an older or broken build may have written', () => {
    expect(deserializeDrafts(null).size).toBe(0)
    expect(deserializeDrafts('not json').size).toBe(0)
    expect(deserializeDrafts('{"a":1}').size).toBe(0)
    // A pre-#157 entry survives as ranks with no identity, not as a decision
    // about whatever now stands at that rank.
    const legacy = deserializeDrafts(
      JSON.stringify([['0-KaLM-sif_abtt', { correctRank: 2, notes: 'old' }]]),
    )
    expect(legacy.get('0-KaLM-sif_abtt')).toEqual({ correctRank: 2, notes: 'old' })
  })

  it('drops junk fields rather than repairing them into an answer', () => {
    expect(sanitizeDraft({ correctRank: '2', notes: 5 })).toEqual({
      correctRank: null,
      notes: '',
    })
    expect(
      sanitizeDraft({
        correctRank: 1,
        selections: [
          { rank: 1, dirName: 'candidate-1', source: 'model' },
          { rank: 'two', dirName: 'candidate-2', source: 'model' },
          { rank: 3, dirName: 42, source: 'nonsense' },
        ],
        notes: '',
      }),
    ).toEqual({
      correctRank: 1,
      selections: [
        { rank: 1, dirName: 'candidate-1', source: 'model' },
        // Kept as a rank with no identity: it must be reconfirmed, and it
        // cannot be silently bound to the directory sitting there now.
        { rank: 3, dirName: null, source: null },
      ],
      notes: '',
    })
    expect(sanitizeDraft('nope')).toBeNull()
  })
})
