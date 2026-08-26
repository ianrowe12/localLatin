import { describe, expect, it } from 'vitest'
import { formatNoteAttribution, formatNoteDate } from './noteAttribution'
import type { FeedbackEntry } from '../../api/feedback'

const entry: FeedbackEntry = {
  id: 1,
  query_id: 7,
  timestamp: '2026-08-24 09:15:00',
  model_slug: 'bowphs_LaTa',
  variant: 'sif_abtt',
  outcome: 'matched_rank',
  correct_rank: 2,
  correct_dir: 'candidate-2',
  selected_ranks: null,
  notes: 'shares the incipit',
  reviewer: 'Alice Antiqua',
  reviewer_account_id: 1,
  reviewer_username: 'alice',
  schema_version: 2,
}

describe('formatNoteDate', () => {
  it('reads the date off a SQLite timestamp without going through Date', () => {
    // 'YYYY-MM-DD HH:MM:SS' has no offset, so parsing it would shift the day in
    // some timezones. The stored calendar date is what reviewers should see.
    expect(formatNoteDate('2026-08-24 23:45:00')).toBe('2026-08-24')
  })

  it('reads an ISO timestamp off the same prefix', () => {
    // ISO shares the YYYY-MM-DD prefix, so this never reaches the Date branch.
    expect(formatNoteDate('2026-08-24T09:15:00.000Z')).toBe('2026-08-24')
  })

  it('parses a date that has no YYYY-MM-DD prefix at all', () => {
    // The only input that actually exercises the Date fallback. Anchored at
    // midday UTC so the assertion cannot flip a day under the runner's zone.
    expect(formatNoteDate('Aug 24, 2026 12:00:00 UTC')).toBe('2026-08-24')
  })

  it('passes an unparseable value straight through', () => {
    expect(formatNoteDate('not a date')).toBe('not a date')
  })
})

describe('formatNoteAttribution', () => {
  it('credits the author by login name', () => {
    expect(formatNoteAttribution(entry)).toBe('Last note by alice - 2026-08-24')
  })

  it('falls back to the display name when no account is linked', () => {
    expect(
      formatNoteAttribution({ ...entry, reviewer_username: null }),
    ).toBe('Last note by Alice Antiqua - 2026-08-24')
  })

  it('does not call a decision without notes a note', () => {
    expect(formatNoteAttribution({ ...entry, notes: '   ' })).toBe(
      'Last review by alice - 2026-08-24',
    )
  })

  it('returns null when there is no entry to attribute', () => {
    expect(formatNoteAttribution(null)).toBeNull()
  })
})
