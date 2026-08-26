import { describe, expect, it } from 'vitest'
import {
  getConfidenceBand,
  isNoMatch,
  LIKELY_MATCH_THRESHOLD,
  NO_MATCH_THRESHOLD,
} from './confidenceBands'

describe('confidence band thresholds', () => {
  it('keeps the two thresholds at the meeting-agreed values', () => {
    // These numbers are the whole point of issue #94. Anything that wants a
    // different floor changes them HERE, not at a call site.
    expect(NO_MATCH_THRESHOLD).toBe(0.5)
    expect(LIKELY_MATCH_THRESHOLD).toBe(0.7)
  })

  it.each([
    [0, 'no_match'],
    [0.49, 'no_match'],
    [0.4999, 'no_match'],
    [0.5, 'careful'],
    [0.69, 'careful'],
    [0.6999, 'careful'],
    // Half-open bands: each threshold belongs to the band above it.
    [0.7, 'likely'],
    [0.71, 'likely'],
    [1, 'likely'],
  ])('maps %s to the %s band', (score, band) => {
    expect(getConfidenceBand(score as number)).toBe(band)
  })

  it('treats a missing or non-finite score as no-match, never as a match', () => {
    expect(getConfidenceBand(Number.NaN)).toBe('no_match')
    expect(isNoMatch(null)).toBe(true)
    expect(isNoMatch(undefined)).toBe(true)
    expect(isNoMatch(0.49)).toBe(true)
    expect(isNoMatch(0.5)).toBe(false)
  })
})
