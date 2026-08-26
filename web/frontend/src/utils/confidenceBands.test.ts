import { describe, expect, it } from 'vitest'
import {
  bandsFrom,
  FALLBACK_BANDS,
  getConfidenceBand,
  LIKELY_MATCH_THRESHOLD,
  NO_MATCH_THRESHOLD,
} from './confidenceBands'

describe('confidence band thresholds', () => {
  it('keeps the fallback thresholds at the meeting-agreed values', () => {
    // These numbers are the whole point of issue #94. Since #95 the SERVER
    // owns them (web/bands.py, served on /api/models); these literals are the
    // pre-flight fallback and must stay equal to it.
    expect(NO_MATCH_THRESHOLD).toBe(0.5)
    expect(LIKELY_MATCH_THRESHOLD).toBe(0.7)
    expect(FALLBACK_BANDS).toEqual({ no_match: 0.5, verify: 0.7 })
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

  it('treats a non-finite score as no-match, never as a match', () => {
    expect(getConfidenceBand(Number.NaN)).toBe('no_match')
    expect(getConfidenceBand(Number.NEGATIVE_INFINITY)).toBe('no_match')
  })
})

describe('thresholds are the deployment\'s, not this bundle\'s', () => {
  it('reads them off the /api/models payload', () => {
    const served = { no_match: 0.62, verify: 0.81 }
    const bands = bandsFrom([{ confidence_bands: served }])
    expect(bands).toEqual(served)
    // ... and the band function actually follows them, rather than the
    // literals above.
    expect(getConfidenceBand(0.55, bands)).toBe('no_match')
    expect(getConfidenceBand(0.62, bands)).toBe('careful')
    expect(getConfidenceBand(0.81, bands)).toBe('likely')
  })

  it('falls back before the models call resolves, or on an older backend', () => {
    expect(bandsFrom(null)).toEqual(FALLBACK_BANDS)
    expect(bandsFrom([])).toEqual(FALLBACK_BANDS)
    expect(bandsFrom([{}])).toEqual(FALLBACK_BANDS)
  })
})
