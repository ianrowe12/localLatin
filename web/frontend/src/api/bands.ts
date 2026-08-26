import type { ModelInfo } from './models'

/**
 * Confidence bands on the displayed SIF+ABTT similarity.
 *
 * THE BACKEND OWNS THESE NUMBERS (web/bands.py). They are not merely
 * presentational: the server uses `no_match` to decide whether a
 * reviewer-created directory has left `awaiting_match`, so a hardcoded copy
 * here would be a second source of truth able to drift from the one the API
 * answers with. `GET /api/models` therefore carries a `confidence_bands`
 * object on every entry and `bandsFrom` reads it.
 *
 * The literals below are the pre-flight fallback only, for the render that
 * happens before the first `/api/models` response lands. Keep them equal to
 * web/bands.py.
 */
export interface ConfidenceBands {
  /** Below this: overt "potentially no match", new-directory CTA is default. */
  no_match: number
  /** At or above this: calm "likely match - verify" framing. */
  verify: number
}

export const FALLBACK_BANDS: ConfidenceBands = { no_match: 0.5, verify: 0.7 }

/** The deployment's bands, or the fallback before the models call resolves. */
export function bandsFrom(models: ModelInfo[] | null | undefined): ConfidenceBands {
  return models?.[0]?.confidence_bands ?? FALLBACK_BANDS
}

export type ConfidenceLevel = 'no_match' | 'review' | 'likely'

/** Which band a score falls in. Boundaries are inclusive at the low end. */
export function bandOf(score: number, bands: ConfidenceBands): ConfidenceLevel {
  if (score < bands.no_match) return 'no_match'
  if (score < bands.verify) return 'review'
  return 'likely'
}
