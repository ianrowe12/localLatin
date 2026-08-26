/**
 * Confidence bands on the displayed similarity (issue #94).
 *
 * The score the reviewer sees is the SIF+ABTT cosine similarity between the
 * query and the candidate directory: since issue #94 the app serves exactly
 * one pipeline, so "the displayed similarity" is unambiguous.
 *
 * THE SERVER OWNS THE THRESHOLDS (web/bands.py); this file owns everything
 * built on top of them -- the band function, the copy and the styling. Anything
 * that needs to style, label or gate on confidence imports from here.
 *
 * The numbers are not merely presentational: the backend decides a reviewer
 * directory's status with the no-match band (issue #95), so a second copy in
 * TypeScript would be a source of truth able to drift from the one the API
 * answers with. `GET /api/models` carries them as `confidence_bands` and
 * `bandsFrom` reads them. The literals below are the pre-flight fallback, for
 * the render before that response lands; keep them equal to web/bands.py.
 *
 * Why 0.5 / 0.7: the learned per-model tau under sif_abtt sits at 0.34-0.43
 * (train-fit), so 0.5 is deliberately conservative. A score below it is not
 * "wrong" in the model's own terms, it is "the model has no real opinion" and
 * the reviewer should be told so before they read the ranking.
 *
 * Boundaries are half-open, so each threshold belongs to the band above it:
 *
 *   [0, 0.5)   no_match  overt red: the ranking is probably noise
 *   [0.5, 0.7) careful   read this match carefully before accepting it
 *   [0.7, 1]   likely    calm: likely match, verify it
 */

/** Fallback only. The served value is `confidence_bands.no_match`. */
export const NO_MATCH_THRESHOLD = 0.5

/** Fallback only. The served value is `confidence_bands.verify`. */
export const LIKELY_MATCH_THRESHOLD = 0.7

/**
 * The shape `GET /api/models` serves. Field names match web/models.py
 * ConfidenceBands rather than the band names below, because they name the
 * *edges* between bands, not the bands themselves.
 */
export interface ConfidenceBands {
  /** Below this: overt "potentially no match". */
  no_match: number
  /** At or above this: calm "likely match - verify". */
  verify: number
}

export const FALLBACK_BANDS: ConfidenceBands = {
  no_match: NO_MATCH_THRESHOLD,
  verify: LIKELY_MATCH_THRESHOLD,
}

/**
 * The deployment's thresholds, or the fallback before /api/models resolves.
 *
 * Structurally typed rather than taking `ModelInfo`, so this module stays a
 * leaf and cannot form an import cycle with the API layer. The bands are
 * deployment-wide, replicated onto every model entry the way `default_variant`
 * is, so the first entry is as good as any.
 */
export function bandsFrom(
  models: ReadonlyArray<{ confidence_bands?: ConfidenceBands }> | null | undefined,
): ConfidenceBands {
  return models?.[0]?.confidence_bands ?? FALLBACK_BANDS
}

export type ConfidenceBand = 'no_match' | 'careful' | 'likely'

export function getConfidenceBand(
  score: number,
  bands: ConfidenceBands = FALLBACK_BANDS,
): ConfidenceBand {
  if (!Number.isFinite(score) || score < bands.no_match) return 'no_match'
  if (score < bands.verify) return 'careful'
  return 'likely'
}

export interface BandCopy {
  /** Short chip text shown on the prediction card. */
  label: string
  /** One line explaining what the reviewer should do about it. */
  note: string
}

export const BAND_COPY: Record<ConfidenceBand, BandCopy> = {
  no_match: {
    label: 'Potentially no match',
    note: 'Similarity is below the confidence floor. This fragment may belong to a text that is not in the corpus yet.',
  },
  careful: {
    label: 'Review this match carefully',
    note: 'Similarity is in the uncertain range. Read the evidence before accepting or rejecting it.',
  },
  likely: {
    label: 'Likely match - verify',
    note: 'Similarity is high. Verify the evidence, then record your assessment.',
  },
}

/** Tailwind classes for each band, kept next to the thresholds they follow. */
export const BAND_STYLES: Record<ConfidenceBand, { chip: string; bar: string }> = {
  no_match: {
    chip: 'bg-incorrect/15 text-incorrect ring-1 ring-incorrect/40 font-semibold',
    bar: 'bg-incorrect',
  },
  careful: {
    chip: 'bg-highlight/15 text-highlight-dark dark:text-highlight-light ring-1 ring-highlight/40',
    bar: 'bg-highlight',
  },
  likely: {
    chip: 'bg-stone-100 text-stone-600 dark:bg-stone-700 dark:text-stone-300',
    bar: 'bg-accent',
  },
}

/**
 * Card outline for a no-match hit. Only this band tints the card itself: the
 * other two would turn a ten-card list into a traffic light and cost the red
 * band the contrast that makes it readable in a split second.
 */
export const NO_MATCH_CARD_STYLE = 'border-incorrect/50 bg-incorrect/5'
