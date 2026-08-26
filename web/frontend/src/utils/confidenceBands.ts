/**
 * Confidence bands on the displayed similarity (issue #94).
 *
 * The score the reviewer sees is the SIF+ABTT cosine similarity between the
 * query and the candidate directory: since issue #94 the app serves exactly
 * one pipeline, so "the displayed similarity" is unambiguous.
 *
 * THIS FILE IS THE ONLY PLACE THE TWO THRESHOLDS ARE WRITTEN DOWN. Anything
 * that needs to style, label or gate on confidence imports from here.
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

/** Below this, the top hit is flagged as a potential non-match. */
export const NO_MATCH_THRESHOLD = 0.5

/** At or above this, the match gets calm "verify it" framing. */
export const LIKELY_MATCH_THRESHOLD = 0.7

export type ConfidenceBand = 'no_match' | 'careful' | 'likely'

export function getConfidenceBand(score: number): ConfidenceBand {
  if (!Number.isFinite(score) || score < NO_MATCH_THRESHOLD) return 'no_match'
  if (score < LIKELY_MATCH_THRESHOLD) return 'careful'
  return 'likely'
}

/** True when the no-match treatment (red banner + new-directory CTA) applies. */
export function isNoMatch(score: number | null | undefined): boolean {
  return score == null || getConfidenceBand(score) === 'no_match'
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
export const BAND_STYLES: Record<
  ConfidenceBand,
  { chip: string; bar: string; card: string }
> = {
  no_match: {
    chip: 'bg-incorrect/15 text-incorrect ring-1 ring-incorrect/40 font-semibold',
    bar: 'bg-incorrect',
    card: 'border-incorrect/50 bg-incorrect/5',
  },
  careful: {
    chip: 'bg-highlight/15 text-highlight-dark dark:text-highlight-light ring-1 ring-highlight/40',
    bar: 'bg-highlight',
    card: '',
  },
  likely: {
    chip: 'bg-stone-100 text-stone-600 dark:bg-stone-700 dark:text-stone-300',
    bar: 'bg-accent',
    card: '',
  },
}
