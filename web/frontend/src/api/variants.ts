// Mirrors backend web/variants.py PredictionVariant. This is the vocabulary
// the reviewer-facing variant selector (#48) drives: it picks which
// post-processing variant the prediction list, candidate texts, feedback
// drafts and token highlights are all computed from.
export type PredictionVariant = 'raw' | 'abtt' | 'sif' | 'sif_abtt'

export const PREDICTION_VARIANTS: PredictionVariant[] = [
  'raw',
  'abtt',
  'sif',
  'sif_abtt',
]

export const DEFAULT_VARIANT: PredictionVariant = 'sif_abtt'

export function isPredictionVariant(value: unknown): value is PredictionVariant {
  return PREDICTION_VARIANTS.includes(value as PredictionVariant)
}

/** Reviewer-facing label and one-line explanation for each variant. */
export interface VariantOption {
  key: PredictionVariant
  label: string
  description: string
}

export const VARIANT_OPTIONS: VariantOption[] = [
  { key: 'raw', label: 'Raw', description: 'no post-processing' },
  { key: 'abtt', label: 'ABTT', description: 'removes dominant embedding directions' },
  { key: 'sif', label: 'SIF', description: 'down-weights frequent words' },
  { key: 'sif_abtt', label: 'SIF+ABTT', description: 'both corrections (default)' },
]

// ---------------------------------------------------------------------------
// Prediction variant  <->  attribution variant
// ---------------------------------------------------------------------------

/**
 * Variant names used by the attribution artifacts and the /api/token_map
 * endpoints. Identical to `PredictionVariant` except that the
 * no-post-processing variant is called `baseline` there and `raw` in the
 * prediction CSVs.
 */
export type AttributionVariant = 'baseline' | 'abtt' | 'sif' | 'sif_abtt'

/**
 * THE single raw<->baseline translation point.
 *
 * The prediction pipeline (`unlabelled_predictions_<variant>.csv`) names the
 * uncorrected variant `raw`; the IG/attribution artifacts written by
 * scripts/ig/ name the very same thing `baseline` (`pair_matrix_ig_baseline`,
 * `query_ig_baseline`, ...). Rather than renaming either dataset, every place
 * that hands the reviewer's selected variant to the token-map API goes through
 * this function. Nothing else in the frontend should hardcode 'baseline'.
 */
export function toAttributionVariant(variant: PredictionVariant): AttributionVariant {
  return variant === 'raw' ? 'baseline' : variant
}
