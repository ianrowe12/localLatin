// Mirrors backend web/variants.py PredictionVariant.
//
// The reviewer-facing variant selector (#48) is gone as of issue #94: the app
// now shows one pipeline, DEFAULT_VARIANT, for the prediction list, candidate
// texts, feedback drafts and token highlights alike. The backend still serves
// all four and feedback rows still record which variant they belong to, so
// this vocabulary stays.
export type PredictionVariant = 'raw' | 'abtt' | 'sif' | 'sif_abtt'

export const PREDICTION_VARIANTS: PredictionVariant[] = [
  'raw',
  'abtt',
  'sif',
  'sif_abtt',
]

export const DEFAULT_VARIANT: PredictionVariant = 'sif_abtt'

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
