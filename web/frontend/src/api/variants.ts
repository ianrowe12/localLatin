// Mirrors backend models.py PredictionVariant. Until the variant selector
// lands (issue #48) every caller passes DEFAULT_VARIANT, which matches the
// backend default, so behaviour is unchanged.
export type PredictionVariant = 'raw' | 'abtt' | 'sif' | 'sif_abtt'

export const PREDICTION_VARIANTS: PredictionVariant[] = [
  'raw',
  'abtt',
  'sif',
  'sif_abtt',
]

export const DEFAULT_VARIANT: PredictionVariant = 'sif_abtt'
