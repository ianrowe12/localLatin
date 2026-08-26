import { memo } from 'react'
import type { Prediction } from '../../api/queries'
import {
  BAND_COPY,
  BAND_STYLES,
  getConfidenceBand,
} from '../../utils/confidenceBands'

interface PredictionCardProps {
  prediction: Prediction
  rank: number
  isActive: boolean
  onClick: () => void
}

function PredictionCardInner({ prediction, rank, isActive, onClick }: PredictionCardProps) {
  // One band per card, from the displayed similarity (issue #94). The
  // thresholds live in utils/confidenceBands.ts and nowhere else.
  const band = getConfidenceBand(prediction.score)
  const styles = BAND_STYLES[band]
  const copy = BAND_COPY[band]

  const inactiveBorder =
    band === 'no_match' ? styles.card : 'border-transparent hover:bg-stone-50 dark:hover:bg-stone-800'

  return (
    <button
      type="button"
      onClick={onClick}
      data-band={band}
      className={`p-2.5 rounded-lg cursor-pointer transition-all duration-150 text-left w-full border ${
        isActive
          ? 'border-accent bg-accent/5 dark:bg-accent/10 shadow-sm'
          : inactiveBorder
      }`}
      aria-pressed={isActive}
      aria-label={`Prediction rank ${rank}: ${prediction.dir_name}. ${copy.label}, similarity ${prediction.score.toFixed(3)}`}
    >
      <div className="flex items-start gap-3">
        {/* Rank circle */}
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 ${
            isActive
              ? 'bg-accent text-white'
              : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300'
          }`}
        >
          {rank}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div
            className="text-sm font-medium truncate max-w-full text-stone-800 dark:text-stone-200"
            title={prediction.dir_name}
          >
            {prediction.dir_name}
          </div>

          {/* Score bar */}
          <div className="h-1.5 rounded-full bg-stone-200 dark:bg-stone-700 mt-1.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${styles.bar}`}
              style={{ width: `${Math.max(0, Math.min(1, prediction.score)) * 100}%` }}
            />
          </div>

          {/* Score text */}
          <div className="text-xs text-stone-500 mt-1">
            <span className="font-ui">Similarity:</span>{' '}
            <span className="font-mono">{prediction.score.toFixed(3)}</span>
          </div>

          {/* Confidence band chip */}
          <div
            data-testid={`band-chip-${rank}`}
            className={`mt-1.5 inline-block rounded-full px-2 py-0.5 font-ui text-[11px] leading-tight ${styles.chip}`}
            title={copy.note}
          >
            {copy.label}
          </div>
        </div>
      </div>
    </button>
  )
}

const PredictionCard = memo(PredictionCardInner)
PredictionCard.displayName = 'PredictionCard'

export default PredictionCard
