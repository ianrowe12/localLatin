import { memo } from 'react'
import type { Prediction } from '../../api/queries'

interface PredictionCardProps {
  prediction: Prediction
  rank: number
  isActive: boolean
  onClick: () => void
}

function PredictionCardInner({ prediction, rank, isActive, onClick }: PredictionCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`p-2.5 rounded-lg cursor-pointer transition-all duration-150 text-left w-full ${
        isActive
          ? 'border border-accent bg-accent/5 dark:bg-accent/10 shadow-sm'
          : 'border border-transparent hover:bg-stone-50 dark:hover:bg-stone-800'
      }`}
      aria-pressed={isActive}
      aria-label={`Prediction rank ${rank}: ${prediction.dir_name}`}
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
              className="h-full bg-accent rounded-full transition-all"
              style={{ width: `${prediction.score * 100}%` }}
            />
          </div>

          {/* Score text */}
          <div className="text-xs text-stone-500 font-mono mt-1">
            {prediction.score.toFixed(3)}
          </div>
        </div>
      </div>
    </button>
  )
}

const PredictionCard = memo(PredictionCardInner)
PredictionCard.displayName = 'PredictionCard'

export default PredictionCard
