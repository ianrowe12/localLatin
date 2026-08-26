import { memo } from 'react'
import type { Prediction } from '../../api/queries'

interface ReviewerDirCardProps {
  prediction: Prediction
  isActive: boolean
  onClick: () => void
}

/**
 * A candidate that came from a reviewer, not from the labelled corpus.
 *
 * Deliberately not a variant of PredictionCard. The two carry different kinds
 * of evidence -- a labelled directory is an edited, attested source, a reviewer
 * directory is one colleague's judgement -- and a reviewer scanning the list
 * has to be able to tell them apart without reading. So this card is visually
 * distinct at a glance: a dashed indigo frame instead of a solid rank circle,
 * the reviewer's label as the title with the opaque `reviewer-dir-N` id
 * demoted, and an explicit attribution line. The score itself is on the same
 * scale as the model's candidates (max cosine over member documents), so the
 * bar and the number are rendered the same way and stay comparable.
 */
function ReviewerDirCardInner({ prediction, isActive, onClick }: ReviewerDirCardProps) {
  const title = prediction.label || prediction.dir_name
  const memberCount = prediction.dir_files.length

  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={`reviewer-dir-card-${prediction.dir_name}`}
      className={`p-2.5 rounded-lg cursor-pointer transition-all duration-150 text-left w-full border-dashed ${
        isActive
          ? 'border-2 border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10 shadow-sm'
          : 'border-2 border-indigo-300/70 dark:border-indigo-400/40 hover:bg-indigo-50/60 dark:hover:bg-indigo-500/5'
      }`}
      aria-pressed={isActive}
      aria-label={`Reviewer directory ${title}, rank ${prediction.rank}`}
    >
      <div className="flex items-start gap-3">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold flex-shrink-0 ${
            isActive
              ? 'bg-indigo-500 text-white'
              : 'bg-indigo-100 dark:bg-indigo-500/20 text-indigo-700 dark:text-indigo-300'
          }`}
          aria-hidden="true"
        >
          {prediction.rank}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-300 whitespace-nowrap">
              Reviewer directory
            </span>
          </div>

          <div
            className="text-sm font-medium truncate max-w-full text-stone-800 dark:text-stone-200"
            title={title}
          >
            {title}
          </div>

          <div className="h-1.5 rounded-full bg-stone-200 dark:bg-stone-700 mt-1.5 overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all"
              style={{ width: `${Math.max(0, prediction.score) * 100}%` }}
            />
          </div>

          <div className="text-xs text-stone-500 mt-1">
            <span className="font-ui">Similarity:</span>{' '}
            <span className="font-mono">{prediction.score.toFixed(3)}</span>
          </div>

          <div className="font-ui text-[11px] text-stone-400 dark:text-stone-500 mt-0.5 truncate">
            {prediction.created_by ? `Created by ${prediction.created_by}` : 'Reviewer-created'}
            {' · '}
            {memberCount === 1 ? '1 document' : `${memberCount} documents`}
          </div>
        </div>
      </div>
    </button>
  )
}

const ReviewerDirCard = memo(ReviewerDirCardInner)
ReviewerDirCard.displayName = 'ReviewerDirCard'

export default ReviewerDirCard
