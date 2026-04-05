import { useCallback } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useFeedback } from '../../contexts/FeedbackContext'
import { useReviewer } from '../../contexts/ReviewerContext'
import MatchPills from './MatchPills'
import NotesTextarea from './NotesTextarea'
import SubmitButton from './SubmitButton'

export default function FeedbackPanel() {
  const { activeQueryId, activeModel, setActiveQueryId } = useApp()
  const { getDraft, updateDraft, submitFeedback } = useFeedback()
  const { reviewerName, clearReviewer } = useReviewer()

  const draft = activeQueryId !== null ? getDraft(activeQueryId, activeModel) : null

  const handleMatchChange = useCallback(
    (rank: number | null) => {
      if (activeQueryId === null) return
      updateDraft(activeQueryId, activeModel, { correctRank: rank })
    },
    [activeQueryId, activeModel, updateDraft],
  )

  const handleNotesChange = useCallback(
    (notes: string) => {
      if (activeQueryId === null) return
      updateDraft(activeQueryId, activeModel, { notes })
    },
    [activeQueryId, activeModel, updateDraft],
  )

  const handleSubmit = useCallback(async () => {
    if (activeQueryId === null) return
    await submitFeedback(activeQueryId, activeModel, reviewerName ?? undefined)
    // Auto-advance to next query after a short delay
    setTimeout(() => {
      setActiveQueryId(activeQueryId + 1)
    }, 500)
  }, [activeQueryId, activeModel, reviewerName, submitFeedback, setActiveQueryId])

  const handleSkip = useCallback(() => {
    if (activeQueryId === null) return
    setActiveQueryId(activeQueryId + 1)
  }, [activeQueryId, setActiveQueryId])

  if (activeQueryId === null) {
    return (
      <div data-tour="feedback" className="flex-shrink-0">
        <p className="text-xs text-stone-400 dark:text-stone-500 font-ui text-center py-4">
          Select a query to provide feedback
        </p>
      </div>
    )
  }

  return (
    <div data-tour="feedback" className="flex-shrink-0 flex flex-col gap-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-0">
        Your Assessment
      </div>

      <MatchPills
        selectedRank={draft?.correctRank ?? null}
        onChange={handleMatchChange}
      />

      <NotesTextarea
        value={draft?.notes ?? ''}
        onChange={handleNotesChange}
      />

      {/* Reviewer identity (read-only) */}
      <div className="flex items-center gap-2 text-xs font-ui text-stone-500 dark:text-stone-400">
        <span>
          Reviewing as{' '}
          <span className="font-medium text-stone-700 dark:text-stone-200">
            {reviewerName}
          </span>
        </span>
        <button
          type="button"
          onClick={clearReviewer}
          className="text-accent hover:text-accent-dark transition-colors underline"
        >
          change
        </button>
      </div>

      <SubmitButton
        onSubmit={handleSubmit}
        onSkip={handleSkip}
        disabled={!activeModel}
      />
    </div>
  )
}
