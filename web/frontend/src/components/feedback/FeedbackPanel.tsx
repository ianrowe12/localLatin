import { useCallback, useEffect, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useFeedback } from '../../contexts/FeedbackContext'
import { useReviewer } from '../../contexts/ReviewerContext'
import { fetchNextQuery, usePredictions } from '../../api/queries'
import MatchPills from './MatchPills'
import NotesTextarea from './NotesTextarea'
import SubmitButton from './SubmitButton'

export default function FeedbackPanel() {
  const { activeQueryId, activeModel, setActiveQueryId } = useApp()
  const { getDraft, updateDraft, submitFeedback, skipFeedback } = useFeedback()
  const { reviewerName, clearReviewer } = useReviewer()
  const { data: predictionData } = usePredictions(activeQueryId, activeModel)
  const [skipNeedsNote, setSkipNeedsNote] = useState(false)

  const draft = activeQueryId !== null ? getDraft(activeQueryId, activeModel) : null
  const predictions = predictionData?.predictions?.slice(0, 10) ?? []

  useEffect(() => {
    if (
      activeQueryId === null ||
      !draft?.correctRank ||
      draft.correctRank <= 0 ||
      predictions.length === 0
    ) {
      return
    }
    if (!predictions.some((prediction) => prediction.rank === draft.correctRank)) {
      updateDraft(activeQueryId, activeModel, { correctRank: null })
    }
  }, [activeQueryId, activeModel, draft?.correctRank, predictions, updateDraft])

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
      if (notes.trim()) setSkipNeedsNote(false)
    },
    [activeQueryId, activeModel, updateDraft],
  )

  const advanceToNextActionable = useCallback(
    async (after: number) => {
      const next = await fetchNextQuery(after)
      setActiveQueryId(next.file_id)
    },
    [setActiveQueryId],
  )

  const handleSubmit = useCallback(async () => {
    if (activeQueryId === null) return
    await submitFeedback(activeQueryId, activeModel, predictions)
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [activeQueryId, activeModel, predictions, submitFeedback, advanceToNextActionable])

  const handleSkip = useCallback(async () => {
    if (activeQueryId === null) return
    const notes = draft?.notes.trim() ?? ''
    if (!notes) {
      setSkipNeedsNote(true)
      return
    }
    await skipFeedback(activeQueryId, activeModel, notes)
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [activeQueryId, activeModel, draft?.notes, skipFeedback, advanceToNextActionable])

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
        maxRank={predictions.length || 10}
        onChange={handleMatchChange}
      />

      <NotesTextarea
        value={draft?.notes ?? ''}
        onChange={handleNotesChange}
      />

      {(skipNeedsNote || !(draft?.notes ?? '').trim()) && (
        <p className="text-xs font-ui text-stone-500 dark:text-stone-400">
          Add a note before skipping so the PI can follow up.
        </p>
      )}

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
          sign out
        </button>
      </div>

      <SubmitButton
        onSubmit={handleSubmit}
        onSkip={handleSkip}
        disabled={!activeModel}
        skipDisabled={!activeModel}
      />
    </div>
  )
}
