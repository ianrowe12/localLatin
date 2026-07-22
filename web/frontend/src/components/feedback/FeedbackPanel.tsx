import { useCallback, useEffect, useRef, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useFeedback, type FeedbackDraft } from '../../contexts/FeedbackContext'
import { isFeedbackDraftEmpty } from '../../contexts/feedbackDraft'
import { useReviewer } from '../../contexts/ReviewerContext'
import { fetchNextQuery, usePredictions } from '../../api/queries'
import { fetchLatestFeedback, type FeedbackEntry } from '../../api/feedback'
import MatchPills, { type MatchSelection } from './MatchPills'
import NotesTextarea from './NotesTextarea'
import SubmitButton from './SubmitButton'

function draftFromEntry(entry: FeedbackEntry): FeedbackDraft {
  const notes = entry.notes ?? ''
  if (entry.outcome === 'matched_rank') {
    if (entry.selected_ranks && entry.selected_ranks.length > 0) {
      return {
        correctRank: entry.selected_ranks[0],
        selectedRanks: entry.selected_ranks,
        notes,
      }
    }
    return { correctRank: entry.correct_rank, notes }
  }
  if (entry.outcome === 'none_of_top_k') {
    return { correctRank: 0, notes }
  }
  // skipped / legacy_unresolved → seed the notes only, leave selection unset.
  return { correctRank: null, notes }
}

function draftsEqual(a: FeedbackDraft, b: FeedbackDraft): boolean {
  if (a.correctRank !== b.correctRank || a.notes !== b.notes) return false
  const ar = a.selectedRanks ?? []
  const br = b.selectedRanks ?? []
  return ar.length === br.length && ar.every((value, index) => value === br[index])
}

export default function FeedbackPanel() {
  const { activeQueryId, activeModel, setActiveQueryId } = useApp()
  const {
    drafts,
    getDraft,
    updateDraft,
    seedDraftIfEmpty,
    submitFeedback,
    skipFeedback,
  } = useFeedback()
  const { reviewerName, clearReviewer } = useReviewer()
  const { data: predictionData } = usePredictions(activeQueryId, activeModel)
  const [skipNeedsNote, setSkipNeedsNote] = useState(false)
  const [multiSelect, setMultiSelect] = useState(false)
  const [seeded, setSeeded] = useState<{ key: string; draft: FeedbackDraft } | null>(null)

  // Always-current view of the draft map for use inside async callbacks.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  const draft = activeQueryId !== null ? getDraft(activeQueryId, activeModel) : null
  const predictions = predictionData?.predictions?.slice(0, 10) ?? []
  const draftKey =
    activeQueryId !== null ? `${activeQueryId}-${activeModel}` : null

  // Requirement: multi-select is OFF by default and resets for every new query.
  useEffect(() => {
    setMultiSelect(false)
  }, [activeQueryId, activeModel])

  // Seed notes/selection from the reviewer's last submitted feedback, but never
  // clobber a local unsaved draft (or a late typed draft via the resolve re-check).
  useEffect(() => {
    if (activeQueryId === null || !activeModel) return
    const key = `${activeQueryId}-${activeModel}`
    let cancelled = false
    fetchLatestFeedback(activeQueryId, activeModel)
      .then((entry) => {
        if (cancelled || entry === null) return
        if (!isFeedbackDraftEmpty(draftsRef.current.get(key))) return
        const seededDraft = draftFromEntry(entry)
        seedDraftIfEmpty(activeQueryId, activeModel, seededDraft)
        setSeeded({ key, draft: seededDraft })
        if (entry.selected_ranks && entry.selected_ranks.length > 1) {
          setMultiSelect(true)
        }
      })
      .catch(() => {
        // Best-effort: a failed latest-feedback fetch just leaves the form empty.
      })
    return () => {
      cancelled = true
    }
  }, [activeQueryId, activeModel, seedDraftIfEmpty])

  // Drop any selected ranks that are no longer part of the current predictions.
  useEffect(() => {
    if (activeQueryId === null || predictions.length === 0 || !draft) return
    const validRanks = new Set(predictions.map((prediction) => prediction.rank))

    if (draft.selectedRanks && draft.selectedRanks.length > 0) {
      const pruned = draft.selectedRanks.filter((rank) => validRanks.has(rank))
      if (pruned.length !== draft.selectedRanks.length) {
        updateDraft(activeQueryId, activeModel, {
          selectedRanks: pruned.length > 0 ? pruned : undefined,
          correctRank: pruned.length > 0 ? pruned[0] : null,
        })
        return
      }
    }

    if (
      draft.correctRank &&
      draft.correctRank > 0 &&
      !validRanks.has(draft.correctRank)
    ) {
      updateDraft(activeQueryId, activeModel, {
        correctRank: null,
        selectedRanks: undefined,
      })
    }
  }, [
    activeQueryId,
    activeModel,
    draft?.correctRank,
    draft?.selectedRanks,
    predictions,
    updateDraft,
  ])

  const handleMatchChange = useCallback(
    (next: MatchSelection) => {
      if (activeQueryId === null) return
      if (next.noneSelected) {
        updateDraft(activeQueryId, activeModel, {
          correctRank: 0,
          selectedRanks: undefined,
        })
      } else if (next.selectedRanks.length === 0) {
        updateDraft(activeQueryId, activeModel, {
          correctRank: null,
          selectedRanks: undefined,
        })
      } else if (multiSelect) {
        updateDraft(activeQueryId, activeModel, {
          correctRank: next.selectedRanks[0],
          selectedRanks: next.selectedRanks,
        })
      } else {
        updateDraft(activeQueryId, activeModel, {
          correctRank: next.selectedRanks[0],
          selectedRanks: undefined,
        })
      }
    },
    [activeQueryId, activeModel, multiSelect, updateDraft],
  )

  const handleMultiSelectToggle = useCallback(
    (checked: boolean) => {
      setMultiSelect(checked)
      if (!checked && activeQueryId !== null) {
        // Collapse a multi answer down to its first rank for single mode.
        const current = getDraft(activeQueryId, activeModel)
        if (current.selectedRanks && current.selectedRanks.length > 0) {
          updateDraft(activeQueryId, activeModel, {
            correctRank: current.selectedRanks[0],
            selectedRanks: undefined,
          })
        }
      }
    },
    [activeQueryId, activeModel, getDraft, updateDraft],
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

  const selectedRanks =
    draft?.selectedRanks && draft.selectedRanks.length > 0
      ? draft.selectedRanks
      : draft?.correctRank && draft.correctRank > 0
        ? [draft.correctRank]
        : []
  const noneSelected = draft?.correctRank === 0
  const showSeededHint =
    seeded !== null &&
    draftKey !== null &&
    seeded.key === draftKey &&
    draft !== null &&
    draftsEqual(draft, seeded.draft)

  return (
    <div data-tour="feedback" className="flex-shrink-0 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-0">
          Your Assessment
        </div>
        <label
          data-tour="multi-select"
          className="flex items-center gap-1.5 text-xs font-ui text-stone-500 dark:text-stone-400 cursor-pointer select-none"
        >
          <input
            type="checkbox"
            checked={multiSelect}
            onChange={(event) => handleMultiSelectToggle(event.target.checked)}
            className="h-3.5 w-3.5 rounded border-stone-300 dark:border-stone-600 text-accent focus:ring-2 focus:ring-accent/40"
          />
          Select multiple
        </label>
      </div>

      <MatchPills
        multiSelect={multiSelect}
        selectedRanks={selectedRanks}
        noneSelected={noneSelected}
        maxRank={predictions.length || 10}
        onChange={handleMatchChange}
      />

      {showSeededHint && (
        <p className="text-xs font-ui italic text-stone-400 dark:text-stone-500">
          Loaded from your last submitted review
        </p>
      )}

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
