import { useCallback, useEffect, useRef, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useFeedback, type FeedbackDraft } from '../../contexts/FeedbackContext'
import { isFeedbackDraftEmpty } from '../../contexts/feedbackDraft'
import { useReviewer } from '../../contexts/ReviewerContext'
import { fetchNextQuery, usePredictions } from '../../api/queries'
import { fetchLatestFeedback, type FeedbackEntry } from '../../api/feedback'
import MatchPills, { type MatchSelection } from './MatchPills'
import NotesTextarea from './NotesTextarea'
import { formatNoteAttribution } from './noteAttribution'
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
  const {
    activeQueryId,
    activeModel,
    activeVariant,
    setActiveQueryId,
  } = useApp()
  const {
    drafts,
    makeDraftKey,
    getDraft,
    updateDraft,
    seedDraftIfEmpty,
    submitFeedback,
    skipFeedback,
  } = useFeedback()
  const { reviewerName, clearReviewer } = useReviewer()
  // Drafts, the latest-feedback prefill and the submitted row are all keyed on
  // the reviewer's active variant, so switching variants shows the assessment
  // that belongs to the ranking now on screen (issue #48).
  const { data: predictionData } = usePredictions(
    activeQueryId,
    activeModel,
    activeVariant,
  )
  const [skipNeedsNote, setSkipNeedsNote] = useState(false)
  const [multiSelect, setMultiSelect] = useState(false)
  const [seeded, setSeeded] = useState<{ key: string; draft: FeedbackDraft } | null>(null)
  // The shared latest review for the query on screen, kept whether or not it
  // seeded the form: a local unsaved draft wins the box, but the reviewer still
  // needs to know who last recorded something here (issue #96).
  const [latest, setLatest] = useState<{ key: string; entry: FeedbackEntry } | null>(
    null,
  )

  // Always-current view of the draft map for use inside async callbacks.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  const draft =
    activeQueryId !== null ? getDraft(activeQueryId, activeModel, activeVariant) : null
  const predictions = predictionData?.predictions?.slice(0, 10) ?? []
  const draftKey =
    activeQueryId !== null
      ? makeDraftKey(activeQueryId, activeModel, activeVariant)
      : null

  // Requirement: multi-select is OFF by default and resets for every new query.
  // A variant switch swaps in a different draft too, so it resets as well.
  useEffect(() => {
    setMultiSelect(false)
  }, [activeQueryId, activeModel, activeVariant])

  // Seed notes/selection from the last submitted feedback for this query --
  // from ANY reviewer, since notes are shared (issue #96) -- but never clobber a
  // local unsaved draft (or a late typed draft via the resolve re-check).
  useEffect(() => {
    if (activeQueryId === null || !activeModel) return
    const key = makeDraftKey(activeQueryId, activeModel, activeVariant)
    let cancelled = false
    fetchLatestFeedback(activeQueryId, activeModel, activeVariant)
      .then((entry) => {
        if (cancelled) return
        setLatest(entry === null ? null : { key, entry })
        if (entry === null) return
        if (!isFeedbackDraftEmpty(draftsRef.current.get(key))) return
        const seededDraft = draftFromEntry(entry)
        seedDraftIfEmpty(activeQueryId, activeModel, seededDraft, activeVariant)
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
  }, [activeQueryId, activeModel, activeVariant, makeDraftKey, seedDraftIfEmpty])

  // Drop any selected ranks that are no longer part of the current predictions.
  useEffect(() => {
    if (activeQueryId === null || predictions.length === 0 || !draft) return
    const validRanks = new Set(predictions.map((prediction) => prediction.rank))

    if (draft.selectedRanks && draft.selectedRanks.length > 0) {
      const pruned = draft.selectedRanks.filter((rank) => validRanks.has(rank))
      if (pruned.length !== draft.selectedRanks.length) {
        updateDraft(
          activeQueryId,
          activeModel,
          {
            selectedRanks: pruned.length > 0 ? pruned : undefined,
            correctRank: pruned.length > 0 ? pruned[0] : null,
          },
          activeVariant,
        )
        return
      }
    }

    if (
      draft.correctRank &&
      draft.correctRank > 0 &&
      !validRanks.has(draft.correctRank)
    ) {
      updateDraft(
        activeQueryId,
        activeModel,
        {
          correctRank: null,
          selectedRanks: undefined,
        },
        activeVariant,
      )
    }
  }, [
    activeQueryId,
    activeModel,
    activeVariant,
    draft?.correctRank,
    draft?.selectedRanks,
    predictions,
    updateDraft,
  ])

  const handleMatchChange = useCallback(
    (next: MatchSelection) => {
      if (activeQueryId === null) return
      if (next.noneSelected) {
        updateDraft(
          activeQueryId,
          activeModel,
          {
            correctRank: 0,
            selectedRanks: undefined,
          },
          activeVariant,
        )
      } else if (next.selectedRanks.length === 0) {
        updateDraft(
          activeQueryId,
          activeModel,
          {
            correctRank: null,
            selectedRanks: undefined,
          },
          activeVariant,
        )
      } else if (multiSelect) {
        updateDraft(
          activeQueryId,
          activeModel,
          {
            correctRank: next.selectedRanks[0],
            selectedRanks: next.selectedRanks,
          },
          activeVariant,
        )
      } else {
        updateDraft(
          activeQueryId,
          activeModel,
          {
            correctRank: next.selectedRanks[0],
            selectedRanks: undefined,
          },
          activeVariant,
        )
      }
    },
    [activeQueryId, activeModel, activeVariant, multiSelect, updateDraft],
  )

  const handleMultiSelectToggle = useCallback(
    (checked: boolean) => {
      setMultiSelect(checked)
      if (!checked && activeQueryId !== null) {
        // Collapse a multi answer down to its first rank for single mode.
        const current = getDraft(activeQueryId, activeModel, activeVariant)
        if (current.selectedRanks && current.selectedRanks.length > 0) {
          updateDraft(
            activeQueryId,
            activeModel,
            {
              correctRank: current.selectedRanks[0],
              selectedRanks: undefined,
            },
            activeVariant,
          )
        }
      }
    },
    [activeQueryId, activeModel, activeVariant, getDraft, updateDraft],
  )

  const handleNotesChange = useCallback(
    (notes: string) => {
      if (activeQueryId === null) return
      updateDraft(activeQueryId, activeModel, { notes }, activeVariant)
      if (notes.trim()) setSkipNeedsNote(false)
    },
    [activeQueryId, activeModel, activeVariant, updateDraft],
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
    await submitFeedback(activeQueryId, activeModel, predictions, activeVariant)
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [
    activeQueryId,
    activeModel,
    activeVariant,
    predictions,
    submitFeedback,
    advanceToNextActionable,
  ])

  const handleSkip = useCallback(async () => {
    if (activeQueryId === null) return
    const notes = draft?.notes.trim() ?? ''
    if (!notes) {
      setSkipNeedsNote(true)
      return
    }
    await skipFeedback(activeQueryId, activeModel, notes, activeVariant)
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [
    activeQueryId,
    activeModel,
    activeVariant,
    draft?.notes,
    skipFeedback,
    advanceToNextActionable,
  ])

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
  const attribution =
    latest !== null && draftKey !== null && latest.key === draftKey
      ? formatNoteAttribution(latest.entry)
      : null

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

      {/* Whose note the panel is showing. Notes are shared across the team, so
          this often credits another reviewer; when a local unsaved draft wins
          the box, say so rather than letting the credit read as the content.
          Both conditions are needed: !showSeededHint alone is also true right
          after a save, when submitFeedback has deleted the draft and the panel
          is still mounted for the advance timeout. Claiming an unsaved draft is
          on screen, over an empty box, under someone else's name, is the last
          thing a reviewer who just saved should see. */}
      {attribution ? (
        <p
          data-testid="note-attribution"
          className="text-xs font-ui text-stone-500 dark:text-stone-400"
        >
          {attribution}
          {!showSeededHint && !isFeedbackDraftEmpty(draft ?? undefined) && (
            <span className="italic"> (your unsaved draft is shown below)</span>
          )}
        </p>
      ) : (
        // Reachable only if the author has neither a login name nor a display
        // name. `reviewer` is NOT NULL, so this needs a row stored with an
        // empty string; kept as a cheap floor rather than rendering a bare
        // prefilled box with no explanation at all.
        showSeededHint && (
          <p className="text-xs font-ui italic text-stone-400 dark:text-stone-500">
            Prefilled from the last submitted review
          </p>
        )
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
