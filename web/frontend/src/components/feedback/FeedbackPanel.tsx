import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useFeedback } from '../../contexts/FeedbackContext'
import {
  isFeedbackDraftEmpty,
  draftsEqual,
  type FeedbackDraft,
} from '../../contexts/feedbackDraft'
import {
  unconfirmedRanks as unconfirmedRanksOf,
  type AssessmentCandidate,
  type EvidenceBlock,
  type NoneBlock,
  type SelectionIssue,
} from '../../contexts/assessmentEligibility'
import { useReviewer } from '../../contexts/ReviewerContext'
import { fetchNextQuery } from '../../api/queries'
import { usePredictionState } from '../../contexts/PredictionContext'
import { toApiErrorInfo } from '../../api/client'
import { fetchLatestFeedback, type FeedbackEntry } from '../../api/feedback'
import MatchPills from './MatchPills'
import NotesTextarea from './NotesTextarea'
import { formatNoteAttribution } from './noteAttribution'
import SubmitButton from './SubmitButton'

/**
 * A saved review, restored as a draft.
 *
 * Only the canonical first choice has a recorded directory (`correct_dir`), so
 * only it can be restored as a deliberate identified answer. Every other
 * selected rank comes back without one and is offered for reconfirmation: a
 * rank is not proof of what stood at that rank when it was pressed.
 *
 * `legacy_unresolved` rows are readable history, never a prefilled answer: they
 * predate outcomes and saying "None" on their behalf would invent a decision
 * nobody made.
 */
function draftFromEntry(entry: FeedbackEntry): FeedbackDraft {
  const notes = entry.notes ?? ''
  if (entry.outcome === 'matched_rank') {
    const ranks =
      entry.selected_ranks && entry.selected_ranks.length > 0
        ? entry.selected_ranks
        : entry.correct_rank !== null && entry.correct_rank > 0
          ? [entry.correct_rank]
          : []
    if (ranks.length === 0) return { correctRank: null, notes }
    return {
      correctRank: ranks[0],
      selectedRanks: ranks,
      selections: ranks.map((rank, index) => ({
        rank,
        dirName: index === 0 ? entry.correct_dir : null,
        source: null,
      })),
      notes,
    }
  }
  if (entry.outcome === 'none_of_top_k') {
    return { correctRank: 0, notes }
  }
  // skipped / legacy_unresolved -> the prose only, with nothing pressed.
  return { correctRank: null, notes }
}

/** Why no evaluation can be recorded, in the reviewer's terms. */
function evidenceCopy(block: EvidenceBlock): { title: string; detail: string } {
  if (block === 'no_reviewer') {
    return {
      title: 'Not signed in',
      detail:
        'An assessment is recorded against your account, so sign in before answering.',
    }
  }
  if (block === 'no_query') {
    return {
      title: 'No document selected',
      detail: 'Choose a query to assess.',
    }
  }
  if (block === 'loading') {
    return {
      title: 'Loading the ranking',
      detail:
        'The candidates are not on screen yet. Notes you type are kept as a draft.',
    }
  }
  if (block === 'failed') {
    return {
      title: 'The ranking did not load',
      detail:
        'There is nothing here to assess, and that is a failure to load, not a verdict on this document. Your draft is kept; retry the ranking, or skip with a note.',
    }
  }
  if (block === 'excluded') {
    return {
      title: 'Not assessable for this model',
      detail:
        'The retrieval run excluded this document, so this model produced no candidates to be right or wrong about. Skip with a note if it needs following up.',
    }
  }
  if (block === 'empty') {
    return {
      title: 'No ranking came back',
      detail:
        'The server answered with no candidates and no reason recorded. Nothing here can be assessed. Skip with a note if it needs following up.',
    }
  }
  return {
    title: 'No readable candidate evidence',
    detail:
      'This ranking has no candidate whose text is readable in this deployment, so there is nothing to judge it against. Skip with a note if it needs following up.',
  }
}

function noneCopy(block: NoneBlock | null): string | null {
  if (block === null) return null
  if (block === 'partial_model_evidence') {
    return 'Some model candidates have no readable text in this deployment. Rejecting all of them would claim you had read text that is not here, so "None" is unavailable; a readable candidate can still be chosen, and a note plus Skip records the problem.'
  }
  return null
}

function issueCopy(issue: SelectionIssue): string {
  if (issue.kind === 'vanished') {
    return `Rank #${issue.rank}${
      issue.dirName ? ` (${issue.dirName})` : ''
    } is no longer offered for this document, so that choice was removed. Nothing has been saved.`
  }
  if (issue.kind === 'reassigned') {
    return `Rank #${issue.rank} now holds ${issue.nowDirName}, not ${issue.dirName}, so that choice was removed rather than moved to a different directory.`
  }
  if (issue.kind === 'unreadable') {
    return `Rank #${issue.rank} (${issue.dirName}) has no readable text in this deployment, so it cannot be submitted as an answer.`
  }
  return `Rank #${issue.rank} was restored from an earlier draft with no directory recorded. Select it again to confirm the directory now shown there.`
}

export default function FeedbackPanel() {
  const { activeQueryId, activeModel, activeVariant, setActiveQueryId } = useApp()
  const {
    draft,
    draftKey,
    drafts,
    accountId,
    makeDraftKey,
    legacyDraft,
    adoptLegacyNotes,
    evidence,
    selectionReview,
    readiness,
    issues,
    setNotes,
    toggleCandidate,
    setNone,
    collapseToFirstChoice,
    seedDraftIfEmpty,
    submitFeedback,
    skipFeedback,
  } = useFeedback()
  const { reviewerName, clearReviewer } = useReviewer()
  // The candidates, their eligibility and the answer's validity all come from
  // the SHARED prediction state (issue #156) by way of the feedback context
  // (issue #157). The panel renders that judgement; it does not make its own,
  // and it never passes a candidate array into a save.
  const { refresh } = usePredictionState()
  const [skipNeedsNote, setSkipNeedsNote] = useState(false)
  // null = follow the draft. A restored multi answer must stay in multi-select
  // mode, or the next click silently replaces choices the reviewer still holds.
  const [multiSelectOverride, setMultiSelectOverride] = useState<boolean | null>(null)
  const [seeded, setSeeded] = useState<{ key: string; draft: FeedbackDraft } | null>(
    null,
  )
  // The shared latest review for the query on screen, kept whether or not it
  // seeded the form: a local unsaved draft wins the box, but the reviewer still
  // needs to know who last recorded something here (issue #96).
  const [latest, setLatest] = useState<{ key: string; entry: FeedbackEntry } | null>(
    null,
  )
  const [saveError, setSaveError] = useState<{ message: string; stale: boolean } | null>(
    null,
  )

  // Always-current view of the draft map for use inside async callbacks.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  const selectedRanks = useMemo(
    () => selectionReview.confirmed.map((selection) => selection.rank),
    [selectionReview.confirmed],
  )
  const unconfirmedRanks = useMemo(
    () => unconfirmedRanksOf(selectionReview),
    [selectionReview],
  )
  const multiSelect = multiSelectOverride ?? selectionReview.selections.length > 1

  // Multi-select is off by default for a new assessment; a variant switch swaps
  // in a different draft, so it resets there too.
  useEffect(() => {
    setMultiSelectOverride(null)
    setSaveError(null)
    setSkipNeedsNote(false)
  }, [activeQueryId, activeModel, activeVariant, accountId])

  // Seed notes/selection from the last submitted feedback for this query --
  // from ANY reviewer for the note, from the caller alone for the decision
  // (issue #96) -- but never clobber a local unsaved draft.
  useEffect(() => {
    if (activeQueryId === null || !activeModel || accountId === null) return
    const key = makeDraftKey(activeQueryId, activeModel, activeVariant, accountId)
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
      })
      .catch(() => {
        // Best-effort: a failed latest-feedback fetch just leaves the form empty.
      })
    return () => {
      cancelled = true
    }
  }, [
    accountId,
    activeQueryId,
    activeModel,
    activeVariant,
    makeDraftKey,
    seedDraftIfEmpty,
  ])

  const handleToggleCandidate = useCallback(
    (candidate: AssessmentCandidate) => {
      setSaveError(null)
      toggleCandidate(candidate, multiSelect)
    },
    [multiSelect, toggleCandidate],
  )

  const handleToggleNone = useCallback(() => {
    setSaveError(null)
    setNone(!(draft.correctRank === 0))
  }, [draft.correctRank, setNone])

  const handleMultiSelectToggle = useCallback(
    (checked: boolean) => {
      setMultiSelectOverride(checked)
      if (!checked) collapseToFirstChoice()
    },
    [collapseToFirstChoice],
  )

  const handleNotesChange = useCallback(
    (notes: string) => {
      setNotes(notes)
      if (notes.trim()) setSkipNeedsNote(false)
    },
    [setNotes],
  )

  const advanceToNextActionable = useCallback(
    async (after: number) => {
      const next = await fetchNextQuery(after)
      setActiveQueryId(next.file_id)
    },
    [setActiveQueryId],
  )

  // Save failures keep the draft. A changed candidate identity is called out
  // separately, because the answer to it is to look again rather than retry.
  const reportSaveFailure = useCallback((err: unknown) => {
    const info = toApiErrorInfo(err, 'The assessment could not be saved.')
    setSaveError({
      message: info.message,
      stale: info.code === 'CANDIDATE_IDENTITY_CHANGED',
    })
  }, [])

  const handleSubmit = useCallback(async () => {
    if (activeQueryId === null) return
    setSaveError(null)
    try {
      await submitFeedback()
    } catch (err) {
      reportSaveFailure(err)
      return
    }
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [activeQueryId, advanceToNextActionable, reportSaveFailure, submitFeedback])

  const handleSkip = useCallback(async () => {
    if (activeQueryId === null) return
    if (!draft.notes.trim()) {
      setSkipNeedsNote(true)
      return
    }
    setSaveError(null)
    try {
      await skipFeedback()
    } catch (err) {
      reportSaveFailure(err)
      return
    }
    setTimeout(() => {
      void advanceToNextActionable(activeQueryId)
    }, 500)
  }, [
    activeQueryId,
    advanceToNextActionable,
    draft.notes,
    reportSaveFailure,
    skipFeedback,
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

  const noneSelected = draft.correctRank === 0
  const showSeededHint =
    seeded !== null &&
    draftKey !== null &&
    seeded.key === draftKey &&
    draftsEqual(draft, seeded.draft)
  const attribution =
    latest !== null && draftKey !== null && latest.key === draftKey
      ? formatNoteAttribution(latest.entry)
      : null
  const noneNotice = noneCopy(evidence.noneBlock)
  const legacyNote = legacyDraft?.notes.trim() ?? ''

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
            disabled={!evidence.canEvaluate}
            onChange={(event) => handleMultiSelectToggle(event.target.checked)}
            className="h-3.5 w-3.5 rounded border-stone-300 dark:border-stone-600 text-accent focus:ring-2 focus:ring-accent/40 disabled:opacity-50"
          />
          Select multiple
        </label>
      </div>

      {/* No candidates means no controls. The old panel drew ten pills over a
          failed request and let Submit post "none of top 10" against a ranking
          nobody had ever seen. */}
      {evidence.candidates.length > 0 && (
        <MatchPills
          candidates={evidence.candidates}
          canEvaluate={evidence.canEvaluate}
          selectedRanks={selectedRanks}
          unconfirmedRanks={unconfirmedRanks}
          noneSelected={noneSelected}
          noneAvailable={evidence.noneAvailable}
          modelCandidateCount={evidence.modelCandidates.length}
          onToggleCandidate={handleToggleCandidate}
          onToggleNone={handleToggleNone}
        />
      )}

      {/* Why nothing here can be assessed. Rendered whenever the evidence
          refuses evaluation, not only when there is nothing to draw: a ranking
          whose model candidates are all unreadable still has pills on screen,
          and a disabled Submit with no explanation reads as a broken app. */}
      {evidence.evidenceBlock !== null && (
        <div
          role="status"
          data-testid="assessment-unavailable"
          className="rounded-lg border border-stone-300 dark:border-stone-600 px-2.5 py-2"
        >
          <div className="text-xs font-bold uppercase tracking-wide text-stone-500 dark:text-stone-300">
            {evidenceCopy(evidence.evidenceBlock).title}
          </div>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            {evidenceCopy(evidence.evidenceBlock).detail}
          </p>
        </div>
      )}

      {/* What changed under a draft, and what is still missing. Never silent:
          a choice that lost its meaning is reported, not quietly re-pointed. */}
      {(issues.length > 0 || noneNotice !== null) && (
        <div
          role="status"
          data-testid="assessment-notice"
          className="rounded-lg border border-amber-400/60 bg-amber-50/70 dark:bg-amber-500/10 px-2.5 py-2 flex flex-col gap-1"
        >
          {noneNotice !== null && (
            <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
              {noneNotice}
            </p>
          )}
          {issues.map((issue) => (
            <p
              key={`${issue.kind}-${issue.rank}`}
              className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200"
            >
              {issueCopy(issue)}
            </p>
          ))}
        </div>
      )}

      {/* Multi-select records every choice but files the document under one
          directory, the first one clicked. A reviewer picking two sources
          otherwise has no way to know which of them the assignment follows. */}
      {selectionReview.confirmed.length > 1 && (
        <p
          data-testid="canonical-choice-note"
          className="font-ui text-xs leading-snug text-stone-500 dark:text-stone-400"
        >
          Your first choice, #{selectionReview.confirmed[0].rank}, is the one this
          document is filed under. The others are recorded alongside it.
        </p>
      )}

      {saveError !== null && (
        <div
          role="alert"
          data-testid="assessment-save-error"
          className="rounded-lg border border-incorrect/50 bg-incorrect/10 px-2.5 py-2"
        >
          <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
            {saveError.message}
          </p>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            Your draft is still here. Nothing was saved.
          </p>
          {saveError.stale && (
            <button
              type="button"
              data-testid="assessment-refresh-ranking"
              onClick={() => {
                setSaveError(null)
                refresh()
              }}
              className="mt-2 rounded-md border border-stone-300 dark:border-stone-600 px-3 py-1 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
            >
              Reload the ranking
            </button>
          )}
        </div>
      )}

      {/* Whose note the panel is showing. Notes are shared across the team, so
          this often credits another reviewer; when a local unsaved draft wins
          the box, say so rather than letting the credit read as the content.
          Both conditions are needed: !showSeededHint alone is also true right
          after a save, when the draft has been deleted and the panel is still
          mounted for the advance timeout. */}
      {attribution ? (
        <p
          data-testid="note-attribution"
          className="text-xs font-ui text-stone-500 dark:text-stone-400"
        >
          {attribution}
          {!showSeededHint && !isFeedbackDraftEmpty(draft) && (
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

      <NotesTextarea value={draft.notes} onChange={handleNotesChange} />

      {/* Unsent prose from before drafts had an owner. It is not deleted and it
          is not adopted as this account's answer; only its words can be taken,
          deliberately, and only into the note. */}
      {legacyNote !== '' && draftKey !== null && (
        <div
          data-testid="legacy-draft-notice"
          className="rounded-lg border border-stone-300 dark:border-stone-600 px-2.5 py-2 flex flex-col gap-1"
        >
          <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            This browser holds an unsent note for this document from before
            drafts were kept per reviewer. It is not attributed to anyone and has
            not been saved.
          </p>
          <p className="font-ui text-xs italic leading-snug text-stone-500 dark:text-stone-400">
            {legacyNote}
          </p>
          <button
            type="button"
            data-testid="adopt-legacy-note"
            onClick={adoptLegacyNotes}
            className="self-start rounded-md border border-stone-300 dark:border-stone-600 px-3 py-1 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
          >
            Copy this text into my note
          </button>
        </div>
      )}

      {(skipNeedsNote || !draft.notes.trim()) && (
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
        // Submit records an evaluation, so it needs a deliberate valid choice
        // on a usable current ranking. Skip is a deferral with a note and stays
        // available even when nothing loaded -- that is the whole point of it.
        disabled={!readiness.canSubmit}
        skipDisabled={!activeModel || draftKey === null}
      />
    </div>
  )
}
