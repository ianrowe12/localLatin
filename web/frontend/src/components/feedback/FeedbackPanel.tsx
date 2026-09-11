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
import { toApiErrorInfo } from '../../api/client'
import type { PredictionVariant } from '../../api/variants'
import { usePredictionState } from '../../contexts/PredictionContext'
import {
  classifySaveFailure,
  isIdentityRejection,
  type SaveFailure,
} from '../../contexts/saveFailure'
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
      'This ranking has no candidate whose text this view can show you, so there is nothing to judge it against. Skip with a note if it needs following up.',
  }
}

function noneCopy(block: NoneBlock | null): string | null {
  if (block === null) return null
  if (block === 'partial_model_evidence') {
    return 'Some model candidates cannot be read on this screen: their text is missing in this deployment, or it sits in a file this view cannot open. Rejecting all of them would claim you had read words you were never shown, so "None" is unavailable; a readable candidate can still be chosen, and a note plus Skip records the problem.'
  }
  return null
}

/**
 * One visit to one assessment: a document, under a model, a pipeline and an
 * account, entered once.
 *
 * The post-save read that moves the reviewer on is asked for by a particular
 * assessment and is only ever an answer to that one. Matching the four
 * identifying fields is not enough on its own: leaving a document and coming
 * back to it is a new visit with a new saved answer behind it, and a read left
 * over from the first would otherwise be accepted as an answer to the second.
 * `serial` is what separates them -- it moves whenever any of the four change,
 * so returning never resurrects the visit that was left.
 */
interface AssessmentVisit {
  serial: number
  queryId: number | null
  model: string
  variant: PredictionVariant
  accountId: number | null
}

function sameVisit(a: AssessmentVisit, b: AssessmentVisit): boolean {
  return (
    a.serial === b.serial &&
    a.queryId === b.queryId &&
    a.model === b.model &&
    a.variant === b.variant &&
    a.accountId === b.accountId
  )
}

/**
 * The `file_id` a next-document response actually carries, or `undefined` when
 * it carries nothing this app can use.
 *
 * `apiFetch<NextQueryResponse>` is a TypeScript assertion, not a runtime check.
 * A 200 whose body is `null`, or an object with no `file_id`, resolves happily
 * and fails only when something dereferences it -- which, on a promise the
 * caller discards, is an unhandled rejection outside any catch block. A
 * `file_id` of `null` is a real answer from the real route meaning there is
 * nothing left to review, and is kept as exactly that.
 */
function readNextQueryId(payload: unknown): number | null | undefined {
  if (typeof payload !== 'object' || payload === null) return undefined
  const fileId = (payload as Record<string, unknown>).file_id
  if (fileId === null) return null
  if (typeof fileId === 'number' && Number.isInteger(fileId)) return fileId
  return undefined
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
    if (issue.evidence === 'hidden_witness') {
      return `Rank #${issue.rank} (${issue.dirName}) shows a file with no readable text here. Its other files do carry text, but this view cannot open them, so that choice cannot be submitted from this screen.`
    }
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
    hasQuarantinedLegacyDraft,
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
  const [saveError, setSaveError] = useState<SaveFailure | null>(null)
  // Why the reviewer is still looking at the document they just answered, and
  // which visit that sentence is about. A message with no owner is a message
  // that can be shown over somebody else's work.
  const [advanceError, setAdvanceError] = useState<{
    visit: AssessmentVisit
    message: string
  } | null>(null)
  // Bumped by the reset effect below, which already runs on exactly the four
  // changes that end a visit.
  const [visitSerial, setVisitSerial] = useState(0)

  // Always-current view of the draft map for use inside async callbacks.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  const visit = useMemo<AssessmentVisit>(
    () => ({
      serial: visitSerial,
      queryId: activeQueryId,
      model: activeModel,
      variant: activeVariant,
      accountId,
    }),
    [accountId, activeModel, activeQueryId, activeVariant, visitSerial],
  )
  // The advance runs from a 500ms timer that nothing cancels, so by the time it
  // fires -- and again by the time it answers -- the panel may be gone and the
  // reviewer may be assessing something else entirely. Both are read at the
  // moment of the check rather than closed over.
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const visitRef = useRef(visit)
  visitRef.current = visit
  const ownsCurrentAssessment = useCallback(
    (owner: AssessmentVisit) => mounted.current && sameVisit(owner, visitRef.current),
    [],
  )

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
  // in a different draft, so it resets there too. The serial bump makes the
  // same four changes end the visit any in-flight advance belongs to; it is
  // bookkeeping for that ownership test, NOT the test itself, which every user
  // of a visit performs for itself.
  useEffect(() => {
    setMultiSelectOverride(null)
    setSaveError(null)
    setSkipNeedsNote(false)
    setVisitSerial((serial) => serial + 1)
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
    async (owner: AssessmentVisit) => {
      // Runs 500ms after a save, from a timer nothing cancels, on a promise the
      // caller discards. So it must never reject, and it must never act on
      // behalf of an assessment that is no longer the one in front of the
      // reviewer: `owner` is the visit that asked for this read, and every step
      // below is conditional on that visit still being the current one. The
      // test is repeated after the await because the reviewer can move, sign
      // out or switch model while the request is in flight.
      if (!ownsCurrentAssessment(owner) || owner.queryId === null) return
      setAdvanceError(null)

      let payload: unknown
      try {
        payload = await fetchNextQuery(owner.queryId)
      } catch (err) {
        if (!ownsCurrentAssessment(owner)) return
        // Visible, and honest about what did and did not happen: the
        // assessment is recorded, the move is what failed.
        setAdvanceError({
          visit: owner,
          message: toApiErrorInfo(err, 'Could not reach the server.').message,
        })
        return
      }

      if (!ownsCurrentAssessment(owner)) return
      const fileId = readNextQueryId(payload)
      if (fileId === undefined) {
        // A 200 this app cannot read is a failed move, not an empty queue and
        // not a document. Same channel as any other failed read.
        setAdvanceError({
          visit: owner,
          message: 'The server sent a response this app could not read.',
        })
        return
      }
      // `null` means there is nothing left to review, which is an answer.
      setActiveQueryId(fileId)
    },
    [ownsCurrentAssessment, setActiveQueryId],
  )

  /**
   * The same read, asked for again by hand. It sends nothing and saves nothing,
   * and it continues from the assessment the notice is about rather than from
   * whatever happens to be on screen -- which, because the notice only renders
   * for the current visit, are the same thing.
   */
  const retryAdvance = useCallback(() => {
    if (advanceError === null) return
    void advanceToNextActionable(advanceError.visit)
  }, [advanceError, advanceToNextActionable])

  // Save failures keep the draft. What they are allowed to PROMISE differs:
  // only a local refusal or a server rejection is known not to have written
  // anything (see saveFailure.ts).
  const reportSaveFailure = useCallback((err: unknown) => {
    setSaveError(classifySaveFailure(err, 'The assessment could not be saved.'))
  }, [])

  const handleSubmit = useCallback(async () => {
    if (activeQueryId === null) return
    setSaveError(null)
    // The assessment this save is for, named before the save leaves, so the
    // read that follows answers to it and not to wherever the reviewer has
    // got to by the time it runs.
    const owner = visit
    try {
      await submitFeedback()
    } catch (err) {
      reportSaveFailure(err)
      return
    }
    setTimeout(() => {
      void advanceToNextActionable(owner)
    }, 500)
  }, [activeQueryId, advanceToNextActionable, reportSaveFailure, submitFeedback, visit])

  const handleSkip = useCallback(async () => {
    if (activeQueryId === null) return
    if (!draft.notes.trim()) {
      setSkipNeedsNote(true)
      return
    }
    setSaveError(null)
    const owner = visit
    try {
      await skipFeedback()
    } catch (err) {
      reportSaveFailure(err)
      return
    }
    setTimeout(() => {
      void advanceToNextActionable(owner)
    }, 500)
  }, [
    activeQueryId,
    advanceToNextActionable,
    draft.notes,
    reportSaveFailure,
    skipFeedback,
    visit,
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
  // Ownership decides what is on screen, not just what was computed: a notice
  // left over from an assessment the reviewer has moved on from is never
  // rendered, whatever state still holds it.
  const advanceMessage =
    advanceError !== null && sameVisit(advanceError.visit, visit)
      ? advanceError.message
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
          data-outcome={saveError.outcome}
          className="rounded-lg border border-incorrect/50 bg-incorrect/10 px-2.5 py-2"
        >
          <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
            {saveError.message}
          </p>
          {/* Only a refusal this app made, or one the server made before it
              writes, can promise an empty log. A lost, unreadable or 500
              answer cannot: the feedback row is committed before the
              reviewer-directory membership write, so "nothing was saved" would
              be a guess dressed as a receipt -- and the reviewer would act on
              it by saving again, into an append-only table. */}
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            {saveError.outcome === 'uncertain'
              ? 'This app did not get an answer it can trust, so it cannot tell whether your assessment was recorded. Your draft is kept here. Reload the ranking and check the last review for this document before saving again, because saving again may record it twice.'
              : 'Your draft is still here. Nothing was saved.'}
          </p>
          {isIdentityRejection(saveError) && (
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

      {/* The save landed; the move to the next document did not. Said out loud
          because the alternative is what this used to do -- leave the reviewer
          sitting on a document they had just answered with no word of why,
          whose obvious next move is to answer it again into an append-only
          log. Rendered only for the assessment it is about (see
          `advanceMessage`): "your response was recorded" over somebody's
          unanswered document is a receipt for work nobody did. Not framed as a
          save failure, and it makes no claim about the next document, because
          neither would be true. */}
      {advanceMessage !== null && (
        <div
          role="alert"
          data-testid="assessment-advance-error"
          className="rounded-lg border border-amber-400/60 bg-amber-50/70 dark:bg-amber-500/10 px-2.5 py-2"
        >
          <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
            Your response was recorded. What failed was moving on to the next
            document: {advanceMessage}
          </p>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            You are still on the document you just answered, and it does not
            need answering again. Try the move again, or pick the next document
            from the list yourself.
          </p>
          <button
            type="button"
            data-testid="assessment-advance-retry"
            onClick={retryAdvance}
            className="mt-2 rounded-md border border-stone-300 dark:border-stone-600 px-3 py-1 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
          >
            Try again
          </button>
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

      {/* Unsent prose from before drafts had an owner, kept and not shown.
          The pre-#157 key names a query, a model and a variant but no
          reviewer, so this text could be anyone's; displaying it to, or
          letting it be copied by, whichever account signs in next would be
          disclosure, and read-only display is not consent. It is not deleted
          either: it is somebody's unsent work. Deliberately submitted notes
          are shared on purpose and still appear above, attributed. */}
      {hasQuarantinedLegacyDraft && draftKey !== null && (
        <p
          data-testid="legacy-draft-notice"
          className="rounded-lg border border-stone-300 dark:border-stone-600 px-2.5 py-2 font-ui text-xs leading-snug text-stone-500 dark:text-stone-400"
        >
          This browser still holds unsent draft text for this document from
          before drafts were kept per reviewer. It cannot be attributed to an
          account, so it is not shown here and cannot be copied; it has not been
          saved, changed or deleted.
        </p>
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
