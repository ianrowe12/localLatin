import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import {
  useFeedback,
  type SaveAcknowledgement,
  type SaveOutcome,
} from '../../contexts/FeedbackContext'
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
import {
  classifySaveFailure,
  isIdentityRejection,
  type SaveFailure,
} from '../../contexts/saveFailure'
import { fetchLatestFeedback, type FeedbackEntry } from '../../api/feedback'
import { toApiErrorInfo } from '../../api/client'
import MatchPills from './MatchPills'
import NotesTextarea from './NotesTextarea'
import { formatNoteAttribution } from './noteAttribution'
import SubmitButton, { type SubmitActionResult } from './SubmitButton'

/**
 * How long the acknowledgement stays on screen before the next document is
 * looked up. Unchanged from the original panel; what is new is that every
 * boundary after it rechecks who is on screen and whether the draft has moved.
 */
const ADVANCE_DELAY_MS = 500

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
    visit,
    getVisit,
    hasNewerDraft,
    getDraftRevision,
    getSaveEpoch,
    pendingSave,
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
  /**
   * What the last acknowledged save on this screen did, and what happened next.
   *
   * Deliberately separate from `saveError`: the write and the move to the next
   * document are two operations, and a failed lookup after a committed row is
   * not a failed save. Saying otherwise sends a reviewer to press Submit again,
   * into an append-only log (issue #158).
   */
  const [saved, setSaved] = useState<{
    visitId: number
    key: string
    /** The revision that was sent, so newer work is recognised at any moment. */
    revision: number
    kind: 'submit' | 'skip'
    /**
     * `superseded` is newer work seen at the moment the receipt arrived;
     * `held` is newer work typed after it, in the pause before the move. Both
     * stay put, and the two read differently to a reviewer.
     */
    navigation: 'pending' | 'superseded' | 'failed' | 'held' | 'done'
    navigationMessage: string | null
  } | null>(null)

  // Always-current view of the draft map for use inside async callbacks.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  // The acknowledged save whose move to the next document is still owed, so a
  // retry can repeat the lookup and never the write.
  const owedAdvanceRef = useRef<SaveAcknowledgement | null>(null)
  // The advance runs from a timer, so by the time it fires -- and again by the
  // time it answers -- the panel may be gone and the reviewer may be assessing
  // something else entirely. Both are read at the moment of the check rather
  // than closed over (issue #171).
  const mountedRef = useRef(true)
  const advanceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearAdvanceTimer = useCallback(() => {
    if (advanceTimerRef.current !== null) {
      clearTimeout(advanceTimerRef.current)
      advanceTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      clearAdvanceTimer()
    }
  }, [clearAdvanceTimer])

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
  // in a different draft, so it resets there too. A save's acknowledgement and
  // its owed navigation belong to the assessment that produced them and leave
  // with it, timer included -- which is what stops a read from one visit
  // speaking over another (issue #171).
  useEffect(() => {
    setMultiSelectOverride(null)
    setSaveError(null)
    setSkipNeedsNote(false)
    setSaved(null)
    owedAdvanceRef.current = null
    clearAdvanceTimer()
  }, [activeQueryId, activeModel, activeVariant, accountId, clearAdvanceTimer])

  // Seed notes/selection from the last submitted feedback for this query --
  // from ANY reviewer for the note, from the caller alone for the decision
  // (issue #96) -- but never clobber a local unsaved draft.
  useEffect(() => {
    if (activeQueryId === null || !activeModel || accountId === null) return
    const key = makeDraftKey(activeQueryId, activeModel, activeVariant, accountId)
    // A save of THIS assessment while the request is out makes its answer older
    // than the draft map: applying it would put the just-saved text back in an
    // empty box and present it as unsent. A save of another assessment says
    // nothing about this one, so it must not silence a colleague's shared note
    // here (issue #158).
    const epochAtRequest = getSaveEpoch(key)
    // What the box held when this was asked for, so a deliberate deletion made
    // since is not overwritten by an answer that predates it.
    const revisionAtRequest = getDraftRevision(key)
    let cancelled = false
    fetchLatestFeedback(activeQueryId, activeModel, activeVariant)
      .then((entry) => {
        if (cancelled || getSaveEpoch(key) !== epochAtRequest) return
        // Attribution is not seeding: who last wrote about this document is
        // still true even when the box may not be filled from it.
        setLatest(entry === null ? null : { key, entry })
        if (entry === null) return
        if (!isFeedbackDraftEmpty(draftsRef.current.get(key))) return
        if (getDraftRevision(key) > revisionAtRequest) return
        const seededDraft = draftFromEntry(entry)
        seedDraftIfEmpty(
          activeQueryId,
          activeModel,
          seededDraft,
          activeVariant,
          revisionAtRequest,
        )
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
    getDraftRevision,
    getSaveEpoch,
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

  /**
   * Whether an operation that started on this screen may still change it.
   *
   * Both halves matter. The key catches another reviewer, query, model or
   * variant; the visit id catches a return to the same assessment, where the
   * key matches again but the reviewer has come back deliberately and must not
   * be thrown forward by a save they made before they left. A view change
   * counts too, because the provider scopes a visit to the review screen.
   *
   * Every handler that touches panel state -- success, failure, timers, owed
   * navigation -- asks this FIRST. Checking only when a replacement timer
   * eventually fires is too late: by then an old assessment's completion has
   * already cancelled the current one's move or replaced its recovery.
   */
  const ownsScreen = useCallback(
    (operation: { key: string; visitId: number }): boolean => {
      if (!mountedRef.current) return false
      const current = getVisit()
      return current.key === operation.key && current.id === operation.visitId
    },
    [getVisit],
  )

  const advanceToNextActionable = useCallback(
    async (ack: SaveAcknowledgement) => {
      // Runs from a timer nothing cancels, on a promise the caller discards. So
      // it must never reject, and it must never act on behalf of an assessment
      // that is no longer the one in front of the reviewer: `ack` names the
      // visit that asked for this read, and every step below is conditional on
      // that visit still being the current one. The test is repeated after the
      // await because the reviewer can move, sign out or switch model while the
      // request is in flight (issue #171).
      if (!ownsScreen(ack)) return
      // Typing after the save, or while the lookup is out, keeps the reviewer
      // where they are: the newer draft is unsent, and navigating would hide it.
      if (hasNewerDraft(ack.key, ack.revision)) {
        setSaved((prev) =>
          prev === null || prev.visitId !== ack.visitId
            ? prev
            : { ...prev, navigation: 'held', navigationMessage: null },
        )
        return
      }
      setSaved((prev) =>
        prev === null || prev.visitId !== ack.visitId
          ? prev
          : { ...prev, navigation: 'pending', navigationMessage: null },
      )
      let payload: unknown
      try {
        payload = await fetchNextQuery(ack.queryId)
      } catch (err) {
        if (!ownsScreen(ack)) return
        const info = toApiErrorInfo(err, 'The next document could not be looked up.')
        setSaved((prev) =>
          prev === null || prev.visitId !== ack.visitId
            ? prev
            : { ...prev, navigation: 'failed', navigationMessage: info.message },
        )
        return
      }
      if (!ownsScreen(ack)) return
      if (hasNewerDraft(ack.key, ack.revision)) {
        setSaved((prev) =>
          prev === null || prev.visitId !== ack.visitId
            ? prev
            : { ...prev, navigation: 'held', navigationMessage: null },
        )
        return
      }
      const fileId = readNextQueryId(payload)
      if (fileId === undefined) {
        // A 200 this app cannot read is a failed move, not an empty queue and
        // not a document. Same channel as any other failed read: reading
        // `file_id` off it used to throw outside every catch block (issue #171).
        setSaved((prev) =>
          prev === null || prev.visitId !== ack.visitId
            ? prev
            : {
                ...prev,
                navigation: 'failed',
                navigationMessage: 'The server sent a response this app could not read.',
              },
        )
        return
      }
      owedAdvanceRef.current = null
      // `null` means there is nothing left to review, which is an answer.
      setActiveQueryId(fileId)
    },
    [hasNewerDraft, ownsScreen, setActiveQueryId],
  )

  const acknowledgeSave = useCallback(
    (ack: SaveAcknowledgement) => {
      // Ownership first, before anything is cancelled or replaced. A save made
      // on query A that lands while B is on screen must not cancel B's timer,
      // take away B's "find the next document" recovery, or claim B's success.
      if (!ownsScreen(ack)) return
      owedAdvanceRef.current = ack
      clearAdvanceTimer()
      const held = ack.supersededByNewerDraft
      setSaved({
        visitId: ack.visitId,
        key: ack.key,
        revision: ack.revision,
        kind: ack.kind,
        navigation: held ? 'superseded' : 'pending',
        navigationMessage: null,
      })
      if (held) return
      advanceTimerRef.current = setTimeout(() => {
        advanceTimerRef.current = null
        void advanceToNextActionable(ack)
      }, ADVANCE_DELAY_MS)
    },
    [advanceToNextActionable, clearAdvanceTimer, ownsScreen],
  )

  /**
   * The same read, asked for again by hand. It sends nothing and saves nothing,
   * and it continues from the assessment whose save is still owed a move rather
   * than from whatever happens to be on screen -- which, because the notice
   * only renders for the current visit, are the same thing.
   */
  const retryAdvance = useCallback(() => {
    const ack = owedAdvanceRef.current
    if (ack === null) return
    void advanceToNextActionable(ack)
  }, [advanceToNextActionable])

  // Save failures keep the draft. What they are allowed to PROMISE differs:
  // only a local refusal or a server rejection is known not to have written
  // anything (see saveFailure.ts). Whose failure it is matters just as much:
  // an assessment the reviewer has left cannot post its error onto the one in
  // front of them.
  const reportSaveFailure = useCallback(
    (err: unknown, operation: { key: string; visitId: number }) => {
      if (!ownsScreen(operation)) return
      setSaveError(classifySaveFailure(err, 'The assessment could not be saved.'))
    },
    [ownsScreen],
  )

  /**
   * Submit, and report to the button what the SERVER said.
   *
   * `{ ok: true }` means an acknowledged receipt for this request on the visit
   * that made it, nothing more; whether the screen then moves on is decided
   * separately below. Returning nothing (an assessment already saving, or a
   * completion that belongs to an assessment the reviewer has left) is not an
   * acknowledgement, and a rejection is never dressed as one.
   */
  const handleSubmit = useCallback(async (): Promise<SubmitActionResult> => {
    const started = getVisit()
    if (started.key === null) return
    const operation = { key: started.key, visitId: started.id }
    setSaveError(null)
    setSaved(null)
    let outcome: SaveOutcome
    try {
      outcome = await submitFeedback()
    } catch (err) {
      reportSaveFailure(err, operation)
      return { ok: false }
    }
    if (outcome.status !== 'saved' || outcome.visitEnded) return
    acknowledgeSave(outcome)
    return { ok: true }
  }, [acknowledgeSave, getVisit, reportSaveFailure, submitFeedback])

  const handleSkip = useCallback(async (): Promise<SubmitActionResult> => {
    if (!draft.notes.trim()) {
      // A deliberate deferral still needs its note. Refused here, with its own
      // explanation already on screen, so this is not a save failure.
      setSkipNeedsNote(true)
      return
    }
    const started = getVisit()
    if (started.key === null) return
    const operation = { key: started.key, visitId: started.id }
    setSaveError(null)
    setSaved(null)
    let outcome: SaveOutcome
    try {
      outcome = await skipFeedback()
    } catch (err) {
      reportSaveFailure(err, operation)
      return { ok: false }
    }
    if (outcome.status !== 'saved' || outcome.visitEnded) return
    acknowledgeSave(outcome)
    return { ok: true }
  }, [acknowledgeSave, draft.notes, getVisit, reportSaveFailure, skipFeedback])

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
  // rendered, whatever state still holds it (issue #171).
  const savedNotice = saved === null || saved.visitId !== visit.id ? null : saved
  const savedWhat =
    savedNotice === null
      ? ''
      : savedNotice.kind === 'skip'
        ? 'Your skip was saved.'
        : 'Your assessment was saved.'
  /**
   * Whether the box currently holds work newer than the revision that was sent.
   *
   * Read now, not remembered from when the receipt arrived: a reviewer who
   * types while the next-document lookup is out -- or after it has failed --
   * has unsent work, and a notice written a second earlier would tell them
   * nothing here needs saving (issue #158).
   */
  const savedHasNewerWork =
    savedNotice !== null && hasNewerDraft(savedNotice.key, savedNotice.revision)
  const savedNoticeText =
    savedNotice === null
      ? ''
      : savedNotice.navigation === 'failed'
        ? savedHasNewerWork
          ? `${savedWhat} Moving to the next document failed. What was sent is recorded, and the newer draft in this box is still unsent.`
          : `${savedWhat} Moving to the next document failed, so nothing here needs saving again.`
        : !savedHasNewerWork
          ? `${savedWhat} This stayed on the same document, and nothing here needs saving again.`
          : savedNotice.navigation === 'held'
            ? `${savedWhat} You have been editing since, so this stayed on the same document; your newer draft is unsent.`
            : `${savedWhat} You edited this document while it was saving, so what was sent is recorded and the draft in the box is still unsent.`

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

      {/* An acknowledged write, and separately what happened to the move to
          the next document. A lookup that fails after a committed row is not a
          failed save: repeating the save would append a second human decision
          to a log nothing can remove. Rendered only for the assessment it is
          about (see `savedNotice`): "your response was recorded" over
          somebody's unanswered document is a receipt for work nobody did
          (issue #171). */}
      {savedNotice !== null &&
        (savedHasNewerWork ||
          savedNotice.navigation === 'failed' ||
          savedNotice.navigation === 'held' ||
          savedNotice.navigation === 'superseded') && (
          <div
            role="status"
            data-testid="assessment-saved-notice"
            data-navigation={savedNotice.navigation}
            data-superseded={savedHasNewerWork ? 'true' : 'false'}
            className="rounded-lg border border-stone-300 dark:border-stone-600 px-2.5 py-2 flex flex-col gap-1"
          >
            <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
              {savedNoticeText}
            </p>
            {/* The save landed; the move to the next document did not. Said out
                loud because the alternative is what this used to do -- leave
                the reviewer sitting on a document they had just answered with
                no word of why, whose obvious next move is to answer it again
                into an append-only log. Not framed as a save failure, and it
                makes no claim about the next document, because neither would
                be true. */}
            {savedNotice.navigation === 'failed' && (
              <div data-testid="assessment-advance-error" className="flex flex-col gap-1">
                <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
                  Your response was recorded. What failed was moving on to the
                  next document
                  {savedNotice.navigationMessage === null
                    ? '.'
                    : `: ${savedNotice.navigationMessage}`}
                </p>
                <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
                  You are still on the document you just answered, and it does
                  not need answering again. Try the move again, or pick the next
                  document from the list yourself.
                </p>
                <button
                  type="button"
                  data-testid="assessment-retry-advance"
                  onClick={retryAdvance}
                  className="mt-1 self-start rounded-md border border-stone-300 dark:border-stone-600 px-3 py-1 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
                >
                  Find the next document
                </button>
              </div>
            )}
          </div>
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
        // The visit, not just the key: coming back to the same assessment is a
        // fresh pair of controls, with no acknowledgement carried over from the
        // save made before leaving.
        operationKey={`${draftKey ?? 'none'}#${visit.id}`}
        // What the provider says is in flight for THIS assessment. The controls
        // unmount whenever the sidebar collapses or the view changes, so their
        // own state cannot be the record of a request that is still out.
        pending={pendingSave}
      />
    </div>
  )
}
