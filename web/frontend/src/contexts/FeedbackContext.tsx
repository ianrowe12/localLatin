import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  FEEDBACK_UPDATED_EVENT,
  submitFeedback as postFeedback,
  type FeedbackEntry,
  type FeedbackPayload,
} from '../api/feedback'
import { DEFAULT_VARIANT, type PredictionVariant } from '../api/variants'
import { useApp } from './AppContext'
import { usePredictionState } from './PredictionContext'
import { useReviewer } from './ReviewerContext'
import {
  assessmentEvidence,
  evaluationReadiness,
  expectedCandidateDirs,
  reviewSelections,
  toggleSelection,
  type AssessmentCandidate,
  type AssessmentEvidence,
  type EvaluationReadiness,
  type SelectionIssue,
  type SelectionReview,
} from './assessmentEligibility'
import {
  deserializeDrafts,
  draftRevision,
  draftSelections,
  draftWithNone,
  draftWithSelections,
  emptyDraft,
  isFeedbackDraftEmpty,
  legacyDraftKey,
  makeDraftKey,
  seedDraftMapIfEmpty,
  serializeDrafts,
  type FeedbackDraft,
} from './feedbackDraft'

export type { DraftSelection, FeedbackDraft } from './feedbackDraft'

/** Which deliberate operation a save is. Never inferred from the draft. */
export type SaveKind = 'submit' | 'skip'

/**
 * One visit to one assessment.
 *
 * `key` alone is not enough: leaving query A for B and coming back is a new
 * visit to the same key, and a completion from the first visit must not be
 * allowed to navigate, clear or acknowledge anything in the second (issue
 * #158). `id` advances every time the assessment on screen changes and is never
 * reused within a session.
 */
export interface AssessmentVisit {
  key: string | null
  id: number
}

/**
 * A save the server acknowledged with a readable receipt for this request.
 *
 * It says nothing about navigation, which happens afterwards and can fail on
 * its own. `supersededByNewerDraft` means the reviewer edited the assessment
 * while the request was in flight: the submitted revision is saved, the newer
 * draft is still unsent, and the caller must not describe what is on screen as
 * saved.
 */
export interface SaveAcknowledgement {
  status: 'saved'
  kind: SaveKind
  key: string
  queryId: number
  visitId: number
  /** The draft revision that was sent. */
  revision: number
  entry: FeedbackEntry
  supersededByNewerDraft: boolean
  /**
   * True when the visit that made this save had already ended by the time the
   * receipt arrived. The write is still a fact -- the row exists -- but nothing
   * on screen was cleared or acknowledged for it, and no caller may present it
   * as the current assessment's success.
   */
  visitEnded: boolean
}

/**
 * The result of asking for a save.
 *
 * `already_pending` is not a failure and not an acknowledgement: an operation
 * for this assessment is already in flight, and this attempt did nothing. A
 * failed save rejects instead, so `classifySaveFailure` can say what is known
 * about whether anything was written.
 */
export type SaveOutcome = SaveAcknowledgement | { status: 'already_pending' }

export interface FeedbackContextValue {
  drafts: Map<string, FeedbackDraft>
  /**
   * The assessment on screen: reviewer account, query, model and variant.
   * Null when there is no query, no model or no signed-in reviewer, in which
   * case nothing can be drafted or saved.
   */
  draftKey: string | null
  accountId: number | null
  makeDraftKey: typeof makeDraftKey
  /** The current draft. Always defined, so callers need no null dance. */
  draft: FeedbackDraft
  getDraft: (
    queryId: number,
    model: string,
    variant?: PredictionVariant,
  ) => FeedbackDraft
  /**
   * Whether this browser still holds an unowned draft for this
   * query/model/variant, written before drafts were scoped to an account.
   *
   * A boolean, not the draft (issue #157 repair). The pre-#157 key carries no
   * reviewer identity, so its prose could belong to anybody who used this
   * browser; showing it to -- or letting it be copied by -- whichever account
   * signs in next is disclosure, and "read-only" is not consent. The entry is
   * kept exactly as stored and never deleted, because it is somebody's unsent
   * work; it is simply not readable from here. Deliberately submitted notes
   * are a different thing entirely and still arrive, attributed, through
   * GET /api/feedback/latest.
   */
  hasQuarantinedLegacyDraft: boolean

  /** What the current ranking supports. The single source for every control. */
  evidence: AssessmentEvidence
  /** The draft's choices measured against that ranking. */
  selectionReview: SelectionReview
  /** Whether Submit may record this draft, and why not. */
  readiness: EvaluationReadiness
  /** Choices dropped or held for reconfirmation, for the reviewer to read. */
  issues: SelectionIssue[]

  updateDraft: (patch: Partial<FeedbackDraft>) => void
  setNotes: (notes: string) => void
  /** Toggle one candidate. Choice order is preserved; None is cleared. */
  toggleCandidate: (candidate: AssessmentCandidate, multiSelect: boolean) => void
  setNone: (none: boolean) => void
  /** Collapse a multi answer to its first (canonical) choice. */
  collapseToFirstChoice: () => void
  seedDraftIfEmpty: (
    queryId: number,
    model: string,
    seed: FeedbackDraft,
    variant?: PredictionVariant,
    /**
     * The draft revision observed when this prefill was requested. Seeding is
     * refused if the draft has moved on since, including to a deliberately
     * emptied box.
     */
    observedRevision?: number,
  ) => void

  /**
   * Save the current deliberate evaluation. Takes no candidates: the payload is
   * built from the ranking on screen, so no caller can supply its own.
   * Rejects, without any request, when the evidence does not support the draft.
   *
   * Resolves with the acknowledged receipt, or with `already_pending` when this
   * assessment already has an operation in flight. Rejects with the save
   * failure otherwise; nothing is retried here.
   */
  submitFeedback: () => Promise<SaveOutcome>
  /** Save a deliberate, note-required deferral. Never inferred, never None. */
  skipFeedback: () => Promise<SaveOutcome>
  /** The operation in flight for the assessment on screen, if any. */
  pendingSave: SaveKind | null
  /** This visit to this assessment, as of this render. */
  visit: AssessmentVisit
  /** This visit to this assessment. Read inside async callbacks, never stored. */
  getVisit: () => AssessmentVisit
  /** True when this key holds an edit newer than the given revision. */
  hasNewerDraft: (key: string, revision: number) => boolean
  /** The current revision of one key's draft, for capturing before a request. */
  getDraftRevision: (key: string) => number
  /**
   * How many saves this provider has recorded for ONE assessment. A caller that
   * started a request before a save and applies its answer afterwards can
   * compare this and drop the answer rather than reseeding a draft the save
   * just cleared. Scoped per key: a save on another assessment says nothing
   * about this one.
   */
  getSaveEpoch: (key: string) => number

  /** The last save this session recorded, for a local acknowledgement. */
  lastSubmission: LastSubmission | null
  /**
   * Put the submitted draft back in the box.
   *
   * Local only: the feedback log is append-only, so nothing here removes the
   * saved row or any reviewer-directory membership it created. Refuses when the
   * key has newer unsent work, which restoring would overwrite.
   */
  restoreSubmittedDraft: () => void
  /** Forget the acknowledgement without touching any draft. */
  dismissLastSubmission: () => void
}

const FeedbackContext = createContext<FeedbackContextValue | null>(null)

const STORAGE_KEY = 'locallatin-feedback-drafts'

/**
 * What the acknowledgement toast is allowed to say and offer.
 *
 * `canRestoreDraft` is false once the reviewer has written something new under
 * the same key: the saved row is unreachable from here either way, and putting
 * the old text back would destroy the new.
 */
export interface LastSubmission {
  key: string
  kind: SaveKind
  queryId: number
  entry: FeedbackEntry
  canRestoreDraft: boolean
  supersededByNewerDraft: boolean
}

/** Refused locally, before any request. */
export class AssessmentNotSavableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AssessmentNotSavableError'
  }
}

interface CurrentAssessment {
  key: string | null
  queryId: number | null
  model: string
  variant: PredictionVariant
  draft: FeedbackDraft
  readiness: EvaluationReadiness
}

export function FeedbackProvider({ children }: { children: ReactNode }) {
  const { activeQueryId, activeModel, activeVariant, currentView } = useApp()
  const { phase, predictions } = usePredictionState()
  const { user } = useReviewer()
  const accountId = user?.id ?? null

  const [drafts, setDrafts] = useState<Map<string, FeedbackDraft>>(() =>
    deserializeDrafts(localStorage.getItem(STORAGE_KEY)),
  )
  const [lastSaved, setLastSaved] = useState<{
    key: string
    kind: SaveKind
    queryId: number
    accountId: number | null
    entry: FeedbackEntry
    draft: FeedbackDraft
    revision: number
  } | null>(null)
  // Which assessments have a request in flight, keyed the same way drafts are,
  // so two different assessments can never block each other and the SAME one
  // can never run twice. Held in the provider rather than in a control: the
  // buttons, the keyboard and any direct context call all pass through here,
  // and a control that unmounts mid-save (a collapsed sidebar, a view change)
  // leaves the lock exactly where it was.
  const inFlightRef = useRef(new Map<string, SaveKind>())
  // The same map as render state. The ref is the lock -- synchronous, so two
  // calls in one tick cannot both pass -- and this is what the controls draw.
  const [pendingOps, setPendingOps] = useState<ReadonlyMap<string, SaveKind>>(
    () => new Map(),
  )
  const publishPending = useCallback(() => {
    setPendingOps(new Map(inFlightRef.current))
  }, [])
  // Choices this provider dropped, kept until the reviewer acts: once the draft
  // is rewritten the reason is gone from the data, and silently discarding
  // somebody's answer is the failure this issue exists to prevent.
  const [droppedIssues, setDroppedIssues] = useState<{
    key: string
    issues: SelectionIssue[]
  } | null>(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, serializeDrafts(drafts))
  }, [drafts])

  // The draft map as it is right now, for async callbacks that must not read a
  // value captured when their request started.
  const draftsRef = useRef(drafts)
  draftsRef.current = drafts

  // Who this browser believes is signed in, read the same way: a save must be
  // confirmed against the account that SENT it, not whoever is signed in when
  // the answer arrives.
  const accountIdRef = useRef(accountId)
  accountIdRef.current = accountId

  /**
   * Next revision number for a key.
   *
   * Kept outside the draft map on purpose: a saved draft is deleted, and if the
   * counter went with it the reviewer's next keystroke would produce revision 1
   * again -- lower than the revision just submitted, so the save's own timer
   * would read the new text as "nothing newer" and navigate away from it.
   */
  const revisionsRef = useRef(new Map<string, number>())
  const nextRevision = useCallback((key: string): number => {
    const seen = Math.max(
      revisionsRef.current.get(key) ?? 0,
      draftRevision(draftsRef.current.get(key)),
    )
    const next = seen + 1
    revisionsRef.current.set(key, next)
    return next
  }, [])

  /**
   * How many saves this provider has recorded for one assessment.
   *
   * Per key, not global: a save on query A says nothing about whether a request
   * already in flight for query B is stale, and invalidating B's answer would
   * silently drop a colleague's shared note from a document nobody saved.
   */
  const saveEpochsRef = useRef(new Map<string, number>())
  const getSaveEpoch = useCallback(
    (key: string) => saveEpochsRef.current.get(key) ?? 0,
    [],
  )

  const draftKey =
    activeQueryId === null || !activeModel || accountId === null
      ? null
      : makeDraftKey(activeQueryId, activeModel, activeVariant, accountId)

  const getDraftRevision = useCallback(
    (key: string) => draftRevision(draftsRef.current.get(key)),
    [],
  )

  /**
   * Which visit to this assessment this is.
   *
   * Computed during render, not in an effect: a completion that lands between
   * the key change and a passive effect would otherwise be judged against the
   * visit it has already left.
   *
   * The scope is wider than the draft key. Leaving the review view and coming
   * back, like leaving query A for B and coming back, is a deliberate return to
   * an assessment the reviewer is looking at afresh; the key is identical, so
   * only the visit id can tell a save made before they left from the screen
   * they are looking at now.
   */
  const visitScope = `${draftKey ?? 'none'}@${currentView}`
  const visitRef = useRef<AssessmentVisit & { scope: string }>({
    key: draftKey,
    id: 1,
    scope: visitScope,
  })
  if (visitRef.current.scope !== visitScope) {
    visitRef.current = {
      key: draftKey,
      id: visitRef.current.id + 1,
      scope: visitScope,
    }
  }
  const visit = visitRef.current
  const getVisit = useCallback(() => visitRef.current, [])

  const hasNewerDraft = useCallback(
    (key: string, revision: number) => draftRevision(draftsRef.current.get(key)) > revision,
    [],
  )

  const draft = (draftKey === null ? undefined : drafts.get(draftKey)) ?? emptyDraft()

  // Presence only. The entry is never read into anything the UI can render or
  // copy, and it is never rewritten or removed: unsent prose that cannot be
  // attributed is nobody's to show and nobody's to delete.
  const hasQuarantinedLegacyDraft =
    activeQueryId !== null &&
    Boolean(activeModel) &&
    !isFeedbackDraftEmpty(
      drafts.get(legacyDraftKey(activeQueryId, activeModel, activeVariant)),
    )

  const evidence = useMemo(
    () =>
      assessmentEvidence({
        phase,
        predictions,
        hasReviewerAccount: accountId !== null,
      }),
    [accountId, phase, predictions],
  )

  // Reconciliation happens against THIS assessment's own settled ranking only.
  // A request that is loading, failed, empty or excluded says nothing about
  // whether a draft is still valid, so the draft passes through untouched.
  const selectionReview = useMemo<SelectionReview>(() => {
    const selections = draftSelections(draft)
    if (evidence.phase !== 'ready') {
      return { selections, confirmed: [], issues: [], changed: false }
    }
    return reviewSelections(selections, evidence)
  }, [draft, evidence])

  const readiness = useMemo(
    () => evaluationReadiness(draft, evidence, selectionReview),
    [draft, evidence, selectionReview],
  )

  // Moving to another assessment drops the notice with it. Declared BEFORE the
  // effect that records one, because React runs effects in declaration order
  // within a commit: the reverse order clears the notice the same tick it is
  // set, on the very first render where a draft is reconciled.
  useEffect(() => {
    setDroppedIssues(null)
  }, [draftKey])

  // Persist a reconciliation that removed something, and remember why.
  useEffect(() => {
    if (draftKey === null || !selectionReview.changed) return
    const dropped = selectionReview.issues.filter(
      (issue) => issue.kind === 'vanished' || issue.kind === 'reassigned',
    )
    setDrafts((prev) => {
      const existing = prev.get(draftKey)
      if (!existing) return prev
      const next = new Map(prev)
      // A reconciliation changes what the draft says, so it is a new revision
      // too: a save that was sent before it must not clear the corrected draft.
      next.set(draftKey, {
        ...draftWithSelections(existing, selectionReview.selections),
        revision: nextRevision(draftKey),
      })
      return next
    })
    setDroppedIssues({ key: draftKey, issues: dropped })
  }, [draftKey, nextRevision, selectionReview])

  const issues = useMemo(() => {
    const remembered =
      droppedIssues !== null && droppedIssues.key === draftKey
        ? droppedIssues.issues
        : []
    return [...remembered, ...selectionReview.issues]
  }, [draftKey, droppedIssues, selectionReview.issues])

  const getDraft = useCallback(
    (
      queryId: number,
      model: string,
      variant: PredictionVariant = DEFAULT_VARIANT,
    ): FeedbackDraft =>
      drafts.get(makeDraftKey(queryId, model, variant, accountId)) ?? emptyDraft(),
    [accountId, drafts],
  )

  const writeDraft = useCallback(
    (update: (existing: FeedbackDraft) => FeedbackDraft) => {
      if (draftKey === null) return
      // Every edit is a new revision, including one that restores earlier text:
      // "the same words" is not "the same draft" to an in-flight save.
      const revision = nextRevision(draftKey)
      setDrafts((prev) => {
        const next = new Map(prev)
        next.set(draftKey, {
          ...update(prev.get(draftKey) ?? emptyDraft()),
          revision,
        })
        return next
      })
    },
    [draftKey, nextRevision],
  )

  const updateDraft = useCallback(
    (patch: Partial<FeedbackDraft>) => {
      writeDraft((existing) => ({ ...existing, ...patch }))
    },
    [writeDraft],
  )

  const setNotes = useCallback(
    (notes: string) => {
      writeDraft((existing) => ({ ...existing, notes }))
    },
    [writeDraft],
  )

  const toggleCandidate = useCallback(
    (candidate: AssessmentCandidate, multiSelect: boolean) => {
      // The evidence, not the caller, decides what can be chosen.
      if (!candidate.usable || !evidence.canEvaluate) return
      setDroppedIssues(null)
      writeDraft((existing) =>
        draftWithSelections(
          existing,
          toggleSelection(draftSelections(existing), candidate, multiSelect),
        ),
      )
    },
    [evidence.canEvaluate, writeDraft],
  )

  const setNone = useCallback(
    (none: boolean) => {
      if (none && !evidence.noneAvailable) return
      setDroppedIssues(null)
      writeDraft((existing) => draftWithNone(existing, none))
    },
    [evidence.noneAvailable, writeDraft],
  )

  const collapseToFirstChoice = useCallback(() => {
    writeDraft((existing) => {
      const selections = draftSelections(existing)
      if (selections.length < 2) return existing
      return draftWithSelections(existing, [selections[0]])
    })
  }, [writeDraft])

  const seedDraftIfEmpty = useCallback(
    (
      queryId: number,
      model: string,
      seed: FeedbackDraft,
      variant: PredictionVariant = DEFAULT_VARIANT,
      observedRevision?: number,
    ) => {
      if (accountId === null) return
      const key = makeDraftKey(queryId, model, variant, accountId)
      // A prefill that arrives after the reviewer has moved on belongs to a key
      // nobody is looking at; it must not resurface as a current answer.
      if (key !== draftKey) return
      // Nor may it overwrite work done since it was asked for. Emptiness is not
      // enough: a reviewer who types a sentence and then deletes it has
      // deliberately left the box empty, and that decision is a later revision
      // than the one this answer was fetched against.
      if (
        observedRevision !== undefined &&
        draftRevision(draftsRef.current.get(key)) > observedRevision
      ) {
        return
      }
      const revision = nextRevision(key)
      setDrafts((prev) => seedDraftMapIfEmpty(prev, key, seed, revision))
    },
    [accountId, draftKey, nextRevision],
  )

  // Everything the save paths read, always at its current value: a handler
  // bound in an earlier render must not save an earlier ranking.
  const currentRef = useRef<CurrentAssessment>({
    key: null,
    queryId: null,
    model: '',
    variant: activeVariant,
    draft,
    readiness,
  })
  currentRef.current = {
    key: draftKey,
    queryId: activeQueryId,
    model: activeModel,
    variant: activeVariant,
    draft,
    readiness,
  }

  /**
   * Record an acknowledged save against the visit that made it.
   *
   * Clears exactly the revision that was sent, and only while the reviewer is
   * still on the visit that sent it. If the reviewer edited the assessment
   * while the request was in flight, the newer draft stays in the box: the save
   * is real, but it saved older text, and deleting the newer one would destroy
   * work the reviewer can see and believes is unsent. If they left the
   * assessment entirely -- another query, model, variant, account or view, or a
   * deliberate return to this one -- the receipt stays a fact about its own
   * operation and touches nothing on screen.
   */
  const recordSubmission = useCallback(
    (
      kind: SaveKind,
      key: string,
      queryId: number,
      owner: number | null,
      visitId: number,
      revision: number,
      snapshot: FeedbackDraft,
      entry: FeedbackEntry,
    ): { applied: boolean; supersededByNewerDraft: boolean } => {
      const superseded = draftRevision(draftsRef.current.get(key)) > revision
      if (visitRef.current.id !== visitId) {
        return { applied: false, supersededByNewerDraft: superseded }
      }
      saveEpochsRef.current.set(key, (saveEpochsRef.current.get(key) ?? 0) + 1)
      setDrafts((prev) => {
        // Re-checked against the authoritative map: this is the write that can
        // destroy a reviewer's text, so it never trusts the mirror alone.
        if (draftRevision(prev.get(key)) > revision) return prev
        if (!prev.has(key)) return prev
        const next = new Map(prev)
        next.delete(key)
        return next
      })
      setLastSaved({
        key,
        kind,
        queryId,
        accountId: owner,
        entry,
        draft: snapshot,
        revision,
      })
      window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
      return { applied: true, supersededByNewerDraft: superseded }
    },
    [],
  )

  /**
   * Run one save for one assessment, or decline because one is already running.
   *
   * The claim is synchronous and happens before the first `await`, so two
   * clicks in the same tick, a click racing the keyboard, and a direct context
   * call from anywhere else all see the same lock. It is a client-side guard
   * against this app duplicating a request; it is not exactly-once delivery,
   * which the API does not offer (see web/SAVE_RETRY_POLICY.md).
   */
  const runSave = useCallback(
    async (
      kind: SaveKind,
      build: (current: CurrentAssessment) => {
        payload: FeedbackPayload
        snapshot: FeedbackDraft
      },
    ): Promise<SaveOutcome> => {
      const current = currentRef.current
      const key = current.key
      const queryId = current.queryId
      if (key === null || queryId === null) {
        throw new AssessmentNotSavableError(
          kind === 'submit'
            ? 'There is no assessment on screen to save.'
            : 'There is no assessment on screen to defer.',
        )
      }
      if (inFlightRef.current.has(key)) return { status: 'already_pending' }

      const { payload, snapshot } = build(current)
      const revision = draftRevision(current.draft)
      const visitId = visitRef.current.id
      // The account as it is NOW, not as it is when the answer arrives: the
      // receipt has to be checked against whoever actually sent the request.
      const owner = accountIdRef.current
      inFlightRef.current.set(key, kind)
      publishPending()

      let entry: FeedbackEntry
      try {
        entry = await postFeedback(payload, owner)
      } finally {
        inFlightRef.current.delete(key)
        publishPending()
      }

      const { applied, supersededByNewerDraft } = recordSubmission(
        kind,
        key,
        queryId,
        owner,
        visitId,
        revision,
        snapshot,
        entry,
      )
      return {
        status: 'saved',
        kind,
        key,
        queryId,
        visitId,
        revision,
        entry,
        supersededByNewerDraft,
        visitEnded: !applied,
      }
    },
    [publishPending, recordSubmission],
  )

  const submitFeedback = useCallback(
    (): Promise<SaveOutcome> =>
      runSave('submit', (current) => {
        if (!current.readiness.canSubmit || current.readiness.outcome === null) {
          throw new AssessmentNotSavableError(
            'This assessment cannot be saved against the ranking on screen.',
          )
        }
        const base = {
          query_id: current.queryId as number,
          model_slug: current.model,
          variant: current.variant,
          notes: current.draft.notes,
          // Never an assignment authority server-side; the identity the
          // reviewer saw travels in expected_candidate_dirs instead.
          correct_dir: null,
        }
        const payload: FeedbackPayload =
          current.readiness.outcome === 'none_of_top_k'
            ? { ...base, outcome: 'none_of_top_k', correct_rank: 0 }
            : {
                ...base,
                outcome: 'matched_rank',
                correct_rank: current.readiness.selections[0].rank,
                selected_ranks: current.readiness.selections.map(
                  (selection) => selection.rank,
                ),
                expected_candidate_dirs: expectedCandidateDirs(
                  current.readiness.selections,
                ),
              }
        return { payload, snapshot: { ...current.draft } }
      }),
    [runSave],
  )

  const skipFeedback = useCallback(
    (): Promise<SaveOutcome> =>
      runSave('skip', (current) => {
        const notes = current.draft.notes.trim()
        if (!notes) {
          throw new AssessmentNotSavableError(
            'Add a note before skipping so the PI can follow up.',
          )
        }
        return {
          payload: {
            query_id: current.queryId as number,
            model_slug: current.model,
            variant: current.variant,
            outcome: 'skipped',
            correct_rank: null,
            correct_dir: null,
            notes,
          },
          snapshot: { ...current.draft, notes },
        }
      }),
    [runSave],
  )

  const lastSavedRef = useRef(lastSaved)
  lastSavedRef.current = lastSaved

  /**
   * The acknowledgement, if it belongs to whoever is signed in now.
   *
   * An acknowledgement is one reviewer's receipt for their own work. It is
   * recorded only by the visit that made it, and it stops being shown the
   * moment the account changes: telling the next reviewer that an assessment
   * was saved, and offering them a draft to put back, describes work that is
   * not theirs.
   */
  const lastSavedIsOurs = lastSaved !== null && lastSaved.accountId === accountId
  const newerThanSaved =
    lastSaved !== null && draftRevision(drafts.get(lastSaved.key)) > lastSaved.revision
  const canRestoreDraft = lastSavedIsOurs && !newerThanSaved

  const lastSubmission: LastSubmission | null =
    lastSaved === null || !lastSavedIsOurs
      ? null
      : {
          key: lastSaved.key,
          kind: lastSaved.kind,
          queryId: lastSaved.queryId,
          entry: lastSaved.entry,
          canRestoreDraft,
          supersededByNewerDraft: newerThanSaved,
        }

  const restoreSubmittedDraft = useCallback(() => {
    const saved = lastSavedRef.current
    if (saved === null) return
    // Another reviewer's receipt is not this reviewer's draft to restore.
    if (saved.accountId !== accountIdRef.current) return
    // Newer unsent work outranks a restore: the saved row is in the log either
    // way, and this button cannot take it back.
    if (draftRevision(draftsRef.current.get(saved.key)) > saved.revision) {
      setLastSaved(null)
      return
    }
    const revision = nextRevision(saved.key)
    setDrafts((prev) => {
      if (draftRevision(prev.get(saved.key)) > saved.revision) return prev
      const next = new Map(prev)
      next.set(saved.key, { ...saved.draft, revision })
      return next
    })
    setLastSaved(null)
  }, [nextRevision])

  const dismissLastSubmission = useCallback(() => {
    setLastSaved(null)
  }, [])

  const value: FeedbackContextValue = {
    drafts,
    draftKey,
    accountId,
    makeDraftKey,
    draft,
    getDraft,
    hasQuarantinedLegacyDraft,
    evidence,
    selectionReview,
    readiness,
    issues,
    updateDraft,
    setNotes,
    toggleCandidate,
    setNone,
    collapseToFirstChoice,
    seedDraftIfEmpty,
    submitFeedback,
    skipFeedback,
    pendingSave: draftKey === null ? null : (pendingOps.get(draftKey) ?? null),
    visit,
    getVisit,
    hasNewerDraft,
    getDraftRevision,
    getSaveEpoch,
    lastSubmission,
    restoreSubmittedDraft,
    dismissLastSubmission,
  }

  return <FeedbackContext.Provider value={value}>{children}</FeedbackContext.Provider>
}

export function useFeedback(): FeedbackContextValue {
  const ctx = useContext(FeedbackContext)
  if (!ctx) {
    throw new Error('useFeedback must be used within a FeedbackProvider')
  }
  return ctx
}
