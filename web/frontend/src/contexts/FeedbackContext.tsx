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
  ) => void

  /**
   * Save the current deliberate evaluation. Takes no candidates: the payload is
   * built from the ranking on screen, so no caller can supply its own.
   * Rejects, without any request, when the evidence does not support the draft.
   */
  submitFeedback: () => Promise<void>
  /** Save a deliberate, note-required deferral. Never inferred, never None. */
  skipFeedback: () => Promise<void>
  undoLastSubmit: () => void
  lastSubmittedKey: string | null
}

const FeedbackContext = createContext<FeedbackContextValue | null>(null)

const STORAGE_KEY = 'locallatin-feedback-drafts'

/** Refused locally, before any request. #158 owns in-flight save integrity. */
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
  const { activeQueryId, activeModel, activeVariant } = useApp()
  const { phase, predictions } = usePredictionState()
  const { user } = useReviewer()
  const accountId = user?.id ?? null

  const [drafts, setDrafts] = useState<Map<string, FeedbackDraft>>(() =>
    deserializeDrafts(localStorage.getItem(STORAGE_KEY)),
  )
  const [lastSubmittedKey, setLastSubmittedKey] = useState<string | null>(null)
  const lastSubmittedDraft = useRef<{ key: string; draft: FeedbackDraft } | null>(null)
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

  const draftKey =
    activeQueryId === null || !activeModel || accountId === null
      ? null
      : makeDraftKey(activeQueryId, activeModel, activeVariant, accountId)

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
      next.set(draftKey, draftWithSelections(existing, selectionReview.selections))
      return next
    })
    setDroppedIssues({ key: draftKey, issues: dropped })
  }, [draftKey, selectionReview])

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
      setDrafts((prev) => {
        const next = new Map(prev)
        next.set(draftKey, update(prev.get(draftKey) ?? emptyDraft()))
        return next
      })
    },
    [draftKey],
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
    ) => {
      if (accountId === null) return
      const key = makeDraftKey(queryId, model, variant, accountId)
      // A prefill that arrives after the reviewer has moved on belongs to a key
      // nobody is looking at; it must not resurface as a current answer.
      if (key !== draftKey) return
      setDrafts((prev) => seedDraftMapIfEmpty(prev, key, seed))
    },
    [accountId, draftKey],
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

  const recordSubmission = useCallback((key: string, saved: FeedbackDraft) => {
    lastSubmittedDraft.current = { key, draft: saved }
    setLastSubmittedKey(key)
    setDrafts((prev) => {
      const next = new Map(prev)
      next.delete(key)
      return next
    })
    window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
  }, [])

  const submitFeedback = useCallback(async (): Promise<void> => {
    const current = currentRef.current
    if (current.key === null || current.queryId === null) {
      throw new AssessmentNotSavableError('There is no assessment on screen to save.')
    }
    if (!current.readiness.canSubmit || current.readiness.outcome === null) {
      throw new AssessmentNotSavableError(
        'This assessment cannot be saved against the ranking on screen.',
      )
    }

    const base = {
      query_id: current.queryId,
      model_slug: current.model,
      variant: current.variant,
      notes: current.draft.notes,
      // Never an assignment authority server-side; the identity the reviewer
      // saw travels in expected_candidate_dirs instead.
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
            expected_candidate_dirs: expectedCandidateDirs(current.readiness.selections),
          }

    await postFeedback(payload)
    recordSubmission(current.key, { ...current.draft })
  }, [recordSubmission])

  const skipFeedback = useCallback(async (): Promise<void> => {
    const current = currentRef.current
    if (current.key === null || current.queryId === null) {
      throw new AssessmentNotSavableError('There is no assessment on screen to defer.')
    }
    const notes = current.draft.notes.trim()
    if (!notes) {
      throw new AssessmentNotSavableError(
        'Add a note before skipping so the PI can follow up.',
      )
    }

    await postFeedback({
      query_id: current.queryId,
      model_slug: current.model,
      variant: current.variant,
      outcome: 'skipped',
      correct_rank: null,
      correct_dir: null,
      notes,
    })
    recordSubmission(current.key, { ...current.draft, notes })
  }, [recordSubmission])

  const undoLastSubmit = useCallback(() => {
    if (!lastSubmittedDraft.current) return
    const { key, draft: restored } = lastSubmittedDraft.current
    setDrafts((prev) => {
      const next = new Map(prev)
      next.set(key, restored)
      return next
    })
    lastSubmittedDraft.current = null
    setLastSubmittedKey(null)
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
    undoLastSubmit,
    lastSubmittedKey,
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
