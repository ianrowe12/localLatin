import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  FEEDBACK_UPDATED_EVENT,
  submitFeedback as postFeedback,
  type FeedbackPayload,
} from '../api/feedback'
import type { Prediction } from '../api/queries'
import { DEFAULT_VARIANT, type PredictionVariant } from '../api/variants'
import { seedDraftMapIfEmpty } from './feedbackDraft'

export interface FeedbackDraft {
  correctRank: number | null // 0 = none-of-top-k, null = unset, >0 = single legacy
  selectedRanks?: number[] // present ⇒ multi answer (ranks 1..10)
  notes: string
}

export interface FeedbackContextValue {
  drafts: Map<string, FeedbackDraft>
  makeDraftKey: (queryId: number, model: string, variant?: PredictionVariant) => string
  getDraft: (
    queryId: number,
    model: string,
    variant?: PredictionVariant,
  ) => FeedbackDraft
  updateDraft: (
    queryId: number,
    model: string,
    patch: Partial<FeedbackDraft>,
    variant?: PredictionVariant,
  ) => void
  seedDraftIfEmpty: (
    queryId: number,
    model: string,
    seed: FeedbackDraft,
    variant?: PredictionVariant,
  ) => void
  submitFeedback: (
    queryId: number,
    model: string,
    predictions: Prediction[],
    variant?: PredictionVariant,
  ) => Promise<void>
  skipFeedback: (
    queryId: number,
    model: string,
    notes: string,
    variant?: PredictionVariant,
  ) => Promise<void>
  undoLastSubmit: () => void
  lastSubmittedKey: string | null
}

const FeedbackContext = createContext<FeedbackContextValue | null>(null)

const STORAGE_KEY = 'locallatin-feedback-drafts'

// The variant is part of the key: the same query/model ranks differently per
// variant, so drafts must not bleed across them (#48). The selector it was
// written for is gone (#94 pinned the app to one pipeline), but the key stays:
// it is what stops a stored draft from resurfacing against a different ranking.
function makeDraftKey(
  queryId: number,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): string {
  return `${queryId}-${model}-${variant}`
}

function emptyDraft(): FeedbackDraft {
  return { correctRank: null, notes: '' }
}

function serializeDrafts(map: Map<string, FeedbackDraft>): string {
  return JSON.stringify(Array.from(map.entries()))
}

function deserializeDrafts(raw: string | null): Map<string, FeedbackDraft> {
  if (!raw) return new Map()
  try {
    const entries: [string, FeedbackDraft][] = JSON.parse(raw)
    return new Map(entries)
  } catch {
    return new Map()
  }
}

export function FeedbackProvider({ children }: { children: ReactNode }) {
  const [drafts, setDrafts] = useState<Map<string, FeedbackDraft>>(() =>
    deserializeDrafts(localStorage.getItem(STORAGE_KEY)),
  )
  const [lastSubmittedKey, setLastSubmittedKey] = useState<string | null>(null)
  const lastSubmittedDraft = useRef<{ key: string; draft: FeedbackDraft } | null>(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, serializeDrafts(drafts))
  }, [drafts])

  const getDraft = useCallback(
    (
      queryId: number,
      model: string,
      variant: PredictionVariant = DEFAULT_VARIANT,
    ): FeedbackDraft => {
      const key = makeDraftKey(queryId, model, variant)
      return drafts.get(key) ?? emptyDraft()
    },
    [drafts],
  )

  const updateDraft = useCallback(
    (
      queryId: number,
      model: string,
      patch: Partial<FeedbackDraft>,
      variant: PredictionVariant = DEFAULT_VARIANT,
    ) => {
      setDrafts((prev) => {
        const key = makeDraftKey(queryId, model, variant)
        const existing = prev.get(key) ?? emptyDraft()
        const next = new Map(prev)
        next.set(key, { ...existing, ...patch })
        return next
      })
    },
    [],
  )

  const seedDraftIfEmpty = useCallback(
    (
      queryId: number,
      model: string,
      seed: FeedbackDraft,
      variant: PredictionVariant = DEFAULT_VARIANT,
    ) => {
      setDrafts((prev) => {
        const key = makeDraftKey(queryId, model, variant)
        return seedDraftMapIfEmpty(prev, key, seed)
      })
    },
    [],
  )

  const submitFeedback = useCallback(
    async (
      queryId: number,
      model: string,
      predictions: Prediction[],
      variant: PredictionVariant = DEFAULT_VARIANT,
    ): Promise<void> => {
      const key = makeDraftKey(queryId, model, variant)
      const draft = drafts.get(key) ?? emptyDraft()
      const dirForRank = (rank: number): string | null =>
        predictions.find((prediction) => prediction.rank === rank)?.dir_name ?? null
      const multiRanks =
        draft.correctRank !== 0 && draft.selectedRanks && draft.selectedRanks.length > 0
          ? draft.selectedRanks
          : null

      let payload: FeedbackPayload
      if (draft.correctRank === 0) {
        payload = {
          query_id: queryId,
          model_slug: model,
          variant,
          outcome: 'none_of_top_k',
          correct_rank: 0,
          correct_dir: null,
          notes: draft.notes,
        }
      } else if (multiRanks) {
        // Multi-select (also covers length 1). Backend forces matched_rank,
        // sets correct_rank = selected_ranks[0], and derives correct_dir.
        payload = {
          query_id: queryId,
          model_slug: model,
          variant,
          correct_rank: multiRanks[0],
          correct_dir: dirForRank(multiRanks[0]),
          selected_ranks: multiRanks,
          notes: draft.notes,
        }
      } else if (draft.correctRank && draft.correctRank > 0) {
        payload = {
          query_id: queryId,
          model_slug: model,
          variant,
          outcome: 'matched_rank',
          correct_rank: draft.correctRank,
          correct_dir: dirForRank(draft.correctRank),
          notes: draft.notes,
        }
      } else {
        payload = {
          query_id: queryId,
          model_slug: model,
          variant,
          correct_rank: null,
          correct_dir: null,
          notes: draft.notes,
        }
      }

      await postFeedback(payload)

      lastSubmittedDraft.current = { key, draft: { ...draft } }
      setLastSubmittedKey(key)

      setDrafts((prev) => {
        const next = new Map(prev)
        next.delete(key)
        return next
      })
      window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
    },
    [drafts],
  )

  const skipFeedback = useCallback(
    async (
      queryId: number,
      model: string,
      notes: string,
      variant: PredictionVariant = DEFAULT_VARIANT,
    ): Promise<void> => {
      const key = makeDraftKey(queryId, model, variant)
      const draft = drafts.get(key) ?? emptyDraft()

      await postFeedback({
        query_id: queryId,
        model_slug: model,
        variant,
        outcome: 'skipped',
        correct_rank: null,
        correct_dir: null,
        notes,
      })

      lastSubmittedDraft.current = { key, draft: { ...draft, notes } }
      setLastSubmittedKey(key)

      setDrafts((prev) => {
        const next = new Map(prev)
        next.delete(key)
        return next
      })
      window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
    },
    [drafts],
  )

  const undoLastSubmit = useCallback(() => {
    if (!lastSubmittedDraft.current) return
    const { key, draft } = lastSubmittedDraft.current
    setDrafts((prev) => {
      const next = new Map(prev)
      next.set(key, draft)
      return next
    })
    lastSubmittedDraft.current = null
    setLastSubmittedKey(null)
  }, [])

  const value: FeedbackContextValue = {
    drafts,
    makeDraftKey,
    getDraft,
    updateDraft,
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
