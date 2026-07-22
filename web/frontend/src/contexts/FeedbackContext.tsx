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
import { seedDraftMapIfEmpty } from './feedbackDraft'

export interface FeedbackDraft {
  correctRank: number | null // 0 = none-of-top-k, null = unset, >0 = single legacy
  selectedRanks?: number[] // present ⇒ multi answer (ranks 1..10)
  notes: string
}

export interface FeedbackContextValue {
  drafts: Map<string, FeedbackDraft>
  getDraft: (queryId: number, model: string) => FeedbackDraft
  updateDraft: (queryId: number, model: string, patch: Partial<FeedbackDraft>) => void
  seedDraftIfEmpty: (queryId: number, model: string, seed: FeedbackDraft) => void
  submitFeedback: (queryId: number, model: string, predictions: Prediction[]) => Promise<void>
  skipFeedback: (queryId: number, model: string, notes: string) => Promise<void>
  undoLastSubmit: () => void
  lastSubmittedKey: string | null
}

const FeedbackContext = createContext<FeedbackContextValue | null>(null)

const STORAGE_KEY = 'locallatin-feedback-drafts'

function makeDraftKey(queryId: number, model: string): string {
  return `${queryId}-${model}`
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
    (queryId: number, model: string): FeedbackDraft => {
      const key = makeDraftKey(queryId, model)
      return drafts.get(key) ?? emptyDraft()
    },
    [drafts],
  )

  const updateDraft = useCallback(
    (queryId: number, model: string, patch: Partial<FeedbackDraft>) => {
      setDrafts((prev) => {
        const key = makeDraftKey(queryId, model)
        const existing = prev.get(key) ?? emptyDraft()
        const next = new Map(prev)
        next.set(key, { ...existing, ...patch })
        return next
      })
    },
    [],
  )

  const seedDraftIfEmpty = useCallback(
    (queryId: number, model: string, seed: FeedbackDraft) => {
      setDrafts((prev) => {
        const key = makeDraftKey(queryId, model)
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
    ): Promise<void> => {
      const key = makeDraftKey(queryId, model)
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
          correct_rank: multiRanks[0],
          correct_dir: dirForRank(multiRanks[0]),
          selected_ranks: multiRanks,
          notes: draft.notes,
        }
      } else if (draft.correctRank && draft.correctRank > 0) {
        payload = {
          query_id: queryId,
          model_slug: model,
          outcome: 'matched_rank',
          correct_rank: draft.correctRank,
          correct_dir: dirForRank(draft.correctRank),
          notes: draft.notes,
        }
      } else {
        payload = {
          query_id: queryId,
          model_slug: model,
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
    async (queryId: number, model: string, notes: string): Promise<void> => {
      const key = makeDraftKey(queryId, model)
      const draft = drafts.get(key) ?? emptyDraft()

      await postFeedback({
        query_id: queryId,
        model_slug: model,
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
