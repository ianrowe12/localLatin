import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { submitFeedback as postFeedback } from '../api/feedback'

export interface FeedbackDraft {
  correctRank: number | null
  notes: string
  reviewer: string
}

export interface FeedbackContextValue {
  drafts: Map<string, FeedbackDraft>
  getDraft: (queryId: number, model: string) => FeedbackDraft
  updateDraft: (queryId: number, model: string, patch: Partial<FeedbackDraft>) => void
  submitFeedback: (queryId: number, model: string, reviewer?: string) => Promise<void>
  undoLastSubmit: () => void
  lastSubmittedKey: string | null
}

const FeedbackContext = createContext<FeedbackContextValue | null>(null)

const STORAGE_KEY = 'locallatin-feedback-drafts'

function makeDraftKey(queryId: number, model: string): string {
  return `${queryId}-${model}`
}

function emptyDraft(): FeedbackDraft {
  return { correctRank: null, notes: '', reviewer: '' }
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

  const submitFeedback = useCallback(
    async (queryId: number, model: string, reviewer?: string): Promise<void> => {
      const key = makeDraftKey(queryId, model)
      const draft = drafts.get(key) ?? emptyDraft()

      await postFeedback({
        query_id: queryId,
        model_slug: model,
        correct_rank: draft.correctRank,
        correct_dir: null,
        notes: draft.notes,
        reviewer: reviewer ?? draft.reviewer,
      })

      lastSubmittedDraft.current = { key, draft: { ...draft } }
      setLastSubmittedKey(key)

      setDrafts((prev) => {
        const next = new Map(prev)
        next.delete(key)
        return next
      })
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
    submitFeedback,
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
