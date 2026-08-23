import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'
import { FEEDBACK_UPDATED_EVENT } from './feedback'
import type { PredictionVariant } from './variants'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelInfo {
  slug: string
  display_name: string
  layer: number | null
  pooling: string | null
  prediction_count: number
  // Populated by GET /api/models; consumed by the variant selector (#48).
  available_variants: PredictionVariant[]
  default_variant: PredictionVariant
}

export interface RecentReview {
  file_id: number
  filename: string
  timestamp: string
  model_slug: string
  outcome: string
  reviewer: string
  correct_rank: number | null
}

export interface NeedsAttentionItem {
  file_id: number
  filename: string
  timestamp: string
  model_slug: string
  outcome: string
  notes: string
  reviewer: string
}

export interface StatsResponse {
  total_queries: number
  reviewed_count: number
  skipped_count: number
  unreviewed_count: number
  unresolved_count: number
  feedback_count: number
  reviews_by_model: Record<string, number>
  reviews_by_reviewer: Record<string, number>
  rank_distribution: Record<string, number>
  outcome_distribution: Record<string, number>
  recent_reviews: RecentReview[]
  needs_attention: NeedsAttentionItem[]
  next_unreviewed_ids: number[]
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

interface HookState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

// Module-level rather than per-hook: the model list is immutable for the life
// of a deployment and has more than one consumer (ModelSelector and the
// variant selector), which would otherwise each fetch it.
let modelsCache: ModelInfo[] | null = null

export function useModels(): HookState<ModelInfo[]> {
  const [state, setState] = useState<HookState<ModelInfo[]>>({
    data: modelsCache,
    loading: false,
    error: null,
  })

  useEffect(() => {
    if (modelsCache) {
      setState({ data: modelsCache, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<ModelInfo[]>('/api/models')
      .then((data) => {
        if (cancelled) return
        modelsCache = data
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [])

  return state
}

export function useStats(enabled = true): HookState<StatsResponse> {
  const [state, setState] = useState<HookState<StatsResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef<StatsResponse | null>(null)
  const [refreshVersion, setRefreshVersion] = useState(0)

  useEffect(() => {
    const refresh = () => {
      cache.current = null
      setRefreshVersion((version) => version + 1)
    }
    window.addEventListener(FEEDBACK_UPDATED_EVENT, refresh)
    return () => window.removeEventListener(FEEDBACK_UPDATED_EVENT, refresh)
  }, [])

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null })
      return
    }
    if (cache.current) {
      setState({ data: cache.current, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<StatsResponse>('/api/stats')
      .then((data) => {
        if (cancelled) return
        cache.current = data
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [enabled, refreshVersion])

  return state
}
