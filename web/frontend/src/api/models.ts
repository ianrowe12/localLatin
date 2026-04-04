import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelInfo {
  slug: string
  display_name: string
  layer: number | null
  pooling: string | null
  prediction_count: number
}

export interface RecentReview {
  file_id: number
  filename: string
  timestamp: string
  model_slug: string
}

export interface StatsResponse {
  total_queries: number
  reviewed_count: number
  unreviewed_count: number
  feedback_count: number
  reviews_by_model: Record<string, number>
  reviews_by_reviewer: Record<string, number>
  rank_distribution: Record<string, number>
  recent_reviews: RecentReview[]
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

export function useModels(): HookState<ModelInfo[]> {
  const [state, setState] = useState<HookState<ModelInfo[]>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef<ModelInfo[] | null>(null)

  useEffect(() => {
    if (cache.current) {
      setState({ data: cache.current, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<ModelInfo[]>('/api/models')
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
  }, [])

  return state
}

export function useStats(): HookState<StatsResponse> {
  const [state, setState] = useState<HookState<StatsResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef<StatsResponse | null>(null)

  useEffect(() => {
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
  }, [])

  return state
}
