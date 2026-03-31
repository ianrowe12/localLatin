import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'

// ---------------------------------------------------------------------------
// Types matching backend models.py
// ---------------------------------------------------------------------------

export interface QueryListItem {
  file_id: number
  filename: string
  text_preview: string
  review_status: string // "unreviewed" | "reviewed"
  review_count: number
}

export interface QueryListResponse {
  items: QueryListItem[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export interface TokenInfo {
  text: string
  index: number
  category: string
}

export interface QueryDetail {
  file_id: number
  filename: string
  text: string
  tokens: TokenInfo[]
  char_count: number
  token_count: number
}

export interface CandidateFile {
  filename: string
  text: string
}

export interface Prediction {
  rank: number
  dir_name: string
  score: number
  dir_files: string[]
  preview_text: string
  candidate_files: CandidateFile[] | null
}

export interface PredictionResponse {
  file_id: number
  filename: string
  model: string
  predictions: Prediction[]
}

// ---------------------------------------------------------------------------
// Hook state shape
// ---------------------------------------------------------------------------

interface HookState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

// ---------------------------------------------------------------------------
// useQueryList
// ---------------------------------------------------------------------------

export function useQueryList(
  status: string,
  search: string,
  page: number,
): HookState<QueryListResponse> {
  const [state, setState] = useState<HookState<QueryListResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, QueryListResponse>())

  useEffect(() => {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (search) params.set('search', search)
    params.set('page', String(page))
    const key = params.toString()

    const cached = cache.current.get(key)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<QueryListResponse>(`/api/queries?${key}`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(key, data)
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [status, search, page])

  return state
}

// ---------------------------------------------------------------------------
// useQueryDetail
// ---------------------------------------------------------------------------

export function useQueryDetail(id: number | null): HookState<QueryDetail> {
  const [state, setState] = useState<HookState<QueryDetail>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<number, QueryDetail>())

  useEffect(() => {
    if (id === null) {
      setState({ data: null, loading: false, error: null })
      return
    }

    const cached = cache.current.get(id)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<QueryDetail>(`/api/query/${id}`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(id, data)
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [id])

  return state
}

// ---------------------------------------------------------------------------
// usePredictions
// ---------------------------------------------------------------------------

export function usePredictions(
  queryId: number | null,
  model: string,
): HookState<PredictionResponse> {
  const [state, setState] = useState<HookState<PredictionResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, PredictionResponse>())

  useEffect(() => {
    if (queryId === null || !model) {
      setState({ data: null, loading: false, error: null })
      return
    }

    const key = `${queryId}:${model}`
    const cached = cache.current.get(key)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<PredictionResponse>(`/api/query/${queryId}/predictions?model=${encodeURIComponent(model)}`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(key, data)
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [queryId, model])

  return state
}
