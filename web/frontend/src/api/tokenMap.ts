import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TokenEntry {
  idx: number
  text: string
  is_content: boolean
}

export interface TopMatch {
  candidate_idx: number
  score: number
}

export interface TokenMapResponse {
  example_id: number
  model: string
  layer: number
  D: number
  bucket: string
  query_path: string
  candidate_path: string
  query_tokens: TokenEntry[]
  candidate_tokens: TokenEntry[]
  similarity_matrix: number[][]
  ig_weighted_matrix: number[][] | null
  top_matches: Record<string, TopMatch[]>
  query_ig_baseline: number[]
  query_ig_abtt: number[]
  candidate_ig_baseline: number[]
  candidate_ig_abtt: number[]
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface HookState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

export function useTokenMap(
  queryId: number | null,
  candidateId: string | null,
  model?: string,
): HookState<TokenMapResponse> {
  const [state, setState] = useState<HookState<TokenMapResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, TokenMapResponse>())

  useEffect(() => {
    if (queryId === null || candidateId === null) {
      setState({ data: null, loading: false, error: null })
      return
    }

    const key = `${queryId}:${candidateId}:${model ?? ''}`
    const cached = cache.current.get(key)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    const params = new URLSearchParams({ candidate_dir: candidateId })
    if (model) params.set('model', model)
    apiFetch<TokenMapResponse>(
      `/api/query/${queryId}/token_map?${params}`,
    )
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
  }, [queryId, candidateId, model])

  return state
}
