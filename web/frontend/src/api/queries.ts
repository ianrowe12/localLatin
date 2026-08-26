import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'
import { FEEDBACK_UPDATED_EVENT } from './feedback'
import { REVIEWER_DIRS_UPDATED_EVENT, type ReviewerDir } from './reviewerDirs'
import { DEFAULT_VARIANT, type PredictionVariant } from './variants'

// ---------------------------------------------------------------------------
// Types matching backend models.py
// ---------------------------------------------------------------------------

export interface QueryListItem {
  file_id: number
  filename: string
  text_preview: string
  review_status: 'unreviewed' | 'reviewed' | 'skipped'
  review_count: number
}

export interface QueryListResponse {
  items: QueryListItem[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export interface NextQueryResponse {
  file_id: number | null
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

/**
 * Where a candidate came from. `model` is a labelled directory the retrieval
 * run ranked; `reviewer` is a directory a reviewer created, scored live from
 * the query-query matrix. Reviewer candidates are appended after the model's
 * ten, so a model candidate's rank still means what it always did.
 */
export type CandidateSource = 'model' | 'reviewer'

export interface Prediction {
  rank: number
  dir_name: string
  score: number
  dir_files: string[]
  preview_text: string
  candidate_files: CandidateFile[] | null
  source?: CandidateSource
  // Reviewer candidates only.
  label?: string | null
  created_by?: string | null
  seed_query_id?: number | null
}

export interface PredictionResponse {
  file_id: number
  filename: string
  model: string
  variant: PredictionVariant
  predictions: Prediction[]
  // Reviewer directories seeded by this query. Drives the badge on the
  // document itself rather than on any candidate card.
  seeded_dirs?: ReviewerDir[]
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
  pageSize = 50,
): HookState<QueryListResponse> {
  const [state, setState] = useState<HookState<QueryListResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, QueryListResponse>())
  const listKeyRef = useRef<string | null>(null)
  const [refreshVersion, setRefreshVersion] = useState(0)

  useEffect(() => {
    const refresh = () => {
      cache.current.clear()
      setRefreshVersion((version) => version + 1)
    }
    window.addEventListener(FEEDBACK_UPDATED_EVENT, refresh)
    return () => window.removeEventListener(FEEDBACK_UPDATED_EVENT, refresh)
  }, [])

  useEffect(() => {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (search) params.set('search', search)
    params.set('page', String(page))
    params.set('page_size', String(pageSize))
    const key = params.toString()
    const listKey = `${status}|${search}|${pageSize}`
    const sameList = listKeyRef.current === listKey
    listKeyRef.current = listKey

    const applyPage = (data: QueryListResponse) => {
      setState((prev) => {
        if (page > 1 && sameList && prev.data) {
          const seen = new Set(prev.data.items.map((item) => item.file_id))
          const mergedItems = [
            ...prev.data.items,
            ...data.items.filter((item) => !seen.has(item.file_id)),
          ]
          return {
            data: { ...data, items: mergedItems },
            loading: false,
            error: null,
          }
        }
        return { data, loading: false, error: null }
      })
    }

    const cached = cache.current.get(key)
    if (cached) {
      applyPage(cached)
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<QueryListResponse>(`/api/queries?${key}`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(key, data)
        applyPage(data)
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [status, search, page, pageSize, refreshVersion])

  return state
}

export async function fetchNextQuery(after?: number): Promise<NextQueryResponse> {
  const params = new URLSearchParams()
  if (after !== undefined) params.set('after', String(after))
  const query = params.toString()
  return apiFetch<NextQueryResponse>(`/api/queries/next${query ? `?${query}` : ''}`)
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

// ---------------------------------------------------------------------------
// useCandidateDirFiles
// ---------------------------------------------------------------------------

export function useCandidateDirFiles(
  candidateDir: string | null,
): HookState<CandidateFile[]> {
  const [state, setState] = useState<HookState<CandidateFile[]>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, CandidateFile[]>())

  useEffect(() => {
    if (candidateDir === null) {
      setState({ data: null, loading: false, error: null })
      return
    }

    const cached = cache.current.get(candidateDir)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    const encoded = encodeURIComponent(candidateDir)
    apiFetch<CandidateFile[]>(`/api/candidate_dir/${encoded}/files`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(candidateDir, data)
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [candidateDir])

  return state
}

export function usePredictions(
  queryId: number | null,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): HookState<PredictionResponse> {
  const [state, setState] = useState<HookState<PredictionResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, PredictionResponse>())
  const [refreshVersion, setRefreshVersion] = useState(0)

  // A new reviewer directory becomes a candidate for every *other* query, so
  // the whole cache is stale, not just the seed's entry.
  useEffect(() => {
    const refresh = () => {
      cache.current.clear()
      setRefreshVersion((version) => version + 1)
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, refresh)
    return () => window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, refresh)
  }, [])

  useEffect(() => {
    if (queryId === null || !model) {
      setState({ data: null, loading: false, error: null })
      return
    }

    const key = `${queryId}:${model}:${variant}`
    const cached = cache.current.get(key)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    // Drop the previous key's data rather than keeping it under `loading`.
    // Holding it would let one paint pair the OLD variant's rank-1 candidate
    // with the NEW variant: a wasted token-map fetch for a candidate that is
    // about to change, the previous variant's text under the new variant's
    // label, and a submit in that window posting the old ranking under the new
    // variant name.
    setState({ data: null, loading: true, error: null })

    apiFetch<PredictionResponse>(
      `/api/query/${queryId}/predictions?model=${encodeURIComponent(model)}&variant=${variant}`,
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
  }, [queryId, model, variant, refreshVersion])

  return state
}
