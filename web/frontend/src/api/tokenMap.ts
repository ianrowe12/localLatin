import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'
import type { AttributionVariant } from './variants'

// The attribution variant vocabulary lives in ./variants next to the
// prediction variants and the single raw<->baseline mapping
// (`toAttributionVariant`). Re-exported here so existing importers of
// `AttributionVariant` from this module keep working.
export type { AttributionVariant } from './variants'

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

export interface AutoHighlight {
  query_idx: number
  ig_score: number
  matches: TopMatch[]
}

export type AttributionMethod =
  | 'ig'
  | 'bertscore'
  | 'ot'
  | 'attention_weighted'
  | 'dla'
  | 'attention_standalone'
  | 'retrieval_mark'

export interface AttributionTopHighlights {
  query: number[]
  candidate: number[]
}

export interface TokenMapResponse {
  example_id: number
  model: string
  layer: number
  /** PCs removed by the `abtt` variant (mean-pooled ABTT fit). */
  D: number
  /** PCs removed by the `sif_abtt` variant (SIF-pooled fit, swept separately). */
  D_sif?: number | null
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
  auto_highlights: AutoHighlight[] | null
  available_methods?: AttributionMethod[]
  // Every variant present in the artifact, regardless of the ?variant= filter
  // applied to this particular response.
  available_variants?: AttributionVariant[]
  query_sif_weights?: number[] | null
  candidate_sif_weights?: number[] | null
  pair_matrices?: Partial<
    Record<AttributionMethod, Partial<Record<AttributionVariant, number[][]>>>
  >
  top_highlights?: Partial<
    Record<
      AttributionMethod,
      Partial<Record<AttributionVariant, AttributionTopHighlights>>
    >
  >
}

export interface TokenMapExampleCard {
  example_id: number
  model_slug: string
  bucket: string
  query_file_id: number
  query_folder_id: string
  query_filename: string
  candidate_folder_id: string
  candidate_label: string
  methods_available: AttributionMethod[]
  variants_available?: AttributionVariant[]
  gold_similar: number
  baseline_pred: number
  abtt_pred: number
}

export interface TokenMapExamplesGroupedResponse {
  by_model: Record<string, TokenMapExampleCard[]>
  bucket_order: string[]
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface HookState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

// Token-map responses run to several MB each even filtered, and the cache key
// now includes method and variant -- so a day-long session toggling variants
// across many pairs would grow without bound. Map preserves insertion order,
// so re-inserting on read gives least-recently-used eviction for free.
const TOKEN_MAP_CACHE_LIMIT = 8

function cacheGet(
  cache: Map<string, TokenMapResponse>,
  key: string,
): TokenMapResponse | undefined {
  const hit = cache.get(key)
  if (hit === undefined) return undefined
  cache.delete(key)
  cache.set(key, hit)
  return hit
}

function cachePut(
  cache: Map<string, TokenMapResponse>,
  key: string,
  value: TokenMapResponse,
): void {
  cache.delete(key)
  cache.set(key, value)
  while (cache.size > TOKEN_MAP_CACHE_LIMIT) {
    const oldest = cache.keys().next().value
    if (oldest === undefined) break
    cache.delete(oldest)
  }
}

/**
 * Fetch the token map for one (query, candidate) pair.
 *
 * `method` and `variant` are forwarded as `?method=&variant=` so the backend
 * serialises only the single matrix the evidence view renders. Unfiltered, the
 * response carries every persisted method x variant grid — tens of megabytes
 * on long pairs (issue #72). `available_methods` / `available_variants` still
 * describe the whole artifact, so narrowing the fetch never hides options.
 *
 * `variant` is an *attribution* variant: callers holding the reviewer's
 * prediction variant must map it through `toAttributionVariant`.
 */
export function useTokenMap(
  queryId: number | null,
  candidateId: string | null,
  model?: string,
  method?: AttributionMethod | null,
  variant?: AttributionVariant,
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

    const key = `${queryId}:${candidateId}:${model ?? ''}:${method ?? ''}:${variant ?? ''}`
    const cached = cacheGet(cache.current, key)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    // Drop the previous key's matrices rather than keeping them under
    // `loading`. Holding them renders the OLD (candidate, variant) pairing's
    // highlights under the NEW variant's label for as long as the multi-MB
    // fetch takes -- and forever if it 404s, since the error branch below only
    // sets `error`. That is the "highlights don't update after switching
    // post-processing" report in issue #73.
    setState({ data: null, loading: true, error: null })

    const params = new URLSearchParams({ candidate_dir: candidateId })
    if (model) params.set('model', model)
    if (method) params.set('method', method)
    if (variant) params.set('variant', variant)
    apiFetch<TokenMapResponse>(
      `/api/query/${queryId}/token_map?${params}`,
    )
      .then((data) => {
        if (cancelled) return
        cachePut(cache.current, key, data)
        setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        // No data for this key: a pair with no artifact for the requested
        // variant must render as "nothing to show", never as the last pair
        // that happened to load.
        setState({ data: null, loading: false, error: err.message })
      })

    // React runs this cleanup before the next effect, so `cancelled` also
    // settles the two-fetches-in-flight race: a slow response for a superseded
    // key can never land after a newer one.
    return () => {
      cancelled = true
    }
  }, [queryId, candidateId, model, method, variant])

  return state
}

// ---------------------------------------------------------------------------
// useTokenMapExamplesGrouped
// ---------------------------------------------------------------------------

export function useTokenMapExamplesGrouped(): HookState<TokenMapExamplesGroupedResponse> {
  const [state, setState] = useState<HookState<TokenMapExamplesGroupedResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef<TokenMapExamplesGroupedResponse | null>(null)

  useEffect(() => {
    if (cache.current) {
      setState({ data: cache.current, loading: false, error: null })
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<TokenMapExamplesGroupedResponse>('/api/token_map_examples_grouped')
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
