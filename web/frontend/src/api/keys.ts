import { useEffect, useRef, useState } from 'react'
import { apiFetch } from './client'

// Types matching `LabelledKeyItem` / `LabelledKeyListResponse` in web/models.py.

export interface LabelledKeyItem {
  key: string
  unit_count: number
}

export interface LabelledKeyListResponse {
  items: LabelledKeyItem[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export const KEY_SUGGESTION_LIMIT = 8
export const KEY_PAGE_SIZE = 50

function labelledKeysPath(search: string, page: number, pageSize: number): string {
  const params = new URLSearchParams()
  if (search) params.set('search', search)
  params.set('page', String(page))
  params.set('page_size', String(pageSize))
  return `/api/labelled_keys?${params.toString()}`
}

interface KeyListState {
  data: LabelledKeyListResponse | null
  loading: boolean
  error: string | null
}

/**
 * One page of the labelled keys matching `search`; later pages of the same
 * search are appended, like `useQueryList`.
 */
export function useLabelledKeys(
  search: string,
  page: number,
  pageSize = KEY_PAGE_SIZE,
): KeyListState {
  const [state, setState] = useState<KeyListState>({
    data: null,
    loading: false,
    error: null,
  })
  const searchRef = useRef<string | null>(null)

  useEffect(() => {
    const sameSearch = searchRef.current === search
    searchRef.current = search
    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<LabelledKeyListResponse>(labelledKeysPath(search, page, pageSize))
      .then((data) => {
        if (cancelled) return
        setState((prev) => {
          if (page > 1 && sameSearch && prev.data) {
            const seen = new Set(prev.data.items.map((item) => item.key))
            return {
              data: {
                ...data,
                items: [
                  ...prev.data.items,
                  ...data.items.filter((item) => !seen.has(item.key)),
                ],
              },
              loading: false,
              error: null,
            }
          }
          return { data, loading: false, error: null }
        })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [search, page, pageSize])

  return state
}

/** The first few labelled keys matching what the reviewer has typed so far. */
export function useKeySuggestions(search: string, enabled: boolean): LabelledKeyItem[] {
  const [result, setResult] = useState<{ search: string; items: LabelledKeyItem[] } | null>(
    null,
  )

  useEffect(() => {
    if (!enabled || !search) return
    let cancelled = false
    apiFetch<LabelledKeyListResponse>(labelledKeysPath(search, 1, KEY_SUGGESTION_LIMIT))
      .then((data) => {
        if (!cancelled) setResult({ search, items: data.items })
      })
      .catch(() => {
        if (!cancelled) setResult({ search, items: [] })
      })
    return () => {
      cancelled = true
    }
  }, [search, enabled])

  // Never show the previous search's suggestions under a newer search.
  return enabled && search && result?.search === search ? result.items : []
}
