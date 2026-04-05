import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from 'react'
import { PIN_COLORS } from '../utils/colors'

export type ViewMode = 'connections' | 'heatmap' | 'ig'
export type PinSource = 'manual' | 'auto'

export interface PinMatch {
  candidateIdx: number
  score: number
  rank: number
}

export interface PinEntry {
  matches: PinMatch[]
  color: string
  source: PinSource
}

export interface HoverMatch {
  candidateIdx: number
  score: number
  rank: number
}

export interface TokenContextValue {
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  hoveredQueryTokenIdx: number | null
  setHoveredQueryTokenIdx: (idx: number | null) => void
  hoveredMatches: HoverMatch[]
  setHoveredMatches: (matches: HoverMatch[]) => void
  pinnedTokens: Map<number, PinEntry>
  pinToken: (queryIdx: number, matches: PinMatch[]) => void
  clearAllPins: () => void
  autoHighlightedTokens: Set<number>
  applyAutoHighlights: (highlights: { queryIdx: number; matches: PinMatch[] }[]) => void
  clearAutoHighlights: () => void
  hasAutoHighlights: boolean
  hasIgData: boolean
  setHasIgData: (v: boolean) => void
}

const TokenContext = createContext<TokenContextValue | null>(null)

export function TokenProvider({ children }: { children: ReactNode }) {
  const [viewMode, setViewMode] = useState<ViewMode>('connections')
  const [hoveredQueryTokenIdx, setHoveredQueryTokenIdx] = useState<number | null>(null)
  const [hoveredMatches, setHoveredMatches] = useState<HoverMatch[]>([])
  const [pinnedTokens, setPinnedTokens] = useState<Map<number, PinEntry>>(new Map())
  const [hasIgData, setHasIgData] = useState(false)
  const [autoHighlightedTokens, setAutoHighlightedTokens] = useState<Set<number>>(new Set())

  const pinToken = useCallback((queryIdx: number, matches: PinMatch[]) => {
    setPinnedTokens((prev) => {
      const next = new Map(prev)
      if (next.has(queryIdx)) {
        next.delete(queryIdx)
      } else {
        const usedColors = new Set(Array.from(next.values()).map((e) => e.color))
        const color =
          PIN_COLORS.find((c) => !usedColors.has(c)) ??
          PIN_COLORS[next.size % PIN_COLORS.length]
        next.set(queryIdx, { matches, color, source: 'manual' })
      }
      return next
    })
    setAutoHighlightedTokens((prev) => {
      if (prev.has(queryIdx)) {
        const next = new Set(prev)
        next.delete(queryIdx)
        return next
      }
      return prev
    })
  }, [])

  const clearAllPins = useCallback(() => {
    setPinnedTokens(new Map())
    setAutoHighlightedTokens(new Set())
  }, [])

  const applyAutoHighlights = useCallback(
    (highlights: { queryIdx: number; matches: PinMatch[] }[]) => {
      setPinnedTokens((prev) => {
        const hasManualPins = Array.from(prev.values()).some((e) => e.source === 'manual')
        if (hasManualPins) return prev
        const next = new Map<number, PinEntry>()
        const usedColors = new Set<string>()
        for (const h of highlights) {
          const color =
            PIN_COLORS.find((c) => !usedColors.has(c)) ??
            PIN_COLORS[next.size % PIN_COLORS.length]
          usedColors.add(color)
          next.set(h.queryIdx, { matches: h.matches, color, source: 'auto' })
        }
        return next
      })
      setAutoHighlightedTokens(new Set(highlights.map((h) => h.queryIdx)))
    },
    [],
  )

  const clearAutoHighlights = useCallback(() => {
    setPinnedTokens((prev) => {
      const next = new Map(prev)
      for (const [idx, entry] of next) {
        if (entry.source === 'auto') {
          next.delete(idx)
        }
      }
      return next
    })
    setAutoHighlightedTokens(new Set())
  }, [])

  const hasAutoHighlights = autoHighlightedTokens.size > 0

  const value: TokenContextValue = {
    viewMode,
    setViewMode,
    hoveredQueryTokenIdx,
    setHoveredQueryTokenIdx,
    hoveredMatches,
    setHoveredMatches,
    pinnedTokens,
    pinToken,
    clearAllPins,
    autoHighlightedTokens,
    applyAutoHighlights,
    clearAutoHighlights,
    hasAutoHighlights,
    hasIgData,
    setHasIgData,
  }

  return <TokenContext.Provider value={value}>{children}</TokenContext.Provider>
}

export function useTokens(): TokenContextValue {
  const ctx = useContext(TokenContext)
  if (!ctx) {
    throw new Error('useTokens must be used within a TokenProvider')
  }
  return ctx
}
