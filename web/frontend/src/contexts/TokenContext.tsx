import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from 'react'
import { PIN_COLORS } from '../utils/colors'

export type ViewMode = 'connections' | 'heatmap' | 'ig'

export interface PinMatch {
  candidateIdx: number
  score: number
  rank: number
}

export interface PinEntry {
  matches: PinMatch[]
  color: string
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
        next.set(queryIdx, { matches, color })
      }
      return next
    })
  }, [])

  const clearAllPins = useCallback(() => {
    setPinnedTokens(new Map())
  }, [])

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
