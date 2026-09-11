import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { PIN_COLORS } from '../utils/colors'
import type { AttributionMethod } from '../api/tokenMap'

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
  // The post-processing variant is NOT held here: it is a single app-wide
  // choice (AppContext.activeVariant) shared with the prediction list, so the
  // highlights can never disagree with the ranking the reviewer is judging.
  selectedMethod: AttributionMethod | null
  /**
   * What the DISPLAYED pair offers, which is not the same thing as what the
   * last publication said (issue #163). Empty whenever the two disagree.
   */
  availableMethods: AttributionMethod[]
  setSelectedMethod: (m: AttributionMethod) => void
  /**
   * Publish an artifact's methods together with the pair they describe.
   *
   * The owner is not decoration. The panel that loads artifacts is not the
   * only consumer of this list -- `AttributionMethodSelector` sits outside it,
   * in the sidebar, where no scope inside the panel can reach -- so the list
   * has to carry enough to be checked against the pair actually on screen.
   */
  setAvailableMethods: (methods: AttributionMethod[], owner: string) => void
  /** The pair whose words are on screen; see `announceDisplayedPair`. */
  displayedPair: string | null
  /**
   * Name the pair on screen, from the same update that puts it there.
   *
   * Applicability used to reach outside consumers through a passive effect,
   * one commit after the words changed, so the first committed frame of a new
   * witness carried the previous witness's method controls. That frame is one
   * a reviewer reads. Announcing from the event that changes the witness makes
   * the mismatch visible in the same render pass: this provider is an ancestor
   * of both the panel and the sidebar, so both re-render together, and the
   * controls are gone in the very frame the new words appear.
   */
  announceDisplayedPair: (owner: string) => void
}

const TokenContext = createContext<TokenContextValue | null>(null)

const NO_METHODS: AttributionMethod[] = []
const NO_ATTRIBUTION: { owner: string | null; methods: AttributionMethod[] } = {
  owner: null,
  methods: NO_METHODS,
}

function sameMethods(a: AttributionMethod[], b: AttributionMethod[]): boolean {
  return a.length === b.length && a.every((m, i) => m === b[i])
}

export function TokenProvider({ children }: { children: ReactNode }) {
  const [viewMode, setViewMode] = useState<ViewMode>('connections')
  const [hoveredQueryTokenIdx, setHoveredQueryTokenIdx] = useState<number | null>(null)
  const [hoveredMatches, setHoveredMatches] = useState<HoverMatch[]>([])
  const [pinnedTokens, setPinnedTokens] = useState<Map<number, PinEntry>>(new Map())
  const [hasIgData, setHasIgData] = useState(false)
  const [autoHighlightedTokens, setAutoHighlightedTokens] = useState<Set<number>>(new Set())
  const [attribution, setAttribution] = useState<{
    owner: string | null
    methods: AttributionMethod[]
  }>(NO_ATTRIBUTION)
  const [displayedPair, setDisplayedPair] = useState<string | null>(null)
  // Seeded with the first method rather than null so the very first token-map
  // fetch can send ?method= and stay small; the effect below corrects it if
  // this pair has no IG matrices.
  const [selectedMethod, setSelectedMethod] = useState<AttributionMethod | null>('ig')

  const setAvailableMethods = useCallback(
    (methods: AttributionMethod[], owner: string) => {
      setAttribution((prev) =>
        prev.owner === owner && sameMethods(prev.methods, methods)
          ? prev
          : { owner, methods },
      )
    },
    [],
  )

  const announceDisplayedPair = useCallback((owner: string) => {
    setDisplayedPair((prev) => (prev === owner ? prev : owner))
  }, [])

  // The gate, and the only reading of the list anything else sees. A
  // publication for another pair is not a weaker claim about this one, it is a
  // claim about something else, so it counts for nothing here. The artifact
  // behind it is untouched and stays cached, so the pair it does describe
  // still costs no request to return to.
  const availableMethods =
    attribution.owner !== null && attribution.owner === displayedPair
      ? attribution.methods
      : NO_METHODS

  // Keep selectedMethod consistent with availableMethods. If the current
  // selection is no longer available (or is null), reset to the first
  // available method. This runs as an effect to avoid stale closures.
  // An empty list means "no pair loaded yet" (or no attribution data at all),
  // which must not wipe the 'ig' seed the first fetch relies on.
  useEffect(() => {
    if (availableMethods.length === 0) return
    setSelectedMethod((prev) =>
      prev && availableMethods.includes(prev) ? prev : availableMethods[0],
    )
  }, [availableMethods])

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
    selectedMethod,
    availableMethods,
    setSelectedMethod,
    setAvailableMethods,
    displayedPair,
    announceDisplayedPair,
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

const NO_PINS: Map<number, PinEntry> = new Map()
const NO_AUTO_HIGHLIGHTS: Set<number> = new Set()
const NO_HOVER_MATCHES: HoverMatch[] = []

/**
 * Confine pins, auto-highlights and hover to one witness (issue #163).
 *
 * The store above is app-wide, and rightly so: the attribution method and the
 * view mode are preferences a reviewer sets once. But a pin is a claim about
 * a PAIR -- "query token 4 matches candidate token 9" -- and candidate token
 * 9 is a different word the moment another member of the group is on screen.
 * Remounting `TokenProvider` per witness would scope those claims correctly
 * and throw away the preferences with them, so this gates instead.
 *
 * The gate is applied while rendering, not in an effect, because an effect
 * runs a commit too late: the first paint after a witness change would carry
 * the previous witness's marks, and that paint is one a reviewer reads. An
 * effect then empties the underlying store, so nothing accumulates and
 * returning to the first witness does not resurrect them.
 *
 * `ownerRef` holds the witness the marks BELONG to, and advances only in that
 * effect -- never while rendering. Advancing it on request would lift the gate
 * in the same render that asked for a new witness, while the old pins were
 * still in the store, which is the stale frame this exists to prevent.
 */
export function WitnessTokenScope({
  witness,
  children,
}: {
  witness: string
  children: ReactNode
}) {
  const outer = useTokens()
  const ownerRef = useRef(witness)
  const stale = ownerRef.current !== witness

  const { clearAllPins, setHoveredQueryTokenIdx, setHoveredMatches } = outer
  useEffect(() => {
    if (ownerRef.current === witness) return
    ownerRef.current = witness
    clearAllPins()
    // `clearAllPins` leaves hover alone, and hover is just as much about the
    // pair: a pointer resting over a query token holds matches into candidate
    // indices that no longer name the same words.
    setHoveredQueryTokenIdx(null)
    setHoveredMatches([])
  }, [witness, clearAllPins, setHoveredQueryTokenIdx, setHoveredMatches])

  // Only the five fields that describe a pair are gated. Everything else --
  // the attribution method, the view mode, the writers themselves -- is a
  // preference or a capability and passes straight through.
  const value = useMemo<TokenContextValue>(
    () => ({
      ...outer,
      pinnedTokens: stale ? NO_PINS : outer.pinnedTokens,
      autoHighlightedTokens: stale ? NO_AUTO_HIGHLIGHTS : outer.autoHighlightedTokens,
      hasAutoHighlights: stale ? false : outer.hasAutoHighlights,
      hoveredQueryTokenIdx: stale ? null : outer.hoveredQueryTokenIdx,
      hoveredMatches: stale ? NO_HOVER_MATCHES : outer.hoveredMatches,
    }),
    [outer, stale],
  )

  return <TokenContext.Provider value={value}>{children}</TokenContext.Provider>
}
