import { useMemo, useCallback } from 'react'
import { useTokens, type PinEntry } from '../../contexts/TokenContext'
import { useTokenRefs } from '../connections/TokenRefRegistry'
import DocumentHeader from './DocumentHeader'
import TokenSpan from './TokenSpan'
import SkeletonLoader from '../common/SkeletonLoader'
import EmptyState from '../empty/EmptyState'
import type { DocumentProvenance } from '../../utils/documentProvenance'

interface TokenLike {
  text: string
  index: number
  category: string
}

export interface TokenMapLike {
  similarity_matrix?: number[][]
  top_matches?: Record<string, { candidate_idx: number; score: number }[]>
  ig_weighted_matrix?: number[][] | null
}

interface DocumentPanelProps {
  side: 'query' | 'candidate'
  filename?: string
  dirLabel?: string
  score?: number
  rank?: number
  tokens?: TokenLike[]
  tokenMap?: TokenMapLike | null
  loading?: boolean
  scrollRef?: React.RefObject<HTMLDivElement>
  /**
   * Kind of text on the candidate side (issue #162), forwarded to
   * DocumentHeader. Pass the current prediction's provenance -- see
   * `provenanceOf` in utils/documentProvenance -- rather than inferring it from
   * the label, the rank or the score. Ignored on the query side, which is
   * always the query witness.
   */
  provenance?: DocumentProvenance
  /** Status pills for the document itself; forwarded to DocumentHeader. */
  badge?: React.ReactNode
  /**
   * The pair whose words these are (issue #163).
   *
   * Registered with every token element, so geometry for this panel can only
   * be read back by a consumer asking about this same pair. A panel being
   * animated away keeps the value it was rendered with, which is the whole
   * point: its words are still on screen and still registered, and they are
   * no longer the words the rest of the app is reasoning about.
   *
   * It also decides which transient token state this panel may read at all.
   * See `owned` below.
   */
  evidenceOwner: string
}

const NO_PINS: Map<number, PinEntry> = new Map()
const NO_AUTO_HIGHLIGHTS: Set<number> = new Set()

export default function DocumentPanel({
  side,
  filename,
  dirLabel,
  score,
  rank,
  tokens,
  tokenMap,
  loading,
  scrollRef,
  provenance,
  badge,
  evidenceOwner,
}: DocumentPanelProps) {
  const {
    hoveredQueryTokenIdx: scopeHoveredQueryTokenIdx,
    setHoveredQueryTokenIdx,
    setHoveredMatches,
    pinnedTokens: scopePinnedTokens,
    viewMode,
    autoHighlightedTokens: scopeAutoHighlightedTokens,
    scopeWitness,
  } = useTokens()

  /**
   * Are the transient marks in this scope about THESE words? (issue #163)
   *
   * `WitnessTokenScope` is outside the animated subtree, so it advances to the
   * incoming pair while this panel is still exiting with its old props. Context
   * still re-renders this panel, and a re-render recomputes every highlight,
   * pin and dim prop from scratch -- the memo comparison in `TokenSpan` runs
   * after that, on values already computed for the wrong pair, so owning the
   * ref callbacks cannot reach it. Reading a hover into the incoming text would
   * repaint a word in the OUTGOING manuscript, which is the claim "this old
   * word is what your query matches", made about a document the reviewer has
   * already left.
   *
   * What this does not touch: the panel's own `tokens` and `tokenMap`. Those
   * are props, they came with the panel, and they are honest evidence about its
   * own pair -- so the exiting text keeps its own shading and its crossfade
   * looks like a fade of what was there, not a blank.
   *
   * Null means there is no scope above (a panel mounted alone in a test), and
   * gates nothing.
   */
  const owned = scopeWitness === null || scopeWitness === evidenceOwner
  const hoveredQueryTokenIdx = owned ? scopeHoveredQueryTokenIdx : null
  const pinnedTokens = owned ? scopePinnedTokens : NO_PINS
  const autoHighlightedTokens = owned ? scopeAutoHighlightedTokens : NO_AUTO_HIGHLIGHTS

  const tokenRefs = useTokenRefs()

  // Compute highlight scores for candidate tokens
  const candidateHighlights = useMemo(() => {
    if (side !== 'candidate' || !tokens || !tokenMap?.similarity_matrix) {
      return null
    }

    const matrix = tokenMap.similarity_matrix
    const scores = new Float64Array(tokens.length)

    // Always compute column max (max similarity from any query token)
    for (let ci = 0; ci < tokens.length; ci++) {
      let maxScore = 0
      for (let qi = 0; qi < matrix.length; qi++) {
        const row = matrix[qi]
        if (row && ci < row.length && row[ci] > maxScore) {
          maxScore = row[ci]
        }
      }
      scores[ci] = maxScore
    }

    return scores
  }, [side, tokens, tokenMap])

  // Compute highlight scores for query tokens (row-max of similarity matrix)
  const queryHighlights = useMemo(() => {
    if (side !== 'query' || !tokens || !tokenMap?.similarity_matrix) {
      return null
    }
    const matrix = tokenMap.similarity_matrix
    const scores = new Float64Array(tokens.length)
    for (let qi = 0; qi < tokens.length && qi < matrix.length; qi++) {
      const row = matrix[qi]
      if (!row) continue
      let maxScore = 0
      for (let ci = 0; ci < row.length; ci++) {
        if (row[ci] > maxScore) maxScore = row[ci]
      }
      scores[qi] = maxScore
    }
    return scores
  }, [side, tokens, tokenMap])

  // Handlers for query tokens. One function per panel, taking the token index,
  // so `TokenSpan` can compare them: a per-token arrow would be a new value on
  // every render and force the memo to ignore it, which is how a hover kept
  // reporting the previous witness's matches. The identity changes exactly
  // when `tokenMap` does, which is exactly when the behaviour changes.
  //
  // They are gated by `owned` for the same reason the reads are: a pointer
  // resting on a retained panel's token would otherwise publish matches into
  // candidate indices that name different words in the pair now on screen.
  const handleQueryMouseEnter = useCallback(
    (idx: number) => {
      if (side === 'query' && owned) {
        setHoveredQueryTokenIdx(idx)
        // Populate hover matches from tokenMap top_matches
        const topMatches = tokenMap?.top_matches?.[String(idx)]
        if (topMatches) {
          setHoveredMatches(
            topMatches.map((m, rank) => ({
              candidateIdx: m.candidate_idx,
              score: m.score,
              rank,
            })),
          )
        } else {
          setHoveredMatches([])
        }
      }
    },
    [side, owned, setHoveredQueryTokenIdx, setHoveredMatches, tokenMap],
  )

  const handleQueryMouseLeave = useCallback(() => {
    if (side === 'query' && owned) {
      setHoveredQueryTokenIdx(null)
      setHoveredMatches([])
    }
  }, [side, owned, setHoveredQueryTokenIdx, setHoveredMatches])

  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col h-full">
        {filename && (
          <DocumentHeader
            filename={filename}
            dirLabel={dirLabel}
            score={score}
            rank={rank}
            side={side}
            provenance={provenance}
            badge={badge}
          />
        )}
        <SkeletonLoader />
      </div>
    )
  }

  // Empty state
  if (!tokens || !filename) {
    return (
      <div className="flex flex-col h-full">
        <EmptyState
          message={
            side === 'query'
              ? 'Select a query to begin reviewing'
              : 'Candidate will appear here'
          }
        />
      </div>
    )
  }

  const isAnyHovered = hoveredQueryTokenIdx != null

  return (
    <div className="flex flex-col h-full">
      <DocumentHeader
        filename={filename}
        dirLabel={dirLabel}
        score={score}
        rank={rank}
        side={side}
        provenance={provenance}
        badge={badge}
      />
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        <div className="leading-relaxed">
          {tokens.map((token) => {
            const idx = token.index
            const pinEntry = side === 'query' ? pinnedTokens.get(idx) : undefined

            // For candidate side, check if this token is pinned via any pin's matches
            let candidatePinColor: string | undefined
            let isCandidatePinned = false
            if (side === 'candidate') {
              for (const [, entry] of pinnedTokens) {
                if (entry.matches.some((m) => m.candidateIdx === idx)) {
                  candidatePinColor = entry.color
                  isCandidatePinned = true
                  break
                }
              }
            }

            const isHovered =
              side === 'query'
                ? hoveredQueryTokenIdx === idx
                : false

            const isPinned = side === 'query' ? !!pinEntry : isCandidatePinned
            const pinColor = side === 'query' ? pinEntry?.color : candidatePinColor

            let isAutoHighlighted = side === 'query'
              ? autoHighlightedTokens.has(idx)
              : false
            if (side === 'candidate' && isCandidatePinned) {
              for (const [queryIdx, entry] of pinnedTokens) {
                if (entry.matches.some((m) => m.candidateIdx === idx) && autoHighlightedTokens.has(queryIdx)) {
                  isAutoHighlighted = true
                  break
                }
              }
            }

            let highlightScore: number | undefined
            if (side === 'query' && queryHighlights) {
              const raw = queryHighlights[idx]
              // Skip non-content tokens; apply power curve to compress middle range
              highlightScore = (token.category === 'empty' || token.category === 'short_subword')
                ? undefined
                : raw < 0.15 ? 0 : Math.pow(raw, 1.5)
            }
            if (side === 'candidate' && candidateHighlights) {
              const raw = candidateHighlights[idx]
              highlightScore = (token.category === 'empty' || token.category === 'short_subword')
                ? undefined
                : raw < 0.15 ? 0 : Math.pow(raw, 1.5)
              if (
                isAnyHovered &&
                viewMode !== 'heatmap' &&
                !isCandidatePinned
              ) {
                const hoverRow = tokenMap?.similarity_matrix?.[hoveredQueryTokenIdx!]
                const hoverScore = hoverRow?.[idx] ?? 0
                if (hoverScore < 0.1) {
                  highlightScore = (highlightScore ?? 0) * 0.2
                }
              }
            }

            const tokenRefId = `${side}:${idx}`

            return (
              <TokenSpan
                key={idx}
                token={token}
                side={side}
                isHovered={isHovered}
                isPinned={isPinned}
                pinColor={pinColor}
                isAutoHighlighted={isAutoHighlighted}
                highlightScore={highlightScore}
                colorPalette={side === 'query' ? 'blue' : 'orange'}
                spanRef={tokenRefs.registerRef(evidenceOwner, tokenRefId)}
                onMouseEnter={side === 'query' ? handleQueryMouseEnter : undefined}
                onMouseLeave={side === 'query' ? handleQueryMouseLeave : undefined}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
