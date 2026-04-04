import { useMemo, useCallback } from 'react'
import { useTokens, type PinMatch } from '../../contexts/TokenContext'
import { useTokenRefs } from '../connections/TokenRefRegistry'
import DocumentHeader from './DocumentHeader'
import TokenSpan from './TokenSpan'
import SkeletonLoader from '../common/SkeletonLoader'
import EmptyState from '../empty/EmptyState'

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
  tokens?: TokenLike[]
  tokenMap?: TokenMapLike | null
  loading?: boolean
  scrollRef?: React.RefObject<HTMLDivElement>
}

export default function DocumentPanel({
  side,
  filename,
  dirLabel,
  score,
  tokens,
  tokenMap,
  loading,
  scrollRef,
}: DocumentPanelProps) {
  const {
    hoveredQueryTokenIdx,
    setHoveredQueryTokenIdx,
    setHoveredMatches,
    pinnedTokens,
    pinToken,
    viewMode,
  } = useTokens()

  const tokenRefs = useTokenRefs()

  // Compute highlight scores for candidate tokens
  const candidateHighlights = useMemo(() => {
    if (side !== 'candidate' || !tokens || !tokenMap?.similarity_matrix) {
      return null
    }

    const matrix = tokenMap.similarity_matrix
    const scores = new Float64Array(tokens.length)

    if (viewMode === 'heatmap') {
      // Max across all query tokens for each candidate token
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
    } else if (hoveredQueryTokenIdx != null) {
      // Highlight by the hovered query token
      const row = matrix[hoveredQueryTokenIdx]
      if (row) {
        for (let ci = 0; ci < tokens.length && ci < row.length; ci++) {
          scores[ci] = row[ci]
        }
      }
    }

    return scores
  }, [side, tokens, tokenMap, hoveredQueryTokenIdx, viewMode])

  // Handlers for query tokens
  const handleQueryMouseEnter = useCallback(
    (idx: number) => {
      if (side === 'query') {
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
    [side, setHoveredQueryTokenIdx, setHoveredMatches, tokenMap],
  )

  const handleQueryMouseLeave = useCallback(() => {
    if (side === 'query') {
      setHoveredQueryTokenIdx(null)
      setHoveredMatches([])
    }
  }, [side, setHoveredQueryTokenIdx, setHoveredMatches])

  const handleQueryClick = useCallback(
    (idx: number) => {
      if (side !== 'query' || !tokenMap?.similarity_matrix) return
      const row = tokenMap.similarity_matrix[idx]
      if (!row) return

      // Build top matches sorted by score descending
      const matches: PinMatch[] = row
        .map((s, ci) => ({ candidateIdx: ci, score: s, rank: 0 }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .map((m, rank) => ({ ...m, rank: rank + 1 }))

      pinToken(idx, matches)
    },
    [side, tokenMap, pinToken],
  )

  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col h-full">
        {filename && (
          <DocumentHeader
            filename={filename}
            dirLabel={dirLabel}
            score={score}
            side={side}
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
        side={side}
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

            let highlightScore: number | undefined
            if (side === 'candidate' && candidateHighlights) {
              highlightScore = candidateHighlights[idx]
              // Dim non-matching tokens when a query token is hovered
              if (
                isAnyHovered &&
                viewMode !== 'heatmap' &&
                (highlightScore == null || highlightScore < 0.1) &&
                !isCandidatePinned
              ) {
                highlightScore = 0
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
                highlightScore={highlightScore}
                spanRef={tokenRefs.registerRef(tokenRefId)}
                onMouseEnter={
                  side === 'query' ? () => handleQueryMouseEnter(idx) : undefined
                }
                onMouseLeave={
                  side === 'query' ? handleQueryMouseLeave : undefined
                }
                onClick={
                  side === 'query' ? () => handleQueryClick(idx) : undefined
                }
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
