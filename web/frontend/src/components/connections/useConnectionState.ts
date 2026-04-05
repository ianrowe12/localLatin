import { useMemo } from 'react'
import { useTokens } from '../../contexts/TokenContext'
import { HOVER_COLOR } from '../../utils/colors'

export interface Connection {
  id: string
  sourceId: string // "query:N"
  targetId: string // "candidate:N"
  color: string
  score: number
  rank: number // 0-based among top matches
  isPinned: boolean
  isAutoHighlighted: boolean
}

export function useConnectionState(): { activeConnections: Connection[] } {
  const { hoveredQueryTokenIdx, hoveredMatches, pinnedTokens } = useTokens()

  return useMemo(() => {
    const conns: Connection[] = []

    // Pinned connections
    for (const [queryIdx, pin] of pinnedTokens.entries()) {
      for (const match of pin.matches) {
        conns.push({
          id: `pin:${queryIdx}->${match.candidateIdx}`,
          sourceId: `query:${queryIdx}`,
          targetId: `candidate:${match.candidateIdx}`,
          color: pin.color,
          score: match.score,
          rank: match.rank,
          isPinned: true,
          isAutoHighlighted: pin.source === 'auto',
        })
      }
    }

    // Hover connections (only if not already pinned)
    if (
      hoveredQueryTokenIdx !== null &&
      !pinnedTokens.has(hoveredQueryTokenIdx)
    ) {
      for (const match of hoveredMatches) {
        conns.push({
          id: `hover:${hoveredQueryTokenIdx}->${match.candidateIdx}`,
          sourceId: `query:${hoveredQueryTokenIdx}`,
          targetId: `candidate:${match.candidateIdx}`,
          color: HOVER_COLOR,
          score: match.score,
          rank: match.rank,
          isPinned: false,
          isAutoHighlighted: false,
        })
      }
    }

    return { activeConnections: conns }
  }, [hoveredQueryTokenIdx, hoveredMatches, pinnedTokens])
}
