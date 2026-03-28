import { memo } from 'react'
import { similarityToColor } from '../../utils/colors'
import { useApp } from '../../contexts/AppContext'

interface TokenSpanProps {
  token: { text: string; index: number; category: string }
  side: 'query' | 'candidate'
  isHovered: boolean
  isPinned: boolean
  pinColor?: string
  highlightScore?: number
  spanRef?: (el: HTMLSpanElement | null) => void
  onMouseEnter?: () => void
  onMouseLeave?: () => void
  onClick?: () => void
}

function categoryClasses(category: string): string {
  switch (category) {
    case 'content':
      return 'font-medium text-stone-800 dark:text-stone-100'
    case 'punctuation':
      return 'font-normal text-stone-400 dark:text-stone-500'
    case 'empty':
    case 'short_subword':
      return 'text-stone-300 dark:text-stone-600 text-[0.85em]'
    case 'number':
      return 'font-medium text-accent/80'
    default:
      return 'text-stone-800 dark:text-stone-100'
  }
}

function TokenSpanInner({
  token,
  side,
  isHovered,
  isPinned,
  pinColor,
  highlightScore,
  spanRef,
  onMouseEnter,
  onMouseLeave,
  onClick,
}: TokenSpanProps) {
  const { theme } = useApp()

  // Build dynamic styles
  let ringClass = ''
  let bgStyle: React.CSSProperties = {}
  let extraClass = ''

  if (isHovered && side === 'query') {
    ringClass = 'ring-2 ring-accent/60 animate-pulse-glow'
    bgStyle = { backgroundColor: 'rgba(99, 102, 241, 0.1)' }
  } else if (isPinned && pinColor) {
    ringClass = 'ring-2'
    bgStyle = {
      backgroundColor: pinColor + '26', // ~15% opacity
      boxShadow: `0 0 0 2px ${pinColor}`,
    }
  } else if (highlightScore != null && highlightScore > 0) {
    bgStyle = { backgroundColor: similarityToColor(highlightScore, theme) }
  }

  // Dimming: if a query token is hovered but this candidate token is not highlighted
  if (
    side === 'candidate' &&
    !isHovered &&
    !isPinned &&
    highlightScore != null &&
    highlightScore < 0.1
  ) {
    extraClass = 'opacity-40 transition-opacity'
  }

  return (
    <span
      ref={spanRef}
      className={`
        inline-block rounded px-0.5 py-[1px] mx-[1px] cursor-pointer
        transition-all duration-150 font-latin
        ${categoryClasses(token.category)}
        ${ringClass}
        ${extraClass}
      `}
      style={bgStyle}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      data-token-idx={token.index}
      data-side={side}
    >
      {token.text}
    </span>
  )
}

const TokenSpan = memo(TokenSpanInner, (prev, next) => {
  return (
    prev.token.index === next.token.index &&
    prev.side === next.side &&
    prev.isHovered === next.isHovered &&
    prev.isPinned === next.isPinned &&
    prev.pinColor === next.pinColor &&
    prev.highlightScore === next.highlightScore &&
    prev.spanRef === next.spanRef
  )
})

TokenSpan.displayName = 'TokenSpan'

export default TokenSpan
