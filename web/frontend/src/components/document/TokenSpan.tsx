import { memo } from 'react'
import { similarityToColor, importanceToColor } from '../../utils/colors'
import { useApp } from '../../contexts/AppContext'

interface TokenSpanProps {
  token: { text: string; index: number; category: string }
  side: 'query' | 'candidate'
  isHovered: boolean
  isPinned: boolean
  pinColor?: string
  isAutoHighlighted?: boolean
  highlightScore?: number
  colorPalette?: 'blue' | 'orange'
  spanRef?: (el: HTMLSpanElement | null) => void
  /**
   * Called with this token's index, rather than through a closure the caller
   * built around it.
   *
   * The distinction is what keeps a memoised span honest. A per-token arrow is
   * a fresh function on every render, so a comparator either ignores it -- and
   * keeps whichever evidence the previous render captured -- or never matches
   * and makes the memo useless. Taking the index lets the caller pass ONE
   * function whose identity changes exactly when its behaviour does.
   */
  onMouseEnter?: (index: number) => void
  onMouseLeave?: (index: number) => void
  onClick?: (index: number) => void
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
  isAutoHighlighted,
  highlightScore,
  colorPalette,
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
    if (isAutoHighlighted) {
      ringClass = ''
      bgStyle = {
        backgroundColor: pinColor + '1A',
        outline: `2px dashed ${pinColor}`,
        outlineOffset: '-1px',
      }
    } else {
      ringClass = 'ring-2'
      bgStyle = {
        backgroundColor: pinColor + '26',
        boxShadow: `0 0 0 2px ${pinColor}`,
      }
    }
  } else if (highlightScore != null && highlightScore > 0) {
    bgStyle = {
      backgroundColor: colorPalette
        ? importanceToColor(highlightScore, colorPalette, theme)
        : similarityToColor(highlightScore, theme),
      boxShadow: 'inset 0 -2px 0 rgba(68, 64, 60, 0.45)',
    }
    extraClass = 'decoration-dotted'
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

  const interactive = onMouseEnter != null || onClick != null
  const accessibleLabel =
    highlightScore != null && highlightScore > 0
      ? `${side} token ${token.text}, highlighted evidence`
      : `${side} token ${token.text}`

  return (
    <span
      ref={spanRef}
      className={`
        inline-block rounded px-0.5 py-[1px] mx-[1px]
        ${interactive ? 'cursor-pointer' : ''}
        transition-all duration-150 font-latin
        ${categoryClasses(token.category)}
        ${ringClass}
        ${extraClass}
      `}
      style={bgStyle}
      tabIndex={interactive ? 0 : undefined}
      aria-label={interactive ? accessibleLabel : undefined}
      onMouseEnter={onMouseEnter && (() => onMouseEnter(token.index))}
      onMouseLeave={onMouseLeave && (() => onMouseLeave(token.index))}
      onFocus={onMouseEnter && (() => onMouseEnter(token.index))}
      onBlur={onMouseLeave && (() => onMouseLeave(token.index))}
      onKeyDown={(event) => {
        if (!onClick) return
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onClick(token.index)
        }
      }}
      onClick={onClick && (() => onClick(token.index))}
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
    // The WORD, not just its position. Until issue #163 the candidate panel
    // showed one fixed file per pair, so a given index could only change text
    // by way of a key change that remounted the whole panel and skipped this
    // comparator entirely. A member selector breaks that: index 4 of the
    // group's second witness is a different word from index 4 of its first,
    // and comparing positions alone left the previous manuscript's words on
    // screen under the new one's filename.
    prev.token.text === next.token.text &&
    // Drives the span's own colour classes, and travels with the text.
    prev.token.category === next.token.category &&
    prev.side === next.side &&
    prev.isHovered === next.isHovered &&
    prev.isPinned === next.isPinned &&
    prev.pinColor === next.pinColor &&
    prev.isAutoHighlighted === next.isAutoHighlighted &&
    prev.highlightScore === next.highlightScore &&
    prev.colorPalette === next.colorPalette &&
    prev.spanRef === next.spanRef &&
    // What the span DOES, not only what it looks like. A hover handler carries
    // the evidence it will write -- the token map it was built from -- so two
    // renders can agree on every visible prop and still disagree about which
    // witness the next hover reports. Skipping the update there left a live
    // handler holding the previous member's matches, and a fresh hover or
    // keyboard focus wrote them back into the current member's state.
    //
    // Comparable because these arrive as ONE function per panel rather than a
    // per-token arrow: the identity changes exactly when the map behind it
    // does, so this both catches the stale case and keeps the memo working.
    prev.onMouseEnter === next.onMouseEnter &&
    prev.onMouseLeave === next.onMouseLeave &&
    prev.onClick === next.onClick
  )
})

TokenSpan.displayName = 'TokenSpan'

export default TokenSpan
