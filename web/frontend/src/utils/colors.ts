/** 8 distinct pin colors for multi-token pinning */
export const PIN_COLORS = [
  '#f97316', // orange
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
  '#f59e0b', // amber
  '#6366f1', // indigo
  '#14b8a6', // teal
] as const

export const HOVER_COLOR = '#3b82f6' // blue-500

/**
 * Map a similarity score [0,1] to an rgba background color.
 * Perceptually uniform: transparent → soft blue → vivid indigo → gold
 */
export function similarityToColor(score: number, mode: 'light' | 'dark' = 'light'): string {
  if (score < 0.1) return 'transparent'
  if (score < 0.3) {
    const a = (score - 0.1) / 0.2
    return mode === 'light'
      ? `rgba(147, 197, 253, ${a * 0.3})`  // blue-300
      : `rgba(147, 197, 253, ${a * 0.2})`
  }
  if (score < 0.6) {
    const a = (score - 0.3) / 0.3
    return mode === 'light'
      ? `rgba(99, 102, 241, ${0.15 + a * 0.35})`  // indigo-500
      : `rgba(129, 140, 248, ${0.15 + a * 0.35})`  // indigo-400
  }
  if (score < 0.8) {
    const a = (score - 0.6) / 0.2
    return mode === 'light'
      ? `rgba(79, 70, 229, ${0.4 + a * 0.3})`  // indigo-600
      : `rgba(165, 180, 252, ${0.3 + a * 0.3})`  // indigo-300
  }
  // >0.8: gold
  const a = Math.min((score - 0.8) / 0.2, 1)
  return mode === 'light'
    ? `rgba(245, 158, 11, ${0.4 + a * 0.4})`  // amber-500
    : `rgba(251, 191, 36, ${0.3 + a * 0.4})`   // amber-400
}

/**
 * Map an importance score [0,1] to a palette-specific rgba background color.
 * Blue palette for query tokens, orange/amber for candidate tokens.
 */
export function importanceToColor(
  score: number,
  palette: 'blue' | 'orange',
  mode: 'light' | 'dark' = 'light',
): string {
  if (score < 0.15) return 'transparent'
  if (palette === 'blue') {
    // Query-side: blue family
    if (score < 0.3) {
      const a = (score - 0.15) / 0.15
      return mode === 'light'
        ? `rgba(147, 197, 253, ${a * 0.25})`   // blue-300
        : `rgba(147, 197, 253, ${a * 0.18})`
    }
    if (score < 0.6) {
      const a = (score - 0.3) / 0.3
      return mode === 'light'
        ? `rgba(59, 130, 246, ${0.12 + a * 0.25})`  // blue-500
        : `rgba(96, 165, 250, ${0.12 + a * 0.25})`  // blue-400
    }
    if (score < 0.8) {
      const a = (score - 0.6) / 0.2
      return mode === 'light'
        ? `rgba(37, 99, 235, ${0.3 + a * 0.25})`    // blue-600
        : `rgba(147, 197, 253, ${0.25 + a * 0.25})`  // blue-300
    }
    const a = Math.min((score - 0.8) / 0.2, 1)
    return mode === 'light'
      ? `rgba(29, 78, 216, ${0.45 + a * 0.3})`      // blue-700
      : `rgba(191, 219, 254, ${0.35 + a * 0.3})`    // blue-200
  }
  // Orange/amber palette for candidate-side
  if (score < 0.3) {
    const a = (score - 0.15) / 0.15
    return mode === 'light'
      ? `rgba(253, 186, 116, ${a * 0.25})`   // orange-300
      : `rgba(253, 186, 116, ${a * 0.18})`
  }
  if (score < 0.6) {
    const a = (score - 0.3) / 0.3
    return mode === 'light'
      ? `rgba(249, 115, 22, ${0.12 + a * 0.25})`  // orange-500
      : `rgba(251, 146, 60, ${0.12 + a * 0.25})`  // orange-400
  }
  if (score < 0.8) {
    const a = (score - 0.6) / 0.2
    return mode === 'light'
      ? `rgba(234, 88, 12, ${0.3 + a * 0.25})`     // orange-600
      : `rgba(253, 186, 116, ${0.25 + a * 0.25})`  // orange-300
  }
  const a = Math.min((score - 0.8) / 0.2, 1)
  return mode === 'light'
    ? `rgba(194, 65, 12, ${0.45 + a * 0.3})`       // orange-700
    : `rgba(254, 215, 170, ${0.35 + a * 0.3})`     // orange-200
}

/**
 * Get a similarity score as a stroke color for SVG lines.
 */
export function similarityToStroke(score: number): string {
  if (score < 0.3) return '#93c5fd'  // blue-300
  if (score < 0.6) return '#6366f1'  // indigo-500
  if (score < 0.8) return '#4f46e5'  // indigo-600
  return '#f59e0b'                     // amber-500
}
