export interface Point {
  x: number
  y: number
}

/**
 * Generate SVG cubic bezier path between two points.
 * Source exits horizontally right, target enters horizontally left.
 * Creates a clean S-curve.
 */
export function computeBezierPath(src: Point, tgt: Point): string {
  const dx = tgt.x - src.x
  const dy = tgt.y - src.y
  const verticalStretch = Math.min(Math.abs(dy) * 0.15, 80)
  const cpOffset = Math.max(40, Math.abs(dx) * 0.35) + verticalStretch

  const cp1: Point = { x: src.x + cpOffset, y: src.y + dy * 0.1 }
  const cp2: Point = { x: tgt.x - cpOffset, y: tgt.y - dy * 0.1 }

  return `M ${src.x} ${src.y} C ${cp1.x} ${cp1.y}, ${cp2.x} ${cp2.y}, ${tgt.x} ${tgt.y}`
}

/**
 * Convert viewport coordinates to SVG-local coordinates.
 */
export function viewportToSvg(
  vx: number,
  vy: number,
  containerRect: DOMRect,
): Point {
  return { x: vx - containerRect.left, y: vy - containerRect.top }
}

/**
 * Check if a DOMRect is at least partially visible within container.
 */
export function isRectVisible(
  elementRect: DOMRect,
  containerRect: DOMRect,
): boolean {
  return (
    elementRect.bottom > containerRect.top &&
    elementRect.top < containerRect.bottom &&
    elementRect.right > containerRect.left &&
    elementRect.left < containerRect.right
  )
}
