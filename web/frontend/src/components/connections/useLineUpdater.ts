import { useState, useRef, useCallback, useEffect } from 'react'
import type { Connection } from './useConnectionState'
import type { TokenRefRegistryValue } from './TokenRefRegistry'
import { computeBezierPath, viewportToSvg, isRectVisible } from './bezierUtils'

/**
 * @param owner  The pair these lines are about. Token geometry is only read
 *   from elements registered by that same pair, so a line is never measured
 *   against a panel that is merely still on screen.
 * @param ownerVersion  Changes when a side's registered owner does, which is
 *   the one event that invalidates every cached path without changing the
 *   connections, the container size or the scroll position.
 */
export function useLineUpdater(
  connections: Connection[],
  containerRef: React.RefObject<HTMLDivElement>,
  leftPanelRef: React.RefObject<HTMLDivElement>,
  rightPanelRef: React.RefObject<HTMLDivElement>,
  tokenRefs: TokenRefRegistryValue,
  owner: string,
  ownerVersion: number,
): Map<string, { d: string; visible: boolean }> {
  const [paths, setPaths] = useState<Map<string, { d: string; visible: boolean }>>(
    () => new Map(),
  )

  const connectionsRef = useRef(connections)
  connectionsRef.current = connections

  const rafIdRef = useRef(0)
  const scheduledRef = useRef(false)

  const updatePaths = useCallback(() => {
    scheduledRef.current = false
    const cRect = containerRef.current?.getBoundingClientRect()
    if (!cRect) return

    const newPaths = new Map<string, { d: string; visible: boolean }>()
    for (const conn of connectionsRef.current) {
      const srcRect = tokenRefs.getRect(conn.sourceId, owner)
      const tgtRect = tokenRefs.getRect(conn.targetId, owner)

      if (!srcRect || !tgtRect) {
        newPaths.set(conn.id, { d: '', visible: false })
        continue
      }

      const srcVisible = isRectVisible(srcRect, cRect)
      const tgtVisible = isRectVisible(tgtRect, cRect)

      if (!srcVisible && !tgtVisible) {
        newPaths.set(conn.id, { d: '', visible: false })
        continue
      }

      const src = viewportToSvg(
        srcRect.right,
        srcRect.top + srcRect.height / 2,
        cRect,
      )
      const tgt = viewportToSvg(
        tgtRect.left,
        tgtRect.top + tgtRect.height / 2,
        cRect,
      )
      const d = computeBezierPath(src, tgt)

      newPaths.set(conn.id, { d, visible: srcVisible && tgtVisible })
    }

    setPaths(newPaths)
  }, [containerRef, tokenRefs, owner])

  const scheduleUpdate = useCallback(() => {
    if (!scheduledRef.current) {
      scheduledRef.current = true
      rafIdRef.current = requestAnimationFrame(updatePaths)
    }
  }, [updatePaths])

  // Update whenever the connections change, and whenever the words they are
  // measured against are replaced -- a new panel's elements land with the same
  // token ids and no other input to this hook moves.
  useEffect(() => {
    updatePaths()
  }, [connections, updatePaths, ownerVersion])

  // Attach scroll listeners and ResizeObserver
  useEffect(() => {
    const leftEl = leftPanelRef.current
    const rightEl = rightPanelRef.current
    const containerEl = containerRef.current

    if (leftEl) {
      leftEl.addEventListener('scroll', scheduleUpdate, { passive: true })
    }
    if (rightEl) {
      rightEl.addEventListener('scroll', scheduleUpdate, { passive: true })
    }

    let resizeObserver: ResizeObserver | undefined
    if (containerEl) {
      resizeObserver = new ResizeObserver(scheduleUpdate)
      resizeObserver.observe(containerEl)
    }

    return () => {
      if (leftEl) {
        leftEl.removeEventListener('scroll', scheduleUpdate)
      }
      if (rightEl) {
        rightEl.removeEventListener('scroll', scheduleUpdate)
      }
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      cancelAnimationFrame(rafIdRef.current)
    }
  }, [containerRef, leftPanelRef, rightPanelRef, scheduleUpdate])

  return paths
}
