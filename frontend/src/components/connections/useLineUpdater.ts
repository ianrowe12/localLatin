import { useState, useRef, useCallback, useEffect } from 'react'
import type { Connection } from './useConnectionState'
import type { TokenRefRegistryValue } from './TokenRefRegistry'
import { computeBezierPath, viewportToSvg, isRectVisible } from './bezierUtils'

export function useLineUpdater(
  connections: Connection[],
  containerRef: React.RefObject<HTMLDivElement>,
  leftPanelRef: React.RefObject<HTMLDivElement>,
  rightPanelRef: React.RefObject<HTMLDivElement>,
  tokenRefs: TokenRefRegistryValue,
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
      const srcRect = tokenRefs.getRect(conn.sourceId)
      const tgtRect = tokenRefs.getRect(conn.targetId)

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
  }, [containerRef, tokenRefs])

  const scheduleUpdate = useCallback(() => {
    if (!scheduledRef.current) {
      scheduledRef.current = true
      rafIdRef.current = requestAnimationFrame(updatePaths)
    }
  }, [updatePaths])

  // Update whenever connections change
  useEffect(() => {
    updatePaths()
  }, [connections, updatePaths])

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
