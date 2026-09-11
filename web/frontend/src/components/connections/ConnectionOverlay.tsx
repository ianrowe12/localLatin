import { AnimatePresence } from 'framer-motion'
import { useConnectionState } from './useConnectionState'
import { useTokenRefs } from './TokenRefRegistry'
import { useLineUpdater } from './useLineUpdater'
import ConnectionLine from './ConnectionLine'

interface ConnectionOverlayProps {
  containerRef: React.RefObject<HTMLDivElement>
  leftPanelRef: React.RefObject<HTMLDivElement>
  rightPanelRef: React.RefObject<HTMLDivElement>
  /**
   * The pair these lines are about (issue #163).
   *
   * `AnimatePresence` deliberately outlives its removed children so they can
   * animate away, which is right when a reviewer stops hovering and wrong when
   * the document underneath them changes: a line from query token 0 to
   * candidate token 0 keeps its geometry while candidate token 0 becomes a
   * word from another manuscript, so for a third of a second the overlay draws
   * a claim nothing ever made. Emptying the token store cannot reach those
   * paths, because an exiting child is no longer in the list.
   *
   * Changing this value discards the presence owner itself, which removes them
   * in the same commit. It does not touch the connection state, the token
   * refs, the cached artifact or any preference.
   */
  evidenceKey?: string
}

export default function ConnectionOverlay({
  containerRef,
  leftPanelRef,
  rightPanelRef,
  evidenceKey,
}: ConnectionOverlayProps) {
  const { activeConnections } = useConnectionState()
  const tokenRefs = useTokenRefs()

  const paths = useLineUpdater(
    activeConnections,
    containerRef,
    leftPanelRef,
    rightPanelRef,
    tokenRefs,
  )

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 10 }}
    >
      <AnimatePresence key={evidenceKey}>
        {activeConnections.map((conn) => {
          const path = paths.get(conn.id)
          if (!path || !path.visible) return null
          return (
            <ConnectionLine
              key={conn.id}
              d={path.d}
              color={conn.color}
              score={conn.score}
              rank={conn.rank}
              isPinned={conn.isPinned}
              isAutoHighlighted={conn.isAutoHighlighted}
            />
          )
        })}
      </AnimatePresence>
    </svg>
  )
}
