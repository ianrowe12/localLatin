import { AnimatePresence } from 'framer-motion'
import { useConnectionState } from './useConnectionState'
import { useTokenRefs } from './TokenRefRegistry'
import { useLineUpdater } from './useLineUpdater'
import ConnectionLine from './ConnectionLine'

interface ConnectionOverlayProps {
  containerRef: React.RefObject<HTMLDivElement>
  leftPanelRef: React.RefObject<HTMLDivElement>
  rightPanelRef: React.RefObject<HTMLDivElement>
}

export default function ConnectionOverlay({
  containerRef,
  leftPanelRef,
  rightPanelRef,
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
      <AnimatePresence>
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
            />
          )
        })}
      </AnimatePresence>
    </svg>
  )
}
