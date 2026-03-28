import { memo, useCallback, useRef, useState } from 'react'

interface DraggableDividerProps {
  onDrag: (newPercent: number) => void
}

function DraggableDividerInner({ onDrag }: DraggableDividerProps) {
  const [dragging, setDragging] = useState(false)
  const dividerRef = useRef<HTMLDivElement>(null)

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      const target = e.currentTarget
      target.setPointerCapture(e.pointerId)
      setDragging(true)
    },
    [],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return
      const parent = dividerRef.current?.parentElement
      if (!parent) return

      const rect = parent.getBoundingClientRect()
      const x = e.clientX - rect.left
      const percent = (x / rect.width) * 100
      const clamped = Math.min(80, Math.max(20, percent))
      onDrag(clamped)
    },
    [dragging, onDrag],
  )

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.currentTarget.releasePointerCapture(e.pointerId)
      setDragging(false)
    },
    [],
  )

  return (
    <div
      ref={dividerRef}
      className="relative flex-shrink-0 w-1 cursor-col-resize select-none group"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize panels"
      tabIndex={0}
    >
      {/* Wider invisible hit area */}
      <div className="absolute inset-y-0 -left-1 -right-1" />
      {/* Visible line */}
      <div
        className={`
          absolute inset-y-0 left-[1px] w-[2px] transition-colors
          ${
            dragging
              ? 'bg-accent'
              : 'bg-stone-300 dark:bg-stone-600 group-hover:bg-accent-light dark:group-hover:bg-accent-light'
          }
        `}
      />
    </div>
  )
}

const DraggableDivider = memo(DraggableDividerInner)
DraggableDivider.displayName = 'DraggableDivider'

export default DraggableDivider
