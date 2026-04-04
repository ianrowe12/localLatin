import { motion } from 'framer-motion'
import { useTokens, type ViewMode } from '../../contexts/TokenContext'

const MODES: { key: ViewMode; label: string; requiresIg?: boolean }[] = [
  { key: 'connections', label: 'Lines' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'ig', label: 'IG', requiresIg: true },
]

export default function ViewModeToggle() {
  const { viewMode, setViewMode, hasIgData } = useTokens()

  const visibleModes = MODES.filter((m) => !m.requiresIg || hasIgData)

  return (
    <div
      className="flex bg-stone-100 dark:bg-stone-800 rounded-lg p-0.5"
      role="radiogroup"
      aria-label="View mode"
    >
      {visibleModes.map((mode) => (
        <button
          key={mode.key}
          type="button"
          role="radio"
          aria-checked={viewMode === mode.key}
          onClick={() => setViewMode(mode.key)}
          className={`relative px-3 py-1 text-xs font-medium rounded-md transition-all ${
            viewMode === mode.key
              ? 'text-stone-800 dark:text-stone-100'
              : 'text-stone-500 hover:text-stone-700 dark:hover:text-stone-300'
          }`}
        >
          {viewMode === mode.key && (
            <motion.div
              layoutId="viewmode-indicator"
              className="absolute inset-0 bg-white dark:bg-stone-700 rounded-md shadow-sm"
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            />
          )}
          <span className="relative z-10">{mode.label}</span>
        </button>
      ))}
    </div>
  )
}
