import { motion } from 'framer-motion'

interface ModelProgressBarsProps {
  reviewsByModel: Record<string, number>
}

export default function ModelProgressBars({
  reviewsByModel,
}: ModelProgressBarsProps) {
  const entries = Object.entries(reviewsByModel).sort(([, a], [, b]) => b - a)
  const maxCount = entries.length > 0 ? entries[0][1] : 1

  if (entries.length === 0) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[200px]">
        <p className="font-ui text-sm text-stone-400 dark:text-stone-500">
          No model data yet
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.35, ease: 'easeOut' }}
      className="glass-panel p-5 space-y-3"
    >
      {entries.map(([model, count]) => (
        <div key={model}>
          <div className="flex items-center justify-between mb-1">
            <span className="font-ui text-sm text-stone-700 dark:text-stone-200 truncate">
              {model}
            </span>
            <span className="font-ui text-xs text-stone-400 dark:text-stone-500 tabular-nums ml-2">
              {count}
            </span>
          </div>
          <div className="w-full h-2 rounded-full bg-stone-200 dark:bg-stone-700 overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-accent"
              initial={{ width: 0 }}
              animate={{ width: `${(count / maxCount) * 100}%` }}
              transition={{ duration: 0.6, delay: 0.4, ease: 'easeOut' }}
            />
          </div>
        </div>
      ))}
    </motion.div>
  )
}

export function ModelProgressSkeleton() {
  return (
    <div className="glass-panel p-5 space-y-3 animate-pulse">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i}>
          <div className="flex items-center justify-between mb-1">
            <div className="h-4 w-28 rounded bg-stone-200 dark:bg-stone-700" />
            <div className="h-3 w-8 rounded bg-stone-200 dark:bg-stone-700" />
          </div>
          <div className="w-full h-2 rounded-full bg-stone-200 dark:bg-stone-700" />
        </div>
      ))}
    </div>
  )
}
