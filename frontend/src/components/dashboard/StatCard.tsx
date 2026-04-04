import { motion } from 'framer-motion'
import ProgressRing from '../sidebar/ProgressRing'

interface StatCardProps {
  label: string
  value: number | string
  icon: React.ReactNode
  color?: string
  ring?: { reviewed: number; total: number }
}

export default function StatCard({
  label,
  value,
  icon,
  color = 'text-accent',
  ring,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="glass-panel p-5 flex items-center gap-4"
    >
      {ring ? (
        <ProgressRing reviewed={ring.reviewed} total={ring.total} />
      ) : (
        <div
          className={`w-10 h-10 rounded-lg flex items-center justify-center ${color} bg-current/10`}
        >
          <span className={color}>{icon}</span>
        </div>
      )}
      <div className="min-w-0">
        <p className="font-ui text-2xl font-bold text-stone-800 dark:text-stone-100 tabular-nums">
          {value}
        </p>
        <p className="font-ui text-xs text-stone-500 dark:text-stone-400 uppercase tracking-wider">
          {label}
        </p>
      </div>
    </motion.div>
  )
}

/** Skeleton placeholder while stats are loading */
export function StatCardSkeleton() {
  return (
    <div className="glass-panel p-5 flex items-center gap-4 animate-pulse">
      <div className="w-10 h-10 rounded-lg bg-stone-200 dark:bg-stone-700" />
      <div className="space-y-2 flex-1">
        <div className="h-6 w-16 rounded bg-stone-200 dark:bg-stone-700" />
        <div className="h-3 w-24 rounded bg-stone-200 dark:bg-stone-700" />
      </div>
    </div>
  )
}
