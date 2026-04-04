import { motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import type { RecentReview } from '../../api/models'

interface RecentActivityListProps {
  reviews: RecentReview[]
}

function formatRelativeTime(timestamp: string): string {
  const now = Date.now()
  const then = new Date(timestamp).getTime()
  const diffMs = now - then

  const seconds = Math.floor(diffMs / 1000)
  if (seconds < 60) return 'just now'

  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`

  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`

  const months = Math.floor(days / 30)
  return `${months}mo ago`
}

export default function RecentActivityList({
  reviews,
}: RecentActivityListProps) {
  const { navigateToQuery } = useApp()

  if (reviews.length === 0) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[200px]">
        <p className="font-ui text-sm text-stone-400 dark:text-stone-500">
          No reviews yet
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3, ease: 'easeOut' }}
      className="glass-panel overflow-hidden"
    >
      <ul className="divide-y divide-stone-200/60 dark:divide-stone-700/60">
        {reviews.map((review, idx) => (
          <li key={`${review.file_id}-${idx}`}>
            <button
              type="button"
              onClick={() => navigateToQuery(review.file_id)}
              className="w-full px-4 py-3 flex items-center gap-3 text-left
                         hover:bg-stone-100/60 dark:hover:bg-stone-700/40 transition-colors"
            >
              <div className="min-w-0 flex-1">
                <p className="font-latin text-sm text-stone-800 dark:text-stone-100 truncate">
                  {review.filename}
                </p>
              </div>
              <span className="flex-shrink-0 px-2 py-0.5 rounded-full text-xs font-ui font-medium bg-accent/10 text-accent dark:bg-accent-light/10 dark:text-accent-light">
                {review.model_slug}
              </span>
              <span className="flex-shrink-0 font-ui text-xs text-stone-400 dark:text-stone-500 tabular-nums">
                {formatRelativeTime(review.timestamp)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </motion.div>
  )
}

export function RecentActivitySkeleton() {
  return (
    <div className="glass-panel overflow-hidden animate-pulse">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="px-4 py-3 flex items-center gap-3 border-b border-stone-200/60 dark:border-stone-700/60 last:border-b-0"
        >
          <div className="flex-1 h-4 rounded bg-stone-200 dark:bg-stone-700" />
          <div className="w-16 h-5 rounded-full bg-stone-200 dark:bg-stone-700" />
          <div className="w-10 h-3 rounded bg-stone-200 dark:bg-stone-700" />
        </div>
      ))}
    </div>
  )
}
