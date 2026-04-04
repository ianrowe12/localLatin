import { motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import type { StatsResponse } from '../../api/models'

interface ResumeCardProps {
  stats: StatsResponse
}

export default function ResumeCard({ stats }: ResumeCardProps) {
  const { navigateToQuery } = useApp()

  const hasUnreviewed =
    stats.next_unreviewed_ids && stats.next_unreviewed_ids.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2, ease: 'easeOut' }}
      data-tour="resume-btn"
      className="glass-panel overflow-hidden"
    >
      <div className="bg-gradient-to-r from-accent to-accent-light p-6 flex items-center justify-between gap-4">
        <div className="min-w-0">
          {hasUnreviewed ? (
            <>
              <h2 className="font-ui text-lg font-bold text-white">
                Resume Reviewing
              </h2>
              <p className="font-ui text-sm text-white/80 mt-1">
                {stats.unreviewed_count} unreviewed{' '}
                {stats.unreviewed_count === 1 ? 'query' : 'queries'} remaining
              </p>
            </>
          ) : (
            <>
              <h2 className="font-ui text-lg font-bold text-white">
                All Caught Up!
              </h2>
              <p className="font-ui text-sm text-white/80 mt-1">
                Every query has been reviewed. Nice work.
              </p>
            </>
          )}
        </div>
        {hasUnreviewed && (
          <button
            type="button"
            onClick={() => navigateToQuery(stats.next_unreviewed_ids[0])}
            className="flex-shrink-0 px-5 py-2.5 rounded-lg font-ui text-sm font-semibold
                       bg-white text-accent hover:bg-white/90 transition-colors
                       shadow-md"
          >
            Continue Reviewing
          </button>
        )}
      </div>
    </motion.div>
  )
}

export function ResumeCardSkeleton() {
  return (
    <div className="glass-panel overflow-hidden animate-pulse">
      <div className="bg-gradient-to-r from-stone-300 to-stone-200 dark:from-stone-700 dark:to-stone-600 p-6 flex items-center justify-between gap-4">
        <div className="space-y-2 flex-1">
          <div className="h-5 w-40 rounded bg-white/30" />
          <div className="h-4 w-56 rounded bg-white/20" />
        </div>
        <div className="h-10 w-36 rounded-lg bg-white/30" />
      </div>
    </div>
  )
}
