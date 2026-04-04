import { motion } from 'framer-motion'

interface ReviewerTableProps {
  reviewsByReviewer: Record<string, number>
}

export default function ReviewerTable({
  reviewsByReviewer,
}: ReviewerTableProps) {
  const entries = Object.entries(reviewsByReviewer).sort(
    ([, a], [, b]) => b - a,
  )

  if (entries.length === 0) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[200px]">
        <p className="font-ui text-sm text-stone-400 dark:text-stone-500">
          No reviewer data yet
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.45, ease: 'easeOut' }}
      className="glass-panel overflow-hidden"
    >
      <table className="w-full font-ui text-sm">
        <thead>
          <tr className="border-b border-stone-200/60 dark:border-stone-700/60">
            <th className="text-left px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-stone-400">
              Reviewer
            </th>
            <th className="text-right px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-stone-400">
              Reviews
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200/60 dark:divide-stone-700/60">
          {entries.map(([name, count]) => (
            <tr
              key={name}
              className="hover:bg-stone-100/60 dark:hover:bg-stone-700/40 transition-colors"
            >
              <td className="px-4 py-2.5 text-stone-700 dark:text-stone-200">
                {name}
              </td>
              <td className="px-4 py-2.5 text-right text-stone-500 dark:text-stone-400 tabular-nums">
                {count}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </motion.div>
  )
}

export function ReviewerTableSkeleton() {
  return (
    <div className="glass-panel overflow-hidden animate-pulse">
      <div className="px-4 py-2.5 border-b border-stone-200/60 dark:border-stone-700/60 flex justify-between">
        <div className="h-3 w-16 rounded bg-stone-200 dark:bg-stone-700" />
        <div className="h-3 w-12 rounded bg-stone-200 dark:bg-stone-700" />
      </div>
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="px-4 py-2.5 flex justify-between border-b border-stone-200/60 dark:border-stone-700/60 last:border-b-0"
        >
          <div className="h-4 w-24 rounded bg-stone-200 dark:bg-stone-700" />
          <div className="h-4 w-8 rounded bg-stone-200 dark:bg-stone-700" />
        </div>
      ))}
    </div>
  )
}
