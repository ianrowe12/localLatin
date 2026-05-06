import { motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import type { NeedsAttentionItem } from '../../api/models'

interface NeedsAttentionListProps {
  items: NeedsAttentionItem[]
}

function outcomeLabel(outcome: string): string {
  if (outcome === 'none_of_top_k') return 'None of top K'
  if (outcome === 'skipped') return 'Skipped'
  return outcome.replace(/_/g, ' ')
}

export default function NeedsAttentionList({ items }: NeedsAttentionListProps) {
  const { navigateToQuery } = useApp()

  if (items.length === 0) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[160px]">
        <p className="font-ui text-sm text-stone-400 dark:text-stone-500">
          No deferred or none-of-top-K reviews
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.25, ease: 'easeOut' }}
      className="glass-panel overflow-hidden"
    >
      <ul className="divide-y divide-stone-200/60 dark:divide-stone-700/60">
        {items.map((item, idx) => (
          <li key={`${item.file_id}-${item.outcome}-${idx}`}>
            <button
              type="button"
              onClick={() => navigateToQuery(item.file_id)}
              className="w-full px-4 py-3 flex items-start gap-3 text-left
                         hover:bg-stone-100/60 dark:hover:bg-stone-700/40 transition-colors"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <p className="font-latin text-sm text-stone-800 dark:text-stone-100 truncate">
                    {item.filename}
                  </p>
                  <span className="flex-shrink-0 px-2 py-0.5 rounded-full text-xs font-ui font-medium bg-stone-200 text-stone-700 dark:bg-stone-700 dark:text-stone-200">
                    {outcomeLabel(item.outcome)}
                  </span>
                </div>
                <p className="font-ui text-xs text-stone-500 dark:text-stone-400 truncate mt-1">
                  {item.notes || 'No note recorded'}
                </p>
              </div>
              <span className="flex-shrink-0 font-ui text-xs text-stone-400 dark:text-stone-500">
                {item.reviewer}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </motion.div>
  )
}

export function NeedsAttentionSkeleton() {
  return (
    <div className="glass-panel overflow-hidden animate-pulse">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="px-4 py-3 flex items-center gap-3 border-b border-stone-200/60 dark:border-stone-700/60 last:border-b-0"
        >
          <div className="flex-1 space-y-2">
            <div className="h-4 rounded bg-stone-200 dark:bg-stone-700" />
            <div className="h-3 w-3/4 rounded bg-stone-200 dark:bg-stone-700" />
          </div>
          <div className="w-16 h-4 rounded bg-stone-200 dark:bg-stone-700" />
        </div>
      ))}
    </div>
  )
}
