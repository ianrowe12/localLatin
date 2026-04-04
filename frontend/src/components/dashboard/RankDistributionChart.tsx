import { motion } from 'framer-motion'

interface RankDistributionChartProps {
  rankDistribution: Record<string, number>
}

/** Maps rank number to an opacity class to produce a gradient-like effect */
function opacityForRank(rank: number, totalRanks: number): string {
  if (totalRanks <= 1) return 'bg-accent'
  const ratio = (rank - 1) / (totalRanks - 1)
  if (ratio < 0.2) return 'bg-accent'
  if (ratio < 0.4) return 'bg-accent/80'
  if (ratio < 0.6) return 'bg-accent/60'
  if (ratio < 0.8) return 'bg-accent/40'
  return 'bg-accent/25'
}

export default function RankDistributionChart({
  rankDistribution,
}: RankDistributionChartProps) {
  const entries = Object.entries(rankDistribution)
    .map(([rank, count]) => ({ rank: Number(rank), count }))
    .sort((a, b) => a.rank - b.rank)

  const maxCount = entries.reduce((max, e) => Math.max(max, e.count), 1)
  const totalRanks = entries.length

  if (entries.length === 0) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[200px]">
        <p className="font-ui text-sm text-stone-400 dark:text-stone-500">
          No rank data yet
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.4, ease: 'easeOut' }}
      className="glass-panel p-5 space-y-2"
    >
      {entries.map(({ rank, count }) => (
        <div key={rank} className="flex items-center gap-3">
          <span className="font-ui text-xs text-stone-500 dark:text-stone-400 w-8 text-right tabular-nums">
            #{rank}
          </span>
          <div className="flex-1 h-5 rounded bg-stone-200/50 dark:bg-stone-700/50 overflow-hidden">
            <motion.div
              className={`h-full rounded ${opacityForRank(rank, totalRanks)}`}
              initial={{ width: 0 }}
              animate={{ width: `${(count / maxCount) * 100}%` }}
              transition={{ duration: 0.6, delay: 0.45, ease: 'easeOut' }}
            />
          </div>
          <span className="font-ui text-xs text-stone-500 dark:text-stone-400 w-8 tabular-nums">
            {count}
          </span>
        </div>
      ))}
    </motion.div>
  )
}

export function RankDistributionSkeleton() {
  return (
    <div className="glass-panel p-5 space-y-2 animate-pulse">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="w-8 h-4 rounded bg-stone-200 dark:bg-stone-700" />
          <div className="flex-1 h-5 rounded bg-stone-200 dark:bg-stone-700" />
          <div className="w-8 h-4 rounded bg-stone-200 dark:bg-stone-700" />
        </div>
      ))}
    </div>
  )
}
