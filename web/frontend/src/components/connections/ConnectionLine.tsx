import { motion } from 'framer-motion'

interface ConnectionLineProps {
  d: string
  color: string
  score: number
  rank: number
  isPinned: boolean
  isAutoHighlighted?: boolean
}

export default function ConnectionLine({
  d,
  color,
  score,
  rank,
  isPinned,
  isAutoHighlighted,
}: ConnectionLineProps) {
  const strokeWidth = isAutoHighlighted ? 2 : isPinned ? 2.5 : [2.5, 2, 1.5][rank] ?? 1.5
  const opacity = isAutoHighlighted ? 0.55 : isPinned ? 0.85 : 0.3 + score * 0.55

  return (
    <motion.path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity }}
      exit={{ pathLength: 0, opacity: 0 }}
      transition={{
        pathLength: { duration: 0.35, ease: [0.65, 0, 0.35, 1] },
        opacity: { duration: 0.2 },
      }}
      strokeDasharray={
        isAutoHighlighted ? '6 4' : rank === 0 ? undefined : rank === 1 ? '6 3' : '3 3'
      }
    />
  )
}
