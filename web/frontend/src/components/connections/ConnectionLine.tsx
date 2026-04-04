import { motion } from 'framer-motion'

interface ConnectionLineProps {
  d: string
  color: string
  score: number
  rank: number
  isPinned: boolean
}

export default function ConnectionLine({
  d,
  color,
  score,
  rank,
  isPinned,
}: ConnectionLineProps) {
  const strokeWidth = isPinned ? 2.5 : [2.5, 2, 1.5][rank] ?? 1.5
  const opacity = isPinned ? 0.85 : 0.3 + score * 0.55

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
        rank === 0 ? undefined : rank === 1 ? '6 3' : '3 3'
      }
    />
  )
}
