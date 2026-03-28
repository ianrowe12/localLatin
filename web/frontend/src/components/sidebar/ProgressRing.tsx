interface ProgressRingProps {
  reviewed: number
  total: number
}

const RADIUS = 24
const STROKE = 4
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

export default function ProgressRing({ reviewed, total }: ProgressRingProps) {
  const pct = total > 0 ? Math.round((reviewed / total) * 100) : 0
  const offset = CIRCUMFERENCE - (CIRCUMFERENCE * pct) / 100

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={64} height={64} viewBox="0 0 64 64" aria-hidden="true">
        {/* Track */}
        <circle
          cx={32}
          cy={32}
          r={RADIUS}
          fill="none"
          stroke="currentColor"
          strokeWidth={STROKE}
          className="text-stone-200 dark:text-stone-600"
        />
        {/* Filled arc */}
        <circle
          cx={32}
          cy={32}
          r={RADIUS}
          fill="none"
          stroke="currentColor"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          className="text-accent transition-[stroke-dashoffset] duration-500 ease-out"
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
        />
        {/* Center text */}
        <text
          x={32}
          y={32}
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-stone-700 dark:fill-stone-200 font-ui text-sm font-semibold"
          style={{ fontSize: '14px' }}
        >
          {pct}%
        </text>
      </svg>
      <span className="text-xs text-stone-500 font-ui">
        {reviewed} / {total}
      </span>
    </div>
  )
}
