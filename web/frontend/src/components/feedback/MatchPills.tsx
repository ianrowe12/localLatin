interface MatchPillsProps {
  selectedRank: number | null
  maxRank?: number
  onChange: (rank: number | null) => void
}

export default function MatchPills({ selectedRank, maxRank = 5, onChange }: MatchPillsProps) {
  const handleClick = (rank: number) => {
    // Toggle: if already selected, deselect; otherwise select
    if (selectedRank === rank) {
      onChange(null)
    } else {
      onChange(rank)
    }
  }

  const matchPills = Array.from({ length: maxRank }, (_, i) => i + 1)

  return (
    <div className="grid grid-cols-3 gap-1.5">
      {matchPills.map((rank) => {
        const isSelected = selectedRank === rank
        return (
          <button
            key={rank}
            type="button"
            onClick={() => handleClick(rank)}
            className={`text-xs py-1.5 px-2 rounded-full text-center cursor-pointer transition-all ${
              isSelected
                ? 'bg-correct/15 text-correct ring-1 ring-correct/40 font-medium'
                : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600'
            }`}
            aria-pressed={isSelected}
          >
            Match #{rank}
          </button>
        )
      })}

      {/* None correct pill */}
      <button
        type="button"
        onClick={() => handleClick(0)}
        className={`col-span-3 text-xs py-1.5 px-2 rounded-full text-center cursor-pointer transition-all ${
          selectedRank === 0
            ? 'bg-incorrect/15 text-incorrect ring-1 ring-incorrect/40 font-medium'
            : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600'
        }`}
        aria-pressed={selectedRank === 0}
      >
        None correct
      </button>
    </div>
  )
}
