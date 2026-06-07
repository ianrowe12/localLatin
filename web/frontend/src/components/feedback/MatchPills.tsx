export interface MatchSelection {
  selectedRanks: number[]
  noneSelected: boolean
}

interface MatchPillsProps {
  multiSelect: boolean
  selectedRanks: number[]
  noneSelected: boolean
  maxRank?: number
  onChange: (next: MatchSelection) => void
}

export default function MatchPills({
  multiSelect,
  selectedRanks,
  noneSelected,
  maxRank = 10,
  onChange,
}: MatchPillsProps) {
  const handleRankClick = (rank: number) => {
    const isSelected = selectedRanks.includes(rank)
    if (multiSelect) {
      // Toggle the rank in/out; selecting any rank clears "None".
      const nextRanks = isSelected
        ? selectedRanks.filter((value) => value !== rank)
        : [...selectedRanks, rank].sort((a, b) => a - b)
      onChange({ selectedRanks: nextRanks, noneSelected: false })
    } else {
      // Single select: clicking the active pill clears it; else select just it.
      onChange({
        selectedRanks: isSelected ? [] : [rank],
        noneSelected: false,
      })
    }
  }

  const handleNoneClick = () => {
    // Mutually exclusive with rank selections.
    onChange({ selectedRanks: [], noneSelected: !noneSelected })
  }

  const matchPills = Array.from({ length: maxRank }, (_, i) => i + 1)

  return (
    <div data-tour="match-options" className="grid grid-cols-5 gap-1.5">
      {matchPills.map((rank) => {
        const isSelected = selectedRanks.includes(rank)
        return (
          <button
            key={rank}
            type="button"
            onClick={() => handleRankClick(rank)}
            className={`text-xs py-1.5 px-2 rounded-full text-center cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-accent/40 ${
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
        onClick={handleNoneClick}
        className={`col-span-5 text-xs py-1.5 px-2 rounded-full text-center cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-accent/40 ${
          noneSelected
            ? 'bg-incorrect/15 text-incorrect ring-1 ring-incorrect/40 font-medium'
            : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600'
        }`}
        aria-pressed={noneSelected}
      >
        None of top {maxRank}
      </button>
    </div>
  )
}
