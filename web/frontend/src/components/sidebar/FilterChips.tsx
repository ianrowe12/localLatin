interface FilterChipsProps {
  active: string
  onChange: (filter: string) => void
}

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'reviewed', label: 'Reviewed' },
  { key: 'skipped', label: 'Skipped' },
] as const

export default function FilterChips({ active, onChange }: FilterChipsProps) {
  return (
    <div data-tour="filter-chips" className="flex gap-1.5" role="group" aria-label="Filter queries">
      {FILTERS.map(({ key, label }) => {
        const isActive = active === key
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            className={`px-2.5 py-1 rounded-full text-xs font-medium font-ui transition-colors
              ${
                isActive
                  ? 'bg-accent text-white'
                  : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600'
              }`}
            aria-pressed={isActive}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}
