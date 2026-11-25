export type SearchMode = 'file' | 'key'

const MODES: { key: SearchMode; label: string }[] = [
  { key: 'file', label: 'File name' },
  { key: 'key', label: 'Suggested key' },
]

interface SearchModeToggleProps {
  mode: SearchMode
  onChange: (mode: SearchMode) => void
}

export default function SearchModeToggle({ mode, onChange }: SearchModeToggleProps) {
  return (
    <div
      className="flex bg-stone-100 dark:bg-stone-800 rounded-lg p-0.5"
      role="radiogroup"
      aria-label="Search by"
    >
      {MODES.map(({ key, label }) => {
        const selected = mode === key
        return (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(key)}
            className={`flex-1 px-3 py-1 text-xs font-medium font-ui rounded-md transition-colors ${
              selected
                ? 'bg-white dark:bg-stone-700 shadow-sm text-stone-800 dark:text-stone-100'
                : 'text-stone-500 hover:text-stone-700 dark:hover:text-stone-300'
            }`}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}
