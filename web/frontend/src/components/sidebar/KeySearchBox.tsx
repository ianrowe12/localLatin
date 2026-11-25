import { useState, type FocusEvent, type KeyboardEvent } from 'react'
import { useKeySuggestions } from '../../api/keys'
import SearchBar from './SearchBar'

interface KeySearchBoxProps {
  value: string
  /** True once a suggestion was picked: the key is matched whole, not as a prefix. */
  exact: boolean
  onSearch: (text: string) => void
  onPick: (key: string) => void
}

export default function KeySearchBox({ value, exact, onSearch, onPick }: KeySearchBoxProps) {
  const [open, setOpen] = useState(false)
  const query = value.trim()
  const suggestions = useKeySuggestions(query, open && !exact)

  const handleBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') setOpen(false)
  }

  return (
    <div
      className="relative"
      onFocus={() => setOpen(true)}
      onInput={() => setOpen(true)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
    >
      <SearchBar
        value={value}
        onChange={onSearch}
        placeholder="Search suggested keys, e.g. CNIC.325 or D"
        label="Search suggested keys"
      />
      {suggestions.length > 0 && (
        <ul
          role="listbox"
          aria-label="Matching keys"
          className="absolute left-0 right-0 top-full z-20 mt-1 max-h-72 overflow-y-auto
                     rounded-lg border border-stone-200 dark:border-stone-600
                     bg-white dark:bg-surface-800 py-1 shadow-glass dark:shadow-glass-dark"
        >
          {suggestions.map(({ key, unit_count }) => (
            <li key={key} role="option" aria-selected={false}>
              <button
                type="button"
                onClick={() => {
                  onPick(key)
                  setOpen(false)
                }}
                className="flex w-full items-baseline justify-between gap-2 px-3 py-1.5 text-left
                           font-ui text-sm text-stone-700 dark:text-stone-200
                           hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors"
              >
                <span className="min-w-0 truncate whitespace-pre">{key}</span>
                <span className="flex-shrink-0 text-xs text-stone-400 dark:text-stone-500">
                  {unit_count === 1 ? '1 unit' : `${unit_count} units`}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
