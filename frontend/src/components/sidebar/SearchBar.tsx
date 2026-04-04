import { useEffect, useRef, useState } from 'react'

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
}

export default function SearchBar({ value, onChange }: SearchBarProps) {
  const [internal, setInternal] = useState(value)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sync external value changes (e.g. programmatic resets)
  useEffect(() => {
    setInternal(value)
  }, [value])

  useEffect(() => {
    timerRef.current = setTimeout(() => {
      onChange(internal)
    }, 300)
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [internal, onChange])

  return (
    <div data-tour="search-bar" className="relative">
      {/* Search icon */}
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400 pointer-events-none"
        aria-hidden="true"
      >
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>

      <input
        type="text"
        value={internal}
        onChange={(e) => setInternal(e.target.value)}
        placeholder="Search files..."
        className="w-full rounded-lg border border-stone-200 dark:border-stone-600
                   bg-white dark:bg-stone-800
                   px-3 py-2 pl-8 pr-8
                   text-sm font-ui text-stone-700 dark:text-stone-200
                   placeholder:text-stone-400 dark:placeholder:text-stone-500
                   focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent
                   transition-colors"
        aria-label="Search files"
      />

      {/* Clear button */}
      {internal && (
        <button
          type="button"
          onClick={() => setInternal('')}
          className="absolute right-2 top-1/2 -translate-y-1/2
                     w-5 h-5 rounded-full flex items-center justify-center
                     text-stone-400 hover:text-stone-600 dark:hover:text-stone-300
                     hover:bg-stone-200 dark:hover:bg-stone-600 transition-colors"
          aria-label="Clear search"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      )}
    </div>
  )
}
