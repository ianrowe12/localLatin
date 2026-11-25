export type KeyModelScope = 'this' | 'all'

const SCOPES: { key: KeyModelScope; label: string }[] = [
  { key: 'this', label: 'This model' },
  { key: 'all', label: 'All models' },
]

interface KeyModelScopeToggleProps {
  scope: KeyModelScope
  onChange: (scope: KeyModelScope) => void
}

export default function KeyModelScopeToggle({ scope, onChange }: KeyModelScopeToggleProps) {
  return (
    <div
      className="flex bg-stone-100 dark:bg-stone-800 rounded-lg p-0.5"
      role="radiogroup"
      aria-label="Suggestions from"
    >
      {SCOPES.map(({ key, label }) => {
        const selected = scope === key
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
