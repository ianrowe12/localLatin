import { useCallback, useState } from 'react'
import { useLabelledKeys, type LabelledKeyItem } from '../../api/keys'
import { useCandidateDirFiles } from '../../api/queries'
import SearchBar from '../sidebar/SearchBar'

function unitLabel(count: number): string {
  return count === 1 ? '1 unit' : `${count} units`
}

function KeyUnits({ keyName }: { keyName: string }) {
  const { data, loading, error } = useCandidateDirFiles(keyName)

  if (loading) {
    return <p className="px-4 py-3 text-xs text-stone-400 font-ui">Loading...</p>
  }
  if (error) {
    return (
      <p className="px-4 py-3 text-xs text-rose-600 dark:text-rose-400 font-ui">
        Failed to load units: {error}
      </p>
    )
  }
  if (!data || data.length === 0) {
    return <p className="px-4 py-3 text-xs text-stone-400 font-ui">No units found</p>
  }
  return (
    <ul className="divide-y divide-stone-100 dark:divide-stone-700/60">
      {data.map((file) => (
        <li key={file.filename} className="px-4 py-3">
          <div className="text-xs font-medium text-stone-600 dark:text-stone-300 font-ui">
            {file.filename}
          </div>
          <p className="mt-1 max-h-40 overflow-y-auto text-sm leading-relaxed text-stone-700 dark:text-stone-200 font-latin">
            {file.text}
          </p>
        </li>
      ))}
    </ul>
  )
}

function KeyRow({ item }: { item: LabelledKeyItem }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800">
      <button
        type="button"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left
                   hover:bg-stone-50 dark:hover:bg-stone-800 rounded-lg transition-colors"
      >
        <span className="min-w-0 truncate whitespace-pre text-sm font-medium text-stone-700 dark:text-stone-200 font-ui">
          {item.key}
        </span>
        <span className="flex flex-shrink-0 items-center gap-2 text-xs text-stone-400 dark:text-stone-500 font-ui">
          {unitLabel(item.unit_count)}
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className={`transition-transform ${expanded ? 'rotate-180' : ''}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </button>
      {expanded && (
        <div className="border-t border-stone-100 dark:border-stone-700/60">
          <KeyUnits keyName={item.key} />
        </div>
      )}
    </div>
  )
}

export default function LabelledKeysPanel() {
  const [search, setSearch] = useState('')
  const [pageState, setPageState] = useState({ search: '', page: 1 })
  const page = pageState.search === search ? pageState.page : 1
  const { data, loading, error } = useLabelledKeys(search, page)

  const handleSearch = useCallback((value: string) => setSearch(value), [])

  const total = data?.total ?? 0
  const trimmed = search.trim()

  return (
    <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
      <div className="max-w-3xl mx-auto px-6 py-8">
        <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-stone-800 dark:text-stone-100 font-latin">
              Labelled keys
            </h1>
            <p
              data-testid="labelled-key-count"
              className="text-sm text-stone-500 dark:text-stone-400 mt-1 font-ui"
            >
              {loading && !data
                ? 'Loading keys...'
                : total === 1
                  ? '1 key'
                  : `${total} keys`}
            </p>
          </div>
          <div className="w-full sm:w-72">
            <SearchBar
              value={search}
              onChange={handleSearch}
              placeholder="Search keys, e.g. CNIC.325 or D"
              label="Search labelled keys"
            />
          </div>
        </div>

        {error && (
          <p className="mb-4 text-sm text-rose-600 dark:text-rose-400 font-ui">
            Failed to load keys: {error}
          </p>
        )}

        {data && data.items.length === 0 && (
          <div className="flex items-center justify-center min-h-[200px]">
            <p className="text-sm text-stone-400 dark:text-stone-500 font-ui">
              No keys match {'\u201c'}
              {trimmed}
              {'\u201d'}
            </p>
          </div>
        )}

        {data && data.items.length > 0 && (
          <>
            <div className="space-y-2">
              {data.items.map((item) => (
                <KeyRow key={item.key} item={item} />
              ))}
            </div>

            <div className="flex justify-center py-6">
              {data.has_more ? (
                <button
                  type="button"
                  onClick={() => setPageState({ search, page: page + 1 })}
                  disabled={loading}
                  className="h-9 px-4 rounded-lg bg-stone-100 dark:bg-stone-700 text-sm font-ui font-medium text-stone-700 dark:text-stone-200 hover:bg-stone-200 dark:hover:bg-stone-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? 'Loading...' : 'Load more'}
                </button>
              ) : (
                <span className="text-xs text-stone-400 dark:text-stone-500 font-ui">
                  End of results
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
