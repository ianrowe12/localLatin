import { useCallback, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useQueryList } from '../../api/queries'
import SearchBar from '../sidebar/SearchBar'
import FilterChips from '../sidebar/FilterChips'
import QueryCard from '../sidebar/QueryCard'

const FILTER_TO_STATUS: Record<string, string> = {
  all: 'all',
  pending: 'unreviewed',
  reviewed: 'reviewed',
  skipped: 'skipped',
}

const PAGE_SIZE = 200

export default function FileBrowser() {
  const { activeQueryId, navigateToQuery } = useApp()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(1)

  const apiStatus = FILTER_TO_STATUS[filter] ?? 'all'
  const { data, loading, error } = useQueryList(apiStatus, search, page, PAGE_SIZE)

  const handleSearch = useCallback((value: string) => {
    setSearch(value)
    setPage(1)
  }, [])

  const handleFilter = useCallback((value: string) => {
    setFilter(value)
    setPage(1)
  }, [])

  const visibleCount = data?.items.length ?? 0
  const total = data?.total ?? 0

  return (
    <div className="flex-1 overflow-y-auto bg-surface-50 dark:bg-surface-900">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-stone-800 dark:text-stone-100 font-latin">
              Manuscripts
            </h1>
            <p className="text-sm text-stone-500 dark:text-stone-400 mt-1 font-ui">
              {loading && !data
                ? 'Loading files...'
                : `Showing ${visibleCount} of ${total} files`}
            </p>
          </div>
          <div className="w-full lg:w-[28rem]">
            <SearchBar value={search} onChange={handleSearch} />
          </div>
        </div>

        <div className="mb-5 flex items-center justify-between gap-3">
          <FilterChips active={filter} onChange={handleFilter} />
          {loading && data && (
            <span className="text-xs text-stone-400 dark:text-stone-500 font-ui">
              Loading...
            </span>
          )}
        </div>

        {error && (
          <p className="text-sm text-rose-600 dark:text-rose-400 font-ui">
            Failed to load files: {error}
          </p>
        )}

        {data && data.items.length === 0 && (
          <div className="flex items-center justify-center min-h-[200px]">
            <p className="text-sm text-stone-400 dark:text-stone-500 font-ui">
              No files found
            </p>
          </div>
        )}

        {data && data.items.length > 0 && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {data.items.map((item) => (
                <div
                  key={item.file_id}
                  className="rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800"
                >
                  <QueryCard
                    item={item}
                    isActive={item.file_id === activeQueryId}
                    onClick={() => navigateToQuery(item.file_id)}
                  />
                </div>
              ))}
            </div>

            <div className="flex justify-center py-6">
              {data.has_more ? (
                <button
                  type="button"
                  onClick={() => setPage((current) => current + 1)}
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
