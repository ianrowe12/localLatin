import { useState, useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Virtuoso } from 'react-virtuoso'
import { useApp } from '../../contexts/AppContext'
import { useReviewer } from '../../contexts/ReviewerContext'
import { useQueryList, type QueryKeyFilter } from '../../api/queries'
import { useStats } from '../../api/models'
import ProgressRing from '../sidebar/ProgressRing'
import SearchBar from '../sidebar/SearchBar'
import SearchModeToggle, { type SearchMode } from '../sidebar/SearchModeToggle'
import KeySearchBox from '../sidebar/KeySearchBox'
import FilterChips from '../sidebar/FilterChips'
import QueryCard from '../sidebar/QueryCard'
import ViewModeToggle from '../common/ViewModeToggle'
import AttributionMethodSelector from '../common/AttributionMethodSelector'

interface LeftSidebarProps {
  isOpen: boolean
  onToggle: () => void
}

const FILTER_TO_STATUS: Record<string, string> = {
  all: 'all',
  pending: 'unreviewed',
  reviewed: 'reviewed',
  skipped: 'skipped',
}

export default function LeftSidebar({ isOpen, onToggle }: LeftSidebarProps) {
  const { activeQueryId, setActiveQueryId, setCurrentView, activeModel, activeVariant } =
    useApp()
  const { isPiAdmin } = useReviewer()
  const [mode, setMode] = useState<SearchMode>('file')
  const [search, setSearch] = useState('')
  const [keyQuery, setKeyQuery] = useState({ text: '', exact: false })
  const [top1, setTop1] = useState(false)
  const [filter, setFilter] = useState('all')
  const [pageState, setPageState] = useState({ list: '', page: 1 })

  const keyFilter = useMemo<QueryKeyFilter | null>(() => {
    const key = keyQuery.text.trim()
    if (mode !== 'key' || !key) return null
    return {
      key,
      match: keyQuery.exact ? 'exact' : 'prefix',
      model: activeModel || undefined,
      variant: activeVariant,
      top1,
    }
  }, [mode, keyQuery, activeModel, activeVariant, top1])

  // The page belongs to the list it was reached in: a new status, search, key or
  // model starts again at page 1 instead of requesting page N of a different list.
  const listId = JSON.stringify([filter, mode === 'file' ? search : '', keyFilter])
  const page = pageState.list === listId ? pageState.page : 1
  const setPage = useCallback(
    (next: (current: number) => number) =>
      setPageState({ list: listId, page: next(page) }),
    [listId, page],
  )

  const apiStatus = FILTER_TO_STATUS[filter] ?? 'all'
  const { data, loading } = useQueryList(
    apiStatus,
    mode === 'file' ? search : '',
    page,
    undefined,
    keyFilter,
  )
  const { data: stats } = useStats(isPiAdmin)

  const handleSearch = useCallback((value: string) => {
    setSearch(value)
  }, [])

  const handleKeySearch = useCallback((text: string) => {
    setKeyQuery((prev) => (prev.text === text ? prev : { text, exact: false }))
  }, [])

  const handleKeyPick = useCallback((key: string) => {
    setKeyQuery({ text: key, exact: true })
  }, [])

  const handleMode = useCallback((next: SearchMode) => {
    setMode(next)
    setSearch('')
    setKeyQuery({ text: '', exact: false })
  }, [])

  const handleFilter = useCallback((f: string) => {
    setFilter(f)
  }, [])

  const handleEndReached = useCallback(() => {
    if (data?.has_more && !loading) {
      setPage((p) => p + 1)
    }
  }, [data?.has_more, loading, setPage])

  const keyMode = mode === 'key'

  return (
    <motion.aside
      data-tour="query-list"
      className="h-full flex-shrink-0 overflow-hidden border-r border-stone-200/60 dark:border-stone-700/60
                 bg-white/60 dark:bg-surface-800/60 backdrop-blur-sm"
      animate={{ width: isOpen ? 280 : 48 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      aria-label="Query sidebar"
    >
      <div className="flex flex-col h-full overflow-hidden">
        {/* Toggle button row */}
        <div className="flex items-center justify-end h-10 px-2 flex-shrink-0">
          <button
            onClick={onToggle}
            type="button"
            className="w-7 h-7 rounded flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-500 dark:text-stone-400"
            aria-label={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              className={`transition-transform duration-200 ${isOpen ? '' : 'rotate-180'}`}
            >
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        </div>

        {isOpen && (
          <>
            {isPiAdmin && (
              <div className="flex justify-center py-3 flex-shrink-0">
                <ProgressRing
                  reviewed={stats?.reviewed_count ?? 0}
                  total={stats?.total_queries ?? 0}
                />
              </div>
            )}

            {/* Search */}
            <div className="px-3 pt-2 flex-shrink-0">
              <SearchModeToggle mode={mode} onChange={handleMode} />
            </div>
            <div className="px-3 py-2 flex-shrink-0">
              {keyMode ? (
                <KeySearchBox
                  value={keyQuery.text}
                  exact={keyQuery.exact}
                  onSearch={handleKeySearch}
                  onPick={handleKeyPick}
                />
              ) : (
                <SearchBar value={search} onChange={handleSearch} />
              )}
            </div>
            {keyMode && (
              <div className="px-3 pb-1 flex-shrink-0 space-y-1">
                <label className="flex items-center gap-2 px-1 font-ui text-xs text-stone-600 dark:text-stone-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={top1}
                    onChange={(e) => setTop1(e.target.checked)}
                    className="rounded border-stone-300 dark:border-stone-600 text-accent focus:ring-accent/50"
                  />
                  Top suggestion only
                </label>
                <button
                  type="button"
                  onClick={() => setCurrentView('keys')}
                  className="px-1 font-ui text-xs text-accent dark:text-accent-light hover:underline"
                >
                  Browse all labelled keys
                </button>
              </div>
            )}

            {/* Filter chips */}
            <div className="px-3 py-1 flex-shrink-0">
              <FilterChips active={filter} onChange={handleFilter} />
            </div>

            {/* Evidence / attribution controls */}
            <div className="px-3 py-2 flex-shrink-0 space-y-2 border-t border-stone-200/60 dark:border-stone-700/60">
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1.5 block">
                  Evidence View
                </label>
                <ViewModeToggle />
              </div>
              {isPiAdmin && (
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1.5 block">
                    Attribution Controls
                  </label>
                  <AttributionMethodSelector />
                </div>
              )}
            </div>

            {/* Query list heading */}
            <div className="px-3 pt-2 pb-1 flex-shrink-0">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-400 px-2">
                Unlabeled Manuscripts
              </h2>
              <p className="font-ui text-xs text-stone-400 dark:text-stone-500 px-2 mt-0.5">
                Select a manuscript to review predictions
              </p>
              {keyFilter && data && (
                <p
                  data-testid="key-match-count"
                  className="font-ui text-xs font-medium text-stone-600 dark:text-stone-300 px-2 mt-1"
                >
                  {data.total === 1 ? '1 text' : `${data.total} texts`}
                </p>
              )}
            </div>

            {/* Query list */}
            <div className="flex-1 overflow-hidden mt-1">
              {loading && !data && (
                <div className="flex items-center justify-center h-20">
                  <span className="text-xs text-stone-400 font-ui">Loading...</span>
                </div>
              )}
              {data && data.items.length === 0 && (
                <div className="flex items-center justify-center h-20 px-4 text-center">
                  <span className="text-xs text-stone-400 font-ui">
                    {keyFilter
                      ? `No texts have a suggested key matching \u201c${keyFilter.key}\u201d`
                      : 'No files found'}
                  </span>
                </div>
              )}
              {data && data.items.length > 0 && (
                <Virtuoso
                  style={{ height: '100%' }}
                  totalCount={data.items.length}
                  itemContent={(index) => {
                    const item = data.items[index]
                    if (!item) return null
                    return (
                      <QueryCard
                        item={item}
                        isActive={item.file_id === activeQueryId}
                        onClick={() => setActiveQueryId(item.file_id)}
                      />
                    )
                  }}
                  endReached={handleEndReached}
                  components={{
                    Footer: () => (
                      <div className="flex justify-center py-3">
                        {data.has_more ? (
                          <button
                            type="button"
                            onClick={handleEndReached}
                            disabled={loading}
                            className="h-8 px-3 rounded-lg bg-stone-100 dark:bg-stone-700 text-xs font-ui font-medium text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          >
                            {loading ? 'Loading...' : 'Load more'}
                          </button>
                        ) : (
                          <span className="text-xs text-stone-400 dark:text-stone-500 font-ui">
                            {data.items.length} of {data.total}
                          </span>
                        )}
                      </div>
                    ),
                  }}
                />
              )}
            </div>
          </>
        )}

        {/* Collapsed icon strip */}
        {!isOpen && (
          <div className="flex flex-col items-center gap-3 pt-2">
            <button
              type="button"
              onClick={onToggle}
              className="w-7 h-7 rounded flex items-center justify-center
                         hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                         text-stone-500 dark:text-stone-400"
              aria-label="Expand sidebar and search"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </button>
            <button
              type="button"
              onClick={onToggle}
              className="w-7 h-7 rounded flex items-center justify-center
                         hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                         text-stone-500 dark:text-stone-400"
              aria-label="Expand sidebar and filter"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </motion.aside>
  )
}
