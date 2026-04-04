import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Virtuoso } from 'react-virtuoso'
import { useApp } from '../../contexts/AppContext'
import { useQueryList } from '../../api/queries'
import { useStats } from '../../api/models'
import ProgressRing from '../sidebar/ProgressRing'
import SearchBar from '../sidebar/SearchBar'
import FilterChips from '../sidebar/FilterChips'
import QueryCard from '../sidebar/QueryCard'

interface LeftSidebarProps {
  isOpen: boolean
  onToggle: () => void
}

const FILTER_TO_STATUS: Record<string, string> = {
  all: 'all',
  pending: 'unreviewed',
  reviewed: 'reviewed',
  skipped: 'all',
}

export default function LeftSidebar({ isOpen, onToggle }: LeftSidebarProps) {
  const { activeQueryId, setActiveQueryId } = useApp()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(1)

  const apiStatus = FILTER_TO_STATUS[filter] ?? 'all'
  const { data, loading } = useQueryList(apiStatus, search, page)
  const { data: stats } = useStats()

  const handleSearch = useCallback((value: string) => {
    setSearch(value)
    setPage(1)
  }, [])

  const handleFilter = useCallback((f: string) => {
    setFilter(f)
    setPage(1)
  }, [])

  const handleEndReached = useCallback(() => {
    if (data?.has_more && !loading) {
      setPage((p) => p + 1)
    }
  }, [data?.has_more, loading])

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
            {/* Progress ring */}
            <div className="flex justify-center py-3 flex-shrink-0">
              <ProgressRing
                reviewed={stats?.reviewed_count ?? 0}
                total={stats?.total_queries ?? 0}
              />
            </div>

            {/* Search */}
            <div className="px-3 py-2 flex-shrink-0">
              <SearchBar value={search} onChange={handleSearch} />
            </div>

            {/* Filter chips */}
            <div className="px-3 py-1 flex-shrink-0">
              <FilterChips active={filter} onChange={handleFilter} />
            </div>

            {/* Query list */}
            <div className="flex-1 overflow-hidden mt-1">
              {loading && !data && (
                <div className="flex items-center justify-center h-20">
                  <span className="text-xs text-stone-400 font-ui">Loading...</span>
                </div>
              )}
              {data && data.items.length === 0 && (
                <div className="flex items-center justify-center h-20">
                  <span className="text-xs text-stone-400 font-ui">No files found</span>
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
