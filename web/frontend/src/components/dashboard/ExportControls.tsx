import { useMemo, useState } from 'react'
import { useModels } from '../../api/models'
import type { StatsResponse } from '../../api/models'

interface ExportControlsProps {
  stats: StatsResponse
}

export default function ExportControls({ stats }: ExportControlsProps) {
  const { data: models } = useModels()
  const [reviewer, setReviewer] = useState('')
  const [model, setModel] = useState('')
  const [outcome, setOutcome] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  const reviewers = useMemo(
    () => Object.keys(stats.reviews_by_reviewer).sort(),
    [stats.reviews_by_reviewer],
  )

  const handleExport = () => {
    const params = new URLSearchParams()
    if (reviewer) params.set('reviewer', reviewer)
    if (model) params.set('model', model)
    if (outcome) params.set('outcome', outcome)
    if (dateFrom) params.set('date_from', dateFrom)
    if (dateTo) params.set('date_to', dateTo)
    const query = params.toString()
    window.open(`/api/feedback/export${query ? `?${query}` : ''}`, '_blank')
  }

  return (
    <div className="glass-panel p-4">
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        <select
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          className="h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 text-sm font-ui"
          aria-label="Filter export by reviewer"
        >
          <option value="">All reviewers</option>
          {reviewers.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>

        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 text-sm font-ui"
          aria-label="Filter export by model"
        >
          <option value="">All models</option>
          {models?.map((item) => (
            <option key={item.slug} value={item.slug}>
              {item.display_name}
            </option>
          ))}
        </select>

        <select
          value={outcome}
          onChange={(e) => setOutcome(e.target.value)}
          className="h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 text-sm font-ui"
          aria-label="Filter export by outcome"
        >
          <option value="">All outcomes</option>
          <option value="matched_rank">Matched rank</option>
          <option value="none_of_top_k">None of top K</option>
          <option value="skipped">Skipped</option>
        </select>

        <input
          type="date"
          value={dateFrom}
          onChange={(e) => setDateFrom(e.target.value)}
          className="h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 text-sm font-ui"
          aria-label="Export date from"
        />

        <div className="flex gap-2">
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="min-w-0 flex-1 h-9 px-2 rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 text-sm font-ui"
            aria-label="Export date to"
          />
          <button
            type="button"
            onClick={handleExport}
            className="h-9 px-3 rounded-lg bg-accent text-white text-sm font-ui font-medium hover:bg-accent-dark transition-colors"
          >
            Export
          </button>
        </div>
      </div>
    </div>
  )
}
