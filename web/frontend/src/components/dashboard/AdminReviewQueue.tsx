import { useEffect, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useReviewer } from '../../contexts/ReviewerContext'
import {
  fetchReviewQueue, QUEUE_FILTERS, REVIEW_STATUS_LABELS,
  type AdminReviewQueue as Queue, type QueueFilter,
} from '../../api/adminReview'
import AdminReviewAnswers from '../feedback/AdminReviewAnswers'

const BUTTON = 'min-h-11 px-3 rounded-lg border border-stone-200 dark:border-stone-700 text-sm font-ui hover:bg-stone-100 dark:hover:bg-stone-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors'

export default function AdminReviewQueue() {
  const { isPiAdmin, user } = useReviewer()
  return isPiAdmin ? <QueuePanel key={user?.id} /> : null
}

function QueuePanel() {
  const { navigateToQuery } = useApp()
  const [filter, setFilter] = useState<QueueFilter>('conflict')
  const [page, setPage] = useState(1)
  const [revision, setRevision] = useState(0)
  const [result, setResult] = useState<{ key: string; data: Queue } | null>(null)
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null)
  const key = `${filter}:${page}:${revision}`
  const data = result?.key === key ? result.data : null
  const error = failure?.key === key ? failure.message : null

  useEffect(() => {
    let cancelled = false
    fetchReviewQueue(filter, page).then(
      (data) => { if (!cancelled) setResult({ key, data }) },
      (err: unknown) => {
        if (!cancelled) setFailure({ key, message: err instanceof Error ? err.message : 'Could not load review queue.' })
      },
    )
    return () => { cancelled = true }
  }, [filter, page, key])

  return (
    <section className="glass-panel p-4 space-y-4 font-ui" aria-label="Admin review queue">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Review queue</h2>
          <p className="text-sm text-stone-500 dark:text-stone-400">Compare counted answers and record a final answer in the review view. PI/admin only.</p>
        </div>
        <button type="button" className={BUTTON} onClick={() => setRevision((value) => value + 1)}>Refresh queue</button>
      </div>
      <div className="flex flex-wrap gap-2" aria-label="Queue filters">
        {QUEUE_FILTERS.map((option) => (
          <button
            type="button" key={option.value} aria-pressed={filter === option.value}
            className={`${BUTTON} ${filter === option.value ? 'bg-stone-100 dark:bg-stone-700 font-semibold' : ''}`}
            onClick={() => { setFilter(option.value); setPage(1) }}
          >
            {option.label}{data ? ` (${data.counts[option.value]})` : ''}
          </button>
        ))}
      </div>
      {error ? <p role="alert" className="text-sm text-red-700 dark:text-red-300">{error} Use Refresh queue to retry.</p>
        : !data ? <p role="status" className="text-sm text-stone-500">Loading review queue...</p>
        : <>
          {data.items.length === 0 ? <p className="text-sm text-stone-500 dark:text-stone-400">No documents match this filter.</p> : (
            <ul className="divide-y divide-stone-200 dark:divide-stone-700">
              {data.items.map((item) => (
                <li key={item.xml_id} className="py-4 space-y-3 min-w-0">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <button type="button" className={`${BUTTON} text-accent dark:text-accent-light break-all text-left`} onClick={() => navigateToQuery(item.query_id)}>
                      Open {item.xml_id}
                    </button>
                    <span className="text-sm text-stone-600 dark:text-stone-300">
                      {item.deciding_answer ? 'Decided' : REVIEW_STATUS_LABELS[item.status]}, {item.reviewer_count} {item.reviewer_count === 1 ? 'reviewer' : 'reviewers'}
                    </span>
                  </div>
                  <AdminReviewAnswers answers={item.answers} />
                  {item.deciding_answer && (
                    <p className="text-sm font-medium text-stone-700 dark:text-stone-200">
                      Final answer: {item.deciding_answer.ccl_key ?? 'None of these'}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-stone-500 dark:text-stone-400" role="status">{data.total} {data.total === 1 ? 'document' : 'documents'}, Page {data.page}</p>
            <div className="flex gap-2">
              <button type="button" className={BUTTON} disabled={page === 1} onClick={() => setPage((value) => value - 1)}>Previous page</button>
              <button type="button" className={BUTTON} disabled={!data.has_more} onClick={() => setPage((value) => value + 1)}>Next page</button>
            </div>
          </div>
        </>}
    </section>
  )
}
