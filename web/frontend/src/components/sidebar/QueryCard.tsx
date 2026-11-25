import { memo } from 'react'
import type { QueryListItem } from '../../api/queries'

interface QueryCardProps {
  item: QueryListItem
  isActive: boolean
  onClick: () => void
}

function QueryCardInner({ item, isActive, onClick }: QueryCardProps) {
  const isReviewed = item.review_status === 'reviewed'
  const isSkipped = item.review_status === 'skipped'
  const statusLabel = isReviewed ? 'Reviewed' : isSkipped ? 'Skipped' : 'Pending'
  const statusColor = isReviewed
    ? 'bg-correct'
    : isSkipped
      ? 'bg-stone-400 dark:bg-stone-500'
      : 'bg-highlight'
  const preview =
    item.text_preview.length > 60
      ? item.text_preview.slice(0, 60) + '...'
      : item.text_preview

  const matchChip = item.key_matches?.[0]
  const extraMatches = (item.key_matches?.length ?? 0) - 1
  const chipTitle = item.key_matches
    ?.map((m) => `${m.dir_name} · rank ${m.rank}`)
    .join('\n')

  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left p-2.5 rounded-lg cursor-pointer transition-colors
        ${
          isActive
            ? 'border-l-2 border-accent bg-accent/5 dark:bg-accent/10'
            : 'border-l-2 border-transparent hover:bg-stone-100 dark:hover:bg-stone-800'
        }`}
      aria-current={isActive ? 'true' : undefined}
    >
      {/* Filename */}
      <div className="font-medium text-sm text-stone-700 dark:text-stone-200 truncate font-ui">
        {item.filename}
      </div>

      {/* Status badge + preview */}
      <div className="flex items-center gap-1.5 mt-1">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${statusColor}`}
          aria-hidden="true"
        />
        <span className="text-xs text-stone-500 dark:text-stone-400 font-ui">
          {statusLabel}
        </span>
      </div>

      {matchChip && (
        <span
          data-testid="key-match-chip"
          title={chipTitle}
          className="mt-1 inline-flex max-w-full items-center gap-1 rounded-full
                     bg-accent/10 dark:bg-accent/20 px-2 py-0.5
                     text-[11px] font-ui font-medium text-accent dark:text-accent-light"
        >
          <span className="truncate whitespace-pre">{matchChip.dir_name}</span>
          <span className="flex-shrink-0">· #{matchChip.rank}</span>
          {extraMatches > 0 && (
            <span className="flex-shrink-0 text-stone-500 dark:text-stone-400">
              +{extraMatches}
            </span>
          )}
        </span>
      )}

      {preview && (
        <p className="text-xs text-stone-400 truncate mt-1 font-latin">
          {preview}
        </p>
      )}
    </button>
  )
}

const QueryCard = memo(QueryCardInner, (prev, next) => {
  return (
    prev.item.file_id === next.item.file_id &&
    prev.isActive === next.isActive &&
    prev.item.key_matches === next.item.key_matches
  )
})

export default QueryCard
