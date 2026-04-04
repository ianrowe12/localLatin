import { memo } from 'react'
import type { QueryListItem } from '../../api/queries'

interface QueryCardProps {
  item: QueryListItem
  isActive: boolean
  onClick: () => void
}

function QueryCardInner({ item, isActive, onClick }: QueryCardProps) {
  const isReviewed = item.review_status === 'reviewed'
  const preview =
    item.text_preview.length > 60
      ? item.text_preview.slice(0, 60) + '...'
      : item.text_preview

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
          className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${
            isReviewed ? 'bg-correct' : 'bg-highlight'
          }`}
          aria-hidden="true"
        />
        <span className="text-xs text-stone-500 dark:text-stone-400 font-ui">
          {isReviewed ? 'Reviewed' : 'Pending'}
        </span>
      </div>

      {preview && (
        <p className="text-xs text-stone-400 truncate mt-1 font-latin">
          {preview}
        </p>
      )}
    </button>
  )
}

const QueryCard = memo(QueryCardInner, (prev, next) => {
  return prev.item.file_id === next.item.file_id && prev.isActive === next.isActive
})

export default QueryCard
