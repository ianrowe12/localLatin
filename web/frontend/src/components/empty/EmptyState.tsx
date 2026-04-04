interface EmptyStateProps {
  message?: string
}

export default function EmptyState({
  message = 'Select a query to begin reviewing',
}: EmptyStateProps) {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center">
        <div className="text-4xl mb-3 opacity-20" aria-hidden="true">
          {'\u{1F4DC}'}
        </div>
        <p className="text-stone-400 dark:text-stone-500 font-ui text-sm">
          {message}
        </p>
      </div>
    </div>
  )
}
