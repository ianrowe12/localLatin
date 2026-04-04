interface DocumentHeaderProps {
  filename: string
  dirLabel?: string
  score?: number
  side: 'query' | 'candidate'
}

export default function DocumentHeader({
  filename,
  dirLabel,
  score,
}: DocumentHeaderProps) {
  return (
    <div className="h-10 px-3 flex items-center gap-2 border-b border-stone-200 dark:border-stone-700 bg-stone-50 dark:bg-surface-800 flex-shrink-0">
      <span className="font-medium text-sm truncate">{filename}</span>

      {dirLabel && (
        <span
          className="text-xs text-stone-500 truncate max-w-[200px]"
          title={dirLabel}
        >
          {dirLabel}
        </span>
      )}

      <span className="flex-1" />

      {score != null && (
        <span className="bg-accent/10 text-accent text-xs px-2 py-0.5 rounded-full font-mono whitespace-nowrap">
          {score.toFixed(3)}
        </span>
      )}
    </div>
  )
}
