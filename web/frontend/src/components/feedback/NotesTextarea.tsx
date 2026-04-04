interface NotesTextareaProps {
  value: string
  onChange: (value: string) => void
}

export default function NotesTextarea({ value, onChange }: NotesTextareaProps) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="Add notes for this query..."
      className="w-full min-h-[60px] max-h-[160px] resize-y rounded-lg
                 border border-stone-200 dark:border-stone-700
                 bg-white dark:bg-surface-800 p-2 text-sm font-ui
                 text-stone-800 dark:text-stone-200
                 placeholder:text-stone-400 dark:placeholder:text-stone-500
                 focus:outline-none focus:ring-2 focus:ring-accent/30"
    />
  )
}
