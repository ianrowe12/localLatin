import { useState } from 'react'
import { createReviewerDir } from '../../api/reviewerDirs'

interface NewDirectoryCtaProps {
  queryId: number
  model: string
  /** Rendered prominently when the top candidate is below the no-match band. */
  emphasised: boolean
  filename?: string
}

/**
 * "This is a new source" -- the creation flow for a reviewer directory.
 *
 * INTEGRATION POINT FOR #94. That issue makes this the *default* top option
 * whenever the best candidate scores below the no-match band, rendering the
 * call to action at the head of the prediction list. Until it merges, this
 * component carries its own trigger at the foot of the list so the loop is
 * usable and testable end to end. When #94 lands it should render
 * `<NewDirectoryCta emphasised />` in the position it wants and drop the
 * fallback placement in PredictionList; nothing about this component's
 * behaviour needs to change, only where it is mounted.
 *
 * `emphasised` is the only styling knob: emphasised is the below-the-band
 * treatment (filled, unmissable), plain is the always-available escape hatch.
 */
export default function NewDirectoryCta({
  queryId,
  model,
  emphasised,
  filename,
}: NewDirectoryCtaProps) {
  const [open, setOpen] = useState(false)
  const [label, setLabel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [created, setCreated] = useState<string | null>(null)

  const suggestion = filename
    ? `New directory from ${filename.replace(/\.txt$/, '')}`
    : 'New directory'

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const dir = await createReviewerDir({
        query_file_id: queryId,
        label: label.trim() || undefined,
        model_slug: model || undefined,
      })
      setCreated(dir.label)
      setOpen(false)
      setLabel('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the directory')
    } finally {
      setSubmitting(false)
    }
  }

  if (created) {
    return (
      <div
        data-testid="new-directory-created"
        className="mt-2 mx-2 rounded-lg border border-indigo-300 dark:border-indigo-400/40 bg-indigo-50 dark:bg-indigo-500/10 px-3 py-2"
      >
        <div className="text-xs font-semibold text-indigo-700 dark:text-indigo-300">
          Directory created
        </div>
        <div className="font-ui text-xs text-stone-600 dark:text-stone-400 mt-0.5">
          {created} — it is now a candidate for every other document.
        </div>
      </div>
    )
  }

  if (!open) {
    return (
      <div className="px-2 mt-2">
        <button
          type="button"
          data-testid="new-directory-cta"
          onClick={() => setOpen(true)}
          className={
            emphasised
              ? 'w-full rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-3 py-2 transition-colors'
              : 'w-full rounded-lg border border-dashed border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 hover:border-indigo-400 hover:text-indigo-600 text-xs font-ui px-3 py-2 transition-colors'
          }
        >
          {emphasised ? 'None of these — start a new directory' : 'Start a new directory'}
        </button>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="px-2 mt-2" data-testid="new-directory-form">
      <label
        className="block text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-1"
        htmlFor="reviewer-dir-label"
      >
        Name the new directory
      </label>
      <input
        id="reviewer-dir-label"
        type="text"
        value={label}
        onChange={(event) => setLabel(event.target.value)}
        placeholder={suggestion}
        maxLength={200}
        autoFocus
        className="w-full rounded-md border border-stone-300 dark:border-stone-600 bg-white dark:bg-surface-800 px-2 py-1.5 text-sm"
      />
      {error && (
        <p role="alert" className="text-xs text-red-600 dark:text-red-400 mt-1">
          {error}
        </p>
      )}
      <div className="flex gap-2 mt-2">
        <button
          type="submit"
          disabled={submitting}
          data-testid="new-directory-submit"
          className="flex-1 rounded-md bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-xs font-medium px-3 py-1.5"
        >
          {submitting ? 'Creating…' : 'Create directory'}
        </button>
        <button
          type="button"
          onClick={() => {
            setOpen(false)
            setError(null)
          }}
          className="rounded-md border border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 text-xs px-3 py-1.5"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}
