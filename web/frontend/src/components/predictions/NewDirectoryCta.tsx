import { useEffect, useRef, useState } from 'react'
import { createReviewerDir } from '../../api/reviewerDirs'

interface NewDirectoryCtaProps {
  queryId: number
  model: string
  /** Rendered prominently when the top candidate is below the no-match band. */
  emphasised: boolean
  filename?: string
  /**
   * Rendered inside the no-match callout rather than standing alone: drops the
   * outer padding and adopts the callout's red, so the two read as one block.
   */
  inline?: boolean
}

/**
 * "This is a new source" -- the creation flow for a reviewer directory.
 *
 * #94 HAS LANDED, so this now renders where that issue's design puts it: as the
 * default top option inside `NoMatchCallout`, at the head of the prediction
 * list, whenever the best candidate scores below the no-match band. That was
 * the integration point this component was written against, and taking it
 * needed no change to the component itself -- only to where it is mounted.
 *
 * It still renders un-emphasised at the foot of the list above the band, since
 * "this is a new source" is a judgement a reviewer can reach at any score.
 *
 * Two styling knobs, no behavioural ones: `emphasised` is the below-the-band
 * treatment (filled red, unmissable), `inline` drops the outer padding so it
 * sits inside the callout's frame.
 *
 * NAMING IS PART OF CREATION, so the button opens a one-field form rather than
 * posting immediately. Both tables are append-only with no rename, so a
 * directory's label is permanent from the moment it exists and a reviewer has
 * to get one chance at it. The field is pre-filled with the seed's filename, so
 * accepting the default is still just a second click.
 */
export default function NewDirectoryCta({
  queryId,
  model,
  emphasised,
  filename,
  inline = false,
}: NewDirectoryCtaProps) {
  const [open, setOpen] = useState(false)
  const [label, setLabel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [created, setCreated] = useState<string | null>(null)

  const suggestion = filename
    ? `New directory from ${filename.replace(/\.txt$/, '')}`
    : 'New directory'

  // The query this instance is currently showing, readable from inside an
  // in-flight request's closure (which still holds the click-time value).
  const currentQuery = useRef(queryId)

  // A different query is a different decision: never carry a created, failed or
  // half-open state across. PredictionList also keys this component by query id,
  // but the guard has to hold on its own for a reused instance.
  useEffect(() => {
    currentQuery.current = queryId
    setOpen(false)
    setLabel('')
    setSubmitting(false)
    setError(null)
    setCreated(null)
  }, [queryId])

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    // Staleness guard: a reviewer can navigate while the POST is in flight, and
    // a response that lands afterwards belongs to a query that is no longer on
    // screen. Dropping it here is what keeps "created" from being painted onto
    // the next fragment.
    const forQuery = queryId
    setSubmitting(true)
    setError(null)
    try {
      const dir = await createReviewerDir({
        query_file_id: forQuery,
        label: label.trim() || undefined,
        model_slug: model || undefined,
      })
      if (currentQuery.current !== forQuery) return
      setCreated(dir.label)
      setOpen(false)
      setLabel('')
    } catch (err) {
      if (currentQuery.current !== forQuery) return
      setError(err instanceof Error ? err.message : 'Could not create the directory')
    } finally {
      if (currentQuery.current === forQuery) setSubmitting(false)
    }
  }

  const pad = inline ? '' : 'mx-2'

  if (created) {
    return (
      <div
        data-testid="new-directory-created"
        className={`mt-2 ${pad} rounded-lg border border-indigo-300 dark:border-indigo-400/40 bg-indigo-50 dark:bg-indigo-500/10 px-3 py-2`}
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
      <div className={inline ? 'mt-2' : 'px-2 mt-2'}>
        <button
          type="button"
          data-testid="new-directory-cta"
          onClick={() => {
            setLabel(suggestion)
            setOpen(true)
          }}
          className={
            emphasised
              ? `w-full rounded-lg bg-incorrect px-2 py-1.5 text-xs font-semibold
                 text-white transition-colors hover:bg-incorrect-light
                 focus:outline-none focus:ring-2 focus:ring-incorrect/40`
              : 'w-full rounded-lg border border-dashed border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 hover:border-indigo-400 hover:text-indigo-600 text-xs font-ui px-3 py-2 transition-colors'
          }
        >
          {emphasised ? 'New directory / New file' : 'Start a new directory'}
        </button>
      </div>
    )
  }

  return (
    <form
      onSubmit={submit}
      className={inline ? 'mt-2' : 'px-2 mt-2'}
      data-testid="new-directory-form"
    >
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
        <p
          role="alert"
          data-testid="new-directory-error"
          className="text-xs text-red-600 dark:text-red-400 mt-1"
        >
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
