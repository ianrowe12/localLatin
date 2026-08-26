import { useEffect, useState } from 'react'
import { createReviewerDir } from '../../api/reviewerDirs'
import { BAND_COPY } from '../../utils/confidenceBands'

interface NoMatchCalloutProps {
  /** Query whose top hit fell below the no-match threshold. */
  queryFileId: number
  /** Best similarity on offer, for the "best is X" line. */
  topScore: number | null
}

type CtaState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'created'; label: string }
  | { kind: 'unavailable' }
  | { kind: 'error'; message: string }

/**
 * The band-1 treatment (issue #94): when the best candidate scores below
 * NO_MATCH_THRESHOLD, the reviewer must see in a split second that the ranking
 * below is probably noise, and the *default top option* is to say so rather
 * than to pick a rank.
 *
 * The button is a call to action for the new-directory loop (issue #95). It
 * calls POST /api/reviewer_dirs; on a deployment where that endpoint does not
 * exist yet it degrades to a friendly "coming with the next update" state
 * instead of an error, so shipping this ahead of the backend is safe.
 */
export default function NoMatchCallout({ queryFileId, topScore }: NoMatchCalloutProps) {
  const [state, setState] = useState<CtaState>({ kind: 'idle' })

  // A different query is a different decision: never carry a created/failed
  // state across.
  useEffect(() => {
    setState({ kind: 'idle' })
  }, [queryFileId])

  const handleClick = async () => {
    setState({ kind: 'submitting' })
    const result = await createReviewerDir(queryFileId)
    if (result.status === 'created') {
      setState({ kind: 'created', label: result.dir.label || result.dir.dir_id })
    } else if (result.status === 'unavailable') {
      setState({ kind: 'unavailable' })
    } else {
      setState({ kind: 'error', message: result.message })
    }
  }

  return (
    <div
      data-testid="no-match-callout"
      role="alert"
      className="mb-2 rounded-lg border border-incorrect/50 bg-incorrect/10 px-2.5 py-2"
    >
      <div className="flex items-center gap-1.5">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className="flex-shrink-0 text-incorrect"
        >
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <span className="text-xs font-bold uppercase tracking-wide text-incorrect">
          {BAND_COPY.no_match.label}
        </span>
      </div>

      <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
        {BAND_COPY.no_match.note}
        {topScore !== null && (
          <>
            {' '}
            Best similarity is{' '}
            <span className="font-mono">{topScore.toFixed(3)}</span>.
          </>
        )}
      </p>

      {state.kind === 'created' ? (
        <p
          data-testid="no-match-cta-created"
          className="mt-2 font-ui text-xs text-stone-700 dark:text-stone-200"
        >
          New directory <span className="font-medium">{state.label}</span> created and
          awaiting a match.
        </p>
      ) : (
        <button
          type="button"
          onClick={handleClick}
          disabled={state.kind === 'submitting'}
          className="mt-2 w-full rounded-lg bg-incorrect px-2 py-1.5 text-xs font-semibold
                     text-white transition-colors hover:bg-incorrect-light
                     focus:outline-none focus:ring-2 focus:ring-incorrect/40
                     disabled:cursor-not-allowed disabled:opacity-60"
        >
          {state.kind === 'submitting' ? 'Creating...' : 'New directory / New file'}
        </button>
      )}

      {state.kind === 'unavailable' && (
        <p
          data-testid="no-match-cta-unavailable"
          className="mt-1.5 font-ui text-xs italic text-stone-500 dark:text-stone-400"
        >
          Creating directories is coming with the next update. For now, record this as
          &ldquo;None of top N&rdquo; with a note.
        </p>
      )}

      {state.kind === 'error' && (
        <p
          data-testid="no-match-cta-error"
          className="mt-1.5 font-ui text-xs text-incorrect"
        >
          {state.message}
        </p>
      )}
    </div>
  )
}
