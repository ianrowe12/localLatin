import { BAND_COPY } from '../../utils/confidenceBands'
import { NO_MATCH_GUIDANCE } from '../../utils/reviewerDirectoryCopy'

interface NoMatchCalloutProps {
  /** Best similarity on offer, for the "best is X" line. */
  topScore: number | null
}

/**
 * The band-1 hint (issue #94, trimmed by #196): the best candidate scores below
 * the no-match threshold, so the ranking below is probably noise.
 *
 * WHAT THIS NO LONGER CARRIES. It used to hold issue #95's `NewDirectoryCta` --
 * the red "New directory / New file" button, its caption, and an
 * already-seeded fallback paragraph. Prof. Firey, 15 September 2026: the red
 * button and its explanation should go "to restore the elegance of the review
 * panel", and a new evaluator needs space around the ten prediction buttons
 * rather than a second, louder call to action beside them. Declaring a new
 * source now happens in one place, the blue "None of the top N" action in the
 * assessment panel, where an optional CCL key says what the source is.
 *
 * The red frame stays, because the fact it reports is still true and still
 * useful: a low score means the model has no useful opinion here, and knowing
 * that before reading ten candidates is worth a two-line notice. It is now a
 * hint and nothing else -- no buttons, no state, no writes.
 */
export default function NoMatchCallout({ topScore }: NoMatchCalloutProps) {
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
        {topScore != null && (
          <>
            {' '}
            Best similarity is{' '}
            <span className="font-mono">{topScore.toFixed(3)}</span>.
          </>
        )}
      </p>

      <p
        data-testid="no-match-caveat"
        className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300"
      >
        {NO_MATCH_GUIDANCE}
      </p>
    </div>
  )
}
