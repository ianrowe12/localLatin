import { BAND_COPY } from '../../utils/confidenceBands'
import NewDirectoryCta from './NewDirectoryCta'

interface NoMatchCalloutProps {
  /** Query whose top hit fell below the no-match threshold. */
  queryFileId: number
  /** Best similarity on offer, for the "best is X" line. */
  topScore: number | null
  /** How many ranks the feedback pills offer, so the fallback copy matches. */
  topK: number
  /** Model slug the directory would be created under. */
  model: string
  /** Seed filename, used to suggest a default label. */
  filename?: string
  /** This document already seeds a directory, so creation is not on offer. */
  alreadySeeded: boolean
}

/**
 * The band-1 treatment (issue #94): when the best candidate scores below the
 * no-match threshold, the reviewer must see in a split second that the ranking
 * below is probably noise, and the *default top option* is to say so rather
 * than to pick a rank.
 *
 * The callout is #94's: red frame, warning glyph, `role="alert"`, the band copy
 * and the best-similarity line. The action inside it is #95's `NewDirectoryCta`,
 * which is the integration point that component always documented -- it now
 * renders here, emphasised, instead of at the foot of the list. #94's own
 * button and its `CtaState` machine are gone: they existed to call an endpoint
 * that did not exist yet, including a "coming with the next update" branch that
 * this PR makes unreachable.
 *
 * Creation state (submitting, created, error) lives in `NewDirectoryCta`, and
 * `PredictionList` keys this component by query id so none of it survives a
 * navigation.
 */
export default function NoMatchCallout({
  queryFileId,
  topScore,
  topK,
  model,
  filename,
  alreadySeeded,
}: NoMatchCalloutProps) {
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

      {alreadySeeded ? (
        <p
          data-testid="no-match-already-seeded"
          className="mt-2 font-ui text-xs italic text-stone-600 dark:text-stone-300"
        >
          You already started a directory from this document. Record your
          assessment as &ldquo;None of top {topK}&rdquo; if none of the ranked
          candidates fits.
        </p>
      ) : (
        <NewDirectoryCta
          queryId={queryFileId}
          model={model}
          filename={filename}
          emphasised
          inline
        />
      )}
    </div>
  )
}
