import { useEffect, useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictionState } from '../../contexts/PredictionContext'
import {
  useSavedDirectory,
} from '../../contexts/SavedDirectoryContext'
import { useModels } from '../../api/models'
import { useKeyboardShortcuts } from '../../utils/keyboard'
import type { ApiErrorInfo } from '../../api/client'
import type { ExclusionReason } from '../../api/queries'
import type { ReviewerDir } from '../../api/reviewerDirs'
import {
  BAND_COPY,
  BAND_STYLES,
  bandsFrom,
  getConfidenceBand,
} from '../../utils/confidenceBands'
import AwaitingMatchBadge from './AwaitingMatchBadge'
import NoMatchCallout from './NoMatchCallout'
import PredictionCard from './PredictionCard'
import ReviewerDirCard from './ReviewerDirCard'
import SavedDirectoryNotice from './SavedDirectoryNotice'

/** Stable empty list, so an unknown record does not remount the badge. */
const EMPTY_DIRS: readonly ReviewerDir[] = []

/**
 * What the reviewer is told when the request failed.
 *
 * Every one of these used to render as "No predictions available", which reads
 * as "the model found nothing" -- a statement about the corpus -- when it in
 * fact meant "we never found out" (issue #156). The server's own prose is used
 * where it exists; a response body is never dumped.
 */
function errorCopy(error: ApiErrorInfo): { title: string; detail: string } {
  if (error.kind === 'network') {
    return {
      title: 'Could not reach the server',
      detail:
        'This document has no ranking on screen. Nothing you have typed has been lost.',
    }
  }
  if (error.kind === 'malformed') {
    return {
      title: 'Unreadable ranking',
      detail:
        'The server answered, but not with a ranking this app can read. Nothing here can be assessed until it does.',
    }
  }
  if (error.status === 401) {
    return {
      title: 'Signed out',
      detail: 'Your session has ended. Sign in again to load this ranking.',
    }
  }
  if (error.status === 403) {
    return { title: 'Not permitted', detail: error.message }
  }
  if (error.status === 404) {
    return { title: 'Ranking not found', detail: error.message }
  }
  if (error.status !== null && error.status >= 500) {
    return {
      title: 'The server could not answer',
      detail: `${error.message} This is a failure to load the ranking, not a verdict on this document.`,
    }
  }
  return { title: 'Could not load the ranking', detail: error.message }
}

/**
 * The retrieval run's stated reason for there being no ranking at all.
 *
 * A non-evaluable result: not an error, and not a "None of top k" answer. The
 * model was never given a usable vector for this document, so there is nothing
 * here for a reviewer to be right or wrong about. The two guards get two
 * explanations, because they are two different facts about the document.
 */
function exclusionCopy(reason: ExclusionReason, raw: string | null): string {
  if (reason === 'blank_source') {
    return 'The retrieval run excluded this document because its source text was blank, so this model produced no ranking for it. There is nothing here to assess.'
  }
  if (reason === 'zero_norm') {
    return 'The retrieval run excluded this document because its embedding for this model was degenerate (zero norm), so no ranking was produced. There is nothing here to assess.'
  }
  return `The retrieval run excluded this document from this model's ranking${
    raw ? ` (${raw})` : ''
  }. There is nothing here to assess.`
}

export default function PredictionList() {
  const {
    activeQueryId,
    activeModel,
    activePredictionRank,
    setActivePredictionRank,
    overrideCandidateDir,
    setOverrideCandidateDir,
  } = useApp()
  const {
    key,
    generation,
    phase,
    error,
    exclusion,
    sourceStatus,
    response,
    predictions: allPredictions,
    modelPredictions,
    reviewerDirs,
    refresh,
  } = usePredictionState()

  // If a gallery override candidate happens to be in the current model's
  // top-10, sync the prediction list selection and release the override so
  // the normal flow takes over. Otherwise leave the override intact —
  // CenterArea uses it directly to render an off-list candidate.
  useEffect(() => {
    if (!overrideCandidateDir || allPredictions.length === 0) return
    const match = allPredictions.find((p) => p.dir_name === overrideCandidateDir)
    if (match) {
      setActivePredictionRank(match.rank)
      setOverrideCandidateDir(null)
    }
  }, [
    overrideCandidateDir,
    allPredictions,
    setActivePredictionRank,
    setOverrideCandidateDir,
  ])

  // Thresholds come from the deployment (GET /api/models), not from a literal
  // in this bundle: the backend decides reviewer-directory status with the same
  // numbers, so there is one source of truth and it is server-side.
  const { data: models } = useModels()
  const bands = bandsFrom(models)

  // The ranked list is the model's candidates and nothing else (issue #196).
  // Reviewer directories arrive in their own unranked field and are drawn
  // below, under their own heading.
  const predictions = modelPredictions

  /**
   * Whether this document already seeds a reviewer directory -- held in the
   * durable store rather than in this component (issue #161).
   *
   * `SavedDirectoryStore` is keyed by seed query and lives above
   * `PredictionProvider`, so the answer survives loading, errors, empty
   * responses, model switches, navigation away and back, and a reload (it is
   * re-established from the server, never from browser storage). Nothing in
   * this panel writes to it any more (issue #196); what it still buys is that a
   * permanent grouping is never hidden by a request that happens to be in
   * flight or to have failed.
   */
  const { identity } = useSavedDirectory(activeQueryId)

  /**
   * Directory status comes from the record, and ONLY from the record.
   *
   * `seeded_dirs` is deliberately not read here. It is raw response data, and
   * the response may be the previous session's: `PredictionProvider` sits above
   * the auth gates and its key carries no account, so a settled ranking, its
   * `seeded_dirs` and its cache all survive a sign-out.
   * `SavedDirectoryObservations` is the one place that decides whether a
   * response is this session's own answer, and reading past it here would
   * reinstate exactly what that admission exists to stop -- for one painted
   * frame, which is one frame of telling a reviewer that a document is already
   * grouped on somebody else's evidence.
   *
   * It costs nothing: admission records the same rows a render later, and until
   * it does, "not yet known" is the honest answer rather than a claim borrowed
   * from a request nobody has vouched for. Nor is this a second admission rule.
   * The store is the single authority and this reads its answer.
   */
  const savedDirs = identity.status === 'saved' ? identity.dirs : EMPTY_DIRS

  // The band of the *best model hit* decides the whole list's framing (issue
  // #94): that is the number a reviewer reads first, and the one that says
  // whether this fragment has a plausible home in the labelled corpus at all.
  // Reviewer directories are deliberately not part of that judgement — one
  // colleague's new directory scoring 0.6 does not mean the corpus has a home
  // for this fragment.
  const topScore = modelPredictions.length > 0 ? modelPredictions[0].score : null
  const topBand = topScore != null ? getConfidenceBand(topScore, bands) : null

  // Arrow keys walk the ranks that actually exist rather than counting by one.
  // A ranking can be sparse (a model row with fewer than ten candidates), so
  // stepping numerically would land on a rank with no card behind it.
  const handlers = useMemo(() => {
    const ranks = predictions.map((p) => p.rank)
    const step = (delta: number) => (e: KeyboardEvent) => {
      e.preventDefault()
      if (ranks.length === 0) return
      const current = ranks.indexOf(activePredictionRank)
      const next = current === -1 ? 0 : current + delta
      const clamped = Math.min(Math.max(next, 0), ranks.length - 1)
      setActivePredictionRank(ranks[clamped])
    }
    return {
      ArrowUp: step(-1),
      ArrowDown: step(1),
      // Number keys address the nth card in the list, which for the model's own
      // candidates is also its rank.
      ...Object.fromEntries(
        Array.from({ length: 9 }, (_, i) => [
          String(i + 1),
          () => {
            if (i < ranks.length) setActivePredictionRank(ranks[i])
          },
        ]),
      ),
    }
  }, [activePredictionRank, predictions, setActivePredictionRank])

  useKeyboardShortcuts(handlers)

  const select = (rank: number) => {
    setActivePredictionRank(rank)
    setOverrideCandidateDir(null)
  }

  const retryButton = (
    <button
      type="button"
      data-testid="predictions-retry"
      onClick={refresh}
      className="mt-2 rounded-md border border-stone-300 dark:border-stone-600 px-3 py-1 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
    >
      Try again
    </button>
  )

  // The saved-directory acknowledgement is shown in every phase, because it is
  // not a claim about the ranking: a directory this document seeds is a
  // permanent record that outlives loading, errors and empty responses. It is
  // now read-only -- nothing in this panel creates a directory since issue #196
  // retired the red button.
  const acknowledgementShown = identity.status === 'saved'

  return (
    <div data-tour="predictions" className="px-2 py-2">
      <div className="flex items-center gap-2 mb-1 px-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-stone-400">
          Predicted Sources
        </span>
        {/* Status is stated once. When the durable acknowledgement is on screen
            it carries the badge itself, beside the directory it describes and
            the person who created it, and that copy survives a failed refresh;
            repeating the same chip in the header would be the same fact twice
            in one narrow panel. */}
        <AwaitingMatchBadge seededDirs={acknowledgementShown ? EMPTY_DIRS : savedDirs} />
      </div>
      <p className="font-ui text-xs text-stone-400 dark:text-stone-500 px-2 mb-2">
        Ranked by similarity
      </p>

      {phase === 'idle' && (
        <p
          data-testid="predictions-no-selection"
          className="text-xs text-stone-400 dark:text-stone-500 text-center mt-4 font-ui"
        >
          Select a query to see predictions
        </p>
      )}

      {phase === 'loading' && (
        <p
          role="status"
          data-testid="predictions-loading"
          className="text-xs text-stone-400 text-center mt-4 font-ui"
        >
          Loading...
        </p>
      )}

      {phase === 'error' && error !== null && (
        <div
          role="alert"
          data-testid="predictions-error"
          className="mt-2 rounded-lg border border-incorrect/50 bg-incorrect/10 px-2.5 py-2"
        >
          <div className="text-xs font-bold uppercase tracking-wide text-incorrect">
            {errorCopy(error).title}
          </div>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            {errorCopy(error).detail}
          </p>
          {retryButton}
        </div>
      )}

      {phase === 'excluded' && (
        <div
          role="status"
          data-testid="predictions-excluded"
          className="mt-2 rounded-lg border border-stone-300 dark:border-stone-600 bg-stone-100 dark:bg-stone-800 px-2.5 py-2"
        >
          <div className="text-xs font-bold uppercase tracking-wide text-stone-500 dark:text-stone-300">
            Not assessable for this model
          </div>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            {exclusionCopy(exclusion ?? 'other', sourceStatus)}
          </p>
        </div>
      )}

      {phase === 'empty' && (
        <div
          role="status"
          data-testid="predictions-empty"
          className="mt-2 rounded-lg border border-stone-300 dark:border-stone-600 px-2.5 py-2"
        >
          <div className="text-xs font-bold uppercase tracking-wide text-stone-500 dark:text-stone-300">
            No ranking came back
          </div>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            The server answered for this document, but with no candidates and no
            reason recorded. Why is unknown, so nothing here should be read as a
            verdict on the document.
          </p>
          {retryButton}
        </div>
      )}

      {(phase === 'ready' || phase === 'empty' || phase === 'excluded') && (
        <div className="flex flex-col gap-1 mt-1">
          {/* Band treatment for the top hit: a hint that the ranking below is
              probably noise, and nothing else. The action that used to live
              inside it is retired (issue #196). */}
          {topBand === 'no_match' && activeQueryId != null && (
            <NoMatchCallout topScore={topScore} />
          )}
          {topBand === 'careful' && (
            <p
              data-testid="careful-review-note"
              className={`mb-2 rounded-lg px-2.5 py-2 font-ui text-xs leading-snug ${BAND_STYLES.careful.chip}`}
            >
              <span className="font-semibold">{BAND_COPY.careful.label}.</span>{' '}
              {BAND_COPY.careful.note}
            </p>
          )}
          {topBand === 'likely' && (
            <p
              data-testid="likely-match-note"
              className="mb-2 rounded-lg bg-stone-100 px-2.5 py-2 font-ui text-xs leading-snug text-stone-600 dark:bg-stone-800 dark:text-stone-300"
            >
              <span className="font-semibold">{BAND_COPY.likely.label}.</span>{' '}
              {BAND_COPY.likely.note}
            </p>
          )}
          {modelPredictions.map((pred) => (
            <PredictionCard
              key={pred.rank}
              prediction={pred}
              rank={pred.rank}
              bands={bands}
              isActive={
                !overrideCandidateDir && pred.rank === activePredictionRank
              }
              onClick={() => select(pred.rank)}
            />
          ))}
          {/* Reviewer directories: their own block, under their own heading,
              WITHOUT ranks (issue #196). They are not the model's answers, and
              a numbered slot beside the ten said they were. Choosing one is not
              a rank press either -- a document is filed here by naming the
              directory's CCL key beside "None of the top N" -- so clicking a
              card only opens its documents for reading. */}
          {reviewerDirs.length > 0 && (
            <div className="mt-3 pt-2 border-t border-stone-200 dark:border-stone-700">
              <div
                data-testid="reviewer-dirs-heading"
                className="text-[10px] font-semibold uppercase tracking-wider text-indigo-500/80 dark:text-indigo-300/70 px-2"
              >
                Directories created by reviewers
              </div>
              <p className="font-ui text-[11px] leading-snug text-stone-400 dark:text-stone-500 px-2 mb-1">
                Not ranked, and not part of the model's answer. Open one to read
                its documents.
              </p>
              <div className="flex flex-col gap-1">
                {reviewerDirs.map((card) => (
                  <ReviewerDirCard
                    key={card.dir_id}
                    card={card}
                    isActive={overrideCandidateDir === card.dir_id}
                    onClick={() => setOverrideCandidateDir(card.dir_id)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* What is already recorded about this document's own directory, in every
          phase: a permanent record must not be a casualty of a refetch. No
          control here writes anything. */}
      {activeQueryId !== null && (
        <SavedDirectoryNotice
          key={activeQueryId}
          queryId={activeQueryId}
          model={activeModel}
        />
      )}
    </div>
  )
}
