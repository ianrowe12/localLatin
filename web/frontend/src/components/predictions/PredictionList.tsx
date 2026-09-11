import { useEffect, useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictionState } from '../../contexts/PredictionContext'
import {
  useSavedDirectory,
  useSavedDirectoryStore,
} from '../../contexts/SavedDirectoryContext'
import { useModels } from '../../api/models'
import { useKeyboardShortcuts } from '../../utils/keyboard'
import type { ApiErrorInfo } from '../../api/client'
import type { ExclusionReason } from '../../api/queries'
import {
  BAND_COPY,
  BAND_STYLES,
  bandsFrom,
  getConfidenceBand,
} from '../../utils/confidenceBands'
import AwaitingMatchBadge from './AwaitingMatchBadge'
import NewDirectoryCta from './NewDirectoryCta'
import NoMatchCallout from './NoMatchCallout'
import PredictionCard from './PredictionCard'
import ReviewerDirCard from './ReviewerDirCard'

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
    reviewerPredictions,
    seededDirs,
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

  // The model contributes at most ten candidates; reviewer-created directories
  // are appended after them by the API and are not subject to that cap. The
  // provider slices the two groups separately for exactly that reason.
  const predictions = useMemo(
    () => [...modelPredictions, ...reviewerPredictions],
    [modelPredictions, reviewerPredictions],
  )

  /**
   * Whether this document already seeds a reviewer directory, and what is
   * happening to any creation of one -- held in the durable store rather than in
   * this component (issue #161).
   *
   * Creating a directory broadcasts `REVIEWER_DIRS_UPDATED_EVENT`, which
   * refetches the ranking, which replaces this subtree with `Loading...` and
   * used to take the acknowledgement of a permanent, unrepeatable write down
   * with it. Holding it one level up fixed the unmount but not the rest: a
   * refetch that then FAILED still left no trace of the write, a reload lost it
   * entirely, and a response whose outcome never arrived left a Create button in
   * front of a directory that already existed.
   *
   * `SavedDirectoryStore` is keyed by seed query and lives above
   * `PredictionProvider`, so the answer survives loading, errors, empty
   * responses, model switches, navigation away and back, and a reload (it is
   * re-established from the server, never from browser storage).
   */
  const savedDirectories = useSavedDirectoryStore()
  const { identity, creation } = useSavedDirectory(activeQueryId)

  // The phases in which the provider is speaking for the current key. `loading`
  // and `error` are statements about a request, not about the database.
  const settled = phase === 'ready' || phase === 'empty' || phase === 'excluded'

  /**
   * Positive `seeded_dirs` from the CURRENT request, folded into the store
   * (issue #161, gate 2).
   *
   * Guarded by #156's own key and generation rather than by a second prediction
   * cache: the evidence is only ingested from a settled response, and it is
   * recorded against the query THAT RESPONSE speaks for, not against whatever
   * is on screen by the time it lands. The store ignores empty lists and rows
   * seeded by another query, so a stale, failed or empty ranking can never
   * erase a saved identity, its historical groups or their membership.
   */
  const observedQueryId = settled && key !== null ? key.queryId : null
  useEffect(() => {
    if (observedQueryId === null) return
    savedDirectories.observeSeededDirs(observedQueryId, seededDirs)
  }, [savedDirectories, observedQueryId, generation, seededDirs])

  // One directory per seed document, enforced by the backend with a 409. A
  // creation this reviewer made counts the moment the server confirms it, even
  // if the refetch that would have reported it never succeeds.
  const alreadySeeded = seededDirs.length > 0 || identity.status === 'saved'

  // The band of the *best model hit* decides the whole list's framing (issue
  // #94): that is the number a reviewer reads first, and the one that says
  // whether this fragment has a plausible home in the labelled corpus at all.
  // Reviewer directories are deliberately not part of that judgement — one
  // colleague's new directory scoring 0.6 does not mean the corpus has a home
  // for this fragment.
  const topScore = modelPredictions.length > 0 ? modelPredictions[0].score : null
  const topBand = topScore != null ? getConfidenceBand(topScore, bands) : null

  // Arrow keys walk the ranks that actually exist rather than counting by one.
  // Reviewer directories are anchored at rank 11 whatever the model returned,
  // so the sequence can jump (…, 10, 11) or have a gap (2, 11), and stepping
  // numerically would land on a rank with no card behind it.
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

  // The CTA renders from exactly ONE of two places: inside the no-match
  // callout, or at the foot of the list. Both read the same store record, so a
  // second mount would be a second copy of one acknowledgement rather than a
  // second decision -- hence the explicit exclusion below.
  //
  // The offer to create waits for a settled ranking. Offering "start a new
  // directory" while the request is still in flight, or after it failed, invites
  // the reviewer to declare a new source on the strength of evidence nobody has
  // seen -- and it is the wrong CTA in the first place if the ranking turns out
  // to be a no-match, where the emphasised one inside the callout is the right
  // one. Excluded and empty rankings still get it: the model said nothing about
  // the document, and "this is a new source" is a judgement the reviewer can
  // still reach on their own reading.
  //
  // ANYTHING ALREADY KNOWN ABOUT THIS QUERY is shown in every phase, because
  // none of it is a claim about the ranking: a confirmed save, a half-typed
  // name, a write whose outcome is still open, and a lookup that could not be
  // answered all outlive loading, errors and empty responses. That is the whole
  // point of the durable record, and it is what stops a failed refresh from
  // re-offering Create for a directory that already exists.
  const hasDurableState =
    identity.status === 'saved' ||
    identity.status === 'unresolved' ||
    creation.status !== 'idle'
  const inlineCtaMounted =
    settled && topBand === 'no_match' && activeQueryId !== null && !alreadySeeded
  const showFootCta =
    activeQueryId !== null &&
    !inlineCtaMounted &&
    (hasDurableState || (settled && !alreadySeeded && topBand !== 'no_match'))
  // The saved acknowledgement is on screen, with its own status badge, group
  // list and creator attribution. `saved` never coincides with the inline CTA,
  // which only mounts when the document is not yet seeded.
  const acknowledgementShown = showFootCta && identity.status === 'saved'

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
        <AwaitingMatchBadge
          seededDirs={acknowledgementShown ? [] : seededDirs}
        />
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
          {/* Band treatment for the top hit. Below the no-match threshold the
              default top option is creating a new directory, not picking a
              rank, so the CTA renders above the ranked cards. */}
          {topBand === 'no_match' && activeQueryId != null && (
            <NoMatchCallout
              // Remount per query so no in-flight CTA state can survive a
              // navigation.
              key={activeQueryId}
              queryFileId={activeQueryId}
              topScore={topScore}
              // The number of model candidates the None pill is named after,
              // so the copy matches the control the reviewer can see. There is
              // no fallback any more: MatchPills draws real ranks only, and
              // this callout renders only when a top hit exists, so the count
              // is never zero here (issue #157).
              topK={modelPredictions.length}
              model={activeModel}
              filename={response?.filename}
              alreadySeeded={alreadySeeded}
            />
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
          {reviewerPredictions.length > 0 && (
            <div className="text-[10px] font-semibold uppercase tracking-wider text-indigo-500/80 dark:text-indigo-300/70 px-2 pt-2">
              Reviewer directories
            </div>
          )}
          {reviewerPredictions.map((pred) => (
            <ReviewerDirCard
              key={pred.rank}
              prediction={pred}
              isActive={
                !overrideCandidateDir && pred.rank === activePredictionRank
              }
              onClick={() => select(pred.rank)}
            />
          ))}
        </div>
      )}

      {/* Above the no-match band the CTA is a quiet escape hatch at the foot of
          the list; below it, NoMatchCallout has already offered it up top. It
          also renders here, in every phase, once anything durable is known
          about this query -- an acknowledgement, an open naming form, a write
          whose outcome is still unknown, or a lookup that failed. None of that
          may be a casualty of a refetch. */}
      {showFootCta && activeQueryId !== null && (
        <NewDirectoryCta
          key={activeQueryId}
          queryId={activeQueryId}
          model={activeModel}
          emphasised={false}
          filename={response?.filename}
        />
      )}
    </div>
  )
}
