import { useEffect, useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictions } from '../../api/queries'
import { useModels } from '../../api/models'
import { useKeyboardShortcuts } from '../../utils/keyboard'
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

export default function PredictionList() {
  const {
    activeQueryId,
    activeModel,
    activeVariant,
    activePredictionRank,
    setActivePredictionRank,
    overrideCandidateDir,
    setOverrideCandidateDir,
  } = useApp()
  const { data, loading } = usePredictions(activeQueryId, activeModel, activeVariant)

  // If a gallery override candidate happens to be in the current model's
  // top-10, sync the prediction list selection and release the override so
  // the normal flow takes over. Otherwise leave the override intact —
  // CenterArea uses it directly to render an off-list candidate.
  useEffect(() => {
    if (!overrideCandidateDir || !data?.predictions) return
    const match = data.predictions.find((p) => p.dir_name === overrideCandidateDir)
    if (match) {
      setActivePredictionRank(match.rank)
      setOverrideCandidateDir(null)
    }
  }, [overrideCandidateDir, data, setActivePredictionRank, setOverrideCandidateDir])

  // Thresholds come from the deployment (GET /api/models), not from a literal
  // in this bundle: the backend decides reviewer-directory status with the same
  // numbers, so there is one source of truth and it is server-side.
  const { data: models } = useModels()
  const bands = bandsFrom(models)

  // The model contributes at most ten candidates; reviewer-created directories
  // are appended after them by the API and are not subject to that cap, so the
  // two groups are sliced separately rather than truncating the whole list.
  const all = data?.predictions ?? []
  const modelPredictions = all
    .filter((p) => (p.source ?? 'model') === 'model')
    .slice(0, 10)
  const reviewerPredictions = all.filter((p) => p.source === 'reviewer')
  const predictions = [...modelPredictions, ...reviewerPredictions]

  const seededDirs = data?.seeded_dirs ?? []
  // One directory per seed document, enforced by the backend with a 409.
  const alreadySeeded = seededDirs.length > 0

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

  if (loading) {
    return (
      <div data-tour="predictions" className="px-2 py-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1 px-2">
          Predicted Sources
        </div>
        <p className="font-ui text-xs text-stone-400 dark:text-stone-500 px-2 mb-2">
          Ranked by similarity
        </p>
        <p className="text-xs text-stone-400 text-center mt-4 font-ui">Loading...</p>
      </div>
    )
  }

  const select = (rank: number) => {
    setActivePredictionRank(rank)
    setOverrideCandidateDir(null)
  }

  return (
    <div data-tour="predictions" className="px-2 py-2">
      <div className="flex items-center gap-2 mb-1 px-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-stone-400">
          Predicted Sources
        </span>
        <AwaitingMatchBadge seededDirs={seededDirs} />
      </div>
      <p className="font-ui text-xs text-stone-400 dark:text-stone-500 px-2 mb-2">
        Ranked by similarity
      </p>
      {predictions.length === 0 ? (
        <p className="text-xs text-stone-400 dark:text-stone-500 text-center mt-4 font-ui">
          {activeQueryId === null
            ? 'Select a query to see predictions'
            : 'No predictions available'}
        </p>
      ) : (
        <div className="flex flex-col gap-1">
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
              // Same fallback as MatchPills, so the copy names the pill the
              // reviewer can actually see.
              topK={modelPredictions.length || 10}
              model={activeModel}
              filename={data?.filename}
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
          the list; below it, NoMatchCallout has already offered it up top. */}
      {activeQueryId !== null && !alreadySeeded && topBand !== 'no_match' && (
        <NewDirectoryCta
          key={activeQueryId}
          queryId={activeQueryId}
          model={activeModel}
          emphasised={false}
          filename={data?.filename}
        />
      )}
    </div>
  )
}
