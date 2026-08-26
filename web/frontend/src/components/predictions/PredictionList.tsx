import { useEffect, useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictions } from '../../api/queries'
import { bandsFrom } from '../../api/bands'
import { useModels } from '../../api/models'
import { useKeyboardShortcuts } from '../../utils/keyboard'
import AwaitingMatchBadge from './AwaitingMatchBadge'
import NewDirectoryCta from './NewDirectoryCta'
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
  const topScore = modelPredictions[0]?.score ?? 0
  const noLabelledMatch = modelPredictions.length === 0 || topScore < bands.no_match

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
          {modelPredictions.map((pred) => (
            <PredictionCard
              key={pred.rank}
              prediction={pred}
              rank={pred.rank}
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
      {activeQueryId !== null && !alreadySeeded && (
        <NewDirectoryCta
          queryId={activeQueryId}
          model={activeModel}
          emphasised={noLabelledMatch}
          filename={data?.filename}
        />
      )}
    </div>
  )
}
