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
  const maxRank = predictions.length > 0 ? predictions[predictions.length - 1].rank : 0

  const seededDirs = data?.seeded_dirs ?? []
  const topScore = modelPredictions[0]?.score ?? 0
  const noLabelledMatch = modelPredictions.length === 0 || topScore < bands.no_match

  const handlers = useMemo(
    () => ({
      ArrowUp: (e: KeyboardEvent) => {
        e.preventDefault()
        setActivePredictionRank(Math.max(1, activePredictionRank - 1))
      },
      ArrowDown: (e: KeyboardEvent) => {
        e.preventDefault()
        if (maxRank > 0) {
          setActivePredictionRank(Math.min(maxRank, activePredictionRank + 1))
        }
      },
      ...Object.fromEntries(
        Array.from({ length: 9 }, (_, i) => [
          String(i + 1),
          () => {
            if (i + 1 <= maxRank) {
              setActivePredictionRank(i + 1)
            }
          },
        ]),
      ),
    }),
    [activePredictionRank, maxRank, setActivePredictionRank],
  )

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
      {activeQueryId !== null && (
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
