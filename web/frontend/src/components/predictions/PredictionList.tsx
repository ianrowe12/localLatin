import { useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictions } from '../../api/queries'
import { useKeyboardShortcuts } from '../../utils/keyboard'
import PredictionCard from './PredictionCard'

export default function PredictionList() {
  const { activeQueryId, activeModel, activePredictionRank, setActivePredictionRank } = useApp()
  const { data, loading } = usePredictions(activeQueryId, activeModel)

  const predictions = data?.predictions?.slice(0, 10) ?? []
  const maxRank = predictions.length

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

  return (
    <div data-tour="predictions" className="px-2 py-2">
      <div className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1 px-2">
        Predicted Sources
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
          {predictions.map((pred) => (
            <PredictionCard
              key={pred.rank}
              prediction={pred}
              rank={pred.rank}
              isActive={pred.rank === activePredictionRank}
              onClick={() => setActivePredictionRank(pred.rank)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
