import { useEffect, useMemo } from 'react'
import { useApp } from '../../contexts/AppContext'
import { usePredictions } from '../../api/queries'
import { useKeyboardShortcuts } from '../../utils/keyboard'
import {
  BAND_COPY,
  BAND_STYLES,
  getConfidenceBand,
} from '../../utils/confidenceBands'
import NoMatchCallout from './NoMatchCallout'
import PredictionCard from './PredictionCard'

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

  const predictions = data?.predictions?.slice(0, 10) ?? []
  const maxRank = predictions.length
  // The band of the *best* hit decides the whole list's framing (issue #94):
  // that is the number a reviewer reads first, and the one that says whether
  // this fragment has a plausible home in the corpus at all.
  const topScore = predictions.length > 0 ? predictions[0].score : null
  const topBand = topScore != null ? getConfidenceBand(topScore) : null

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
          {/* Band treatment for the top hit. Below the no-match threshold the
              default top option is creating a new directory, not picking a
              rank, so the CTA renders above the ranked cards. */}
          {topBand === 'no_match' && activeQueryId != null && (
            <NoMatchCallout
              // Remount per query so no in-flight CTA state can survive a
              // navigation; the component guards its own async result too.
              key={activeQueryId}
              queryFileId={activeQueryId}
              topScore={topScore}
              // Same fallback as MatchPills, so the copy names the pill the
              // reviewer can actually see.
              topK={maxRank || 10}
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
          {predictions.map((pred) => (
            <PredictionCard
              key={pred.rank}
              prediction={pred}
              rank={pred.rank}
              isActive={
                !overrideCandidateDir && pred.rank === activePredictionRank
              }
              onClick={() => {
                setActivePredictionRank(pred.rank)
                setOverrideCandidateDir(null)
              }}
            />
          ))}
        </div>
      )}
    </div>
  )
}
