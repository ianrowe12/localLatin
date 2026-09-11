import {
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from 'react'
import { useApp } from './AppContext'
import {
  candidateHasText,
  modelCandidates,
  predictionKeyString,
  reviewerCandidates,
  samePredictionKey,
  usePredictions,
  type ExclusionReason,
  type Prediction,
  type PredictionPhase,
  type PredictionRequestKey,
  type PredictionResponse,
} from '../api/queries'
import type { ApiErrorInfo } from '../api/client'
import type { ReviewerDir } from '../api/reviewerDirs'

/** The model contributes at most this many candidates; the API caps it too. */
const MAX_MODEL_CANDIDATES = 10

/**
 * A candidate, named in a way that survives a refresh (issue #156).
 *
 * A rank number alone does not identify a candidate over time: reviewer
 * directories are anchored at rank 11 and `web/services/reviewer_dirs.py` skips
 * directories the query already belongs to, so the directory behind rank 11 can
 * change between one response and the next. This string binds the rank to the
 * request it came from AND to the directory it named, so a consumer can tell
 * "the same rank" from "the same candidate".
 *
 * It is a client-side identity for the data on screen. It is not a server
 * ranking revision, and it is not evidence that a reviewer saw anything.
 */
export function candidateIdentity(
  key: PredictionRequestKey,
  generation: number,
  prediction: Prediction,
): string {
  return [
    predictionKeyString(key),
    `g${generation}`,
    `r${prediction.rank}`,
    prediction.source ?? 'model',
    prediction.dir_name,
  ].join('|')
}

export interface PredictionContextValue {
  /** The query/model/variant on screen, or null when nothing is selected. */
  key: PredictionRequestKey | null
  /** Client request id behind the exposed data; 0 while nothing has settled. */
  generation: number
  phase: PredictionPhase
  /** Validated response for `ready` / `empty` / `excluded`; null otherwise. */
  response: PredictionResponse | null
  error: ApiErrorInfo | null
  /** Raw retrieval status from the CSV row (`ok`, `excluded_*`), or null. */
  sourceStatus: string | null
  exclusion: ExclusionReason | null

  /** Every current candidate, model first, exactly as the API ordered them. */
  predictions: Prediction[]
  /** Model candidates only, capped at ten. */
  modelPredictions: Prediction[]
  /** Reviewer-created directories, uncapped by the model's ten. */
  reviewerPredictions: Prediction[]
  seededDirs: ReviewerDir[]

  /** True only in `ready`: the model produced at least one candidate. */
  hasUsableModelRanking: boolean
  isLoading: boolean
  /** No query or no model chosen yet -- not a failure. */
  isMissingSelection: boolean

  /** Current candidate at `rank`, or null. Never index arithmetic. */
  getByRank: (rank: number) => Prediction | null
  /** Identity of a candidate within the current request. */
  identityOf: (prediction: Prediction) => string | null
  /** Whether this candidate carries readable text. Missing text stays missing. */
  hasText: (prediction: Prediction) => boolean
  /** Refetch. Current evidence becomes unavailable until the new request settles. */
  refresh: () => void
}

const PredictionContext = createContext<PredictionContextValue | null>(null)

/**
 * One authoritative ranking for the current query/model/variant (issue #156).
 *
 * Before this, `PredictionList`, `CenterArea` and `FeedbackPanel` each called
 * `usePredictions` independently. That is three requests, three caches and
 * three answers to the same question: the list could render a failed request as
 * "No predictions available" while the assessment panel offered rank pills from
 * a response that had succeeded, for a model the reviewer had already left.
 *
 * The provider owns the single request. Every consumer reads the same key,
 * generation and phase, so they cannot disagree, and a changed selection makes
 * the old data unavailable to all three in the same render rather than one
 * commit later.
 *
 * Deliberately not a state library: this is one keyed request with a generation
 * counter, and the existing hook/provider pattern carries it.
 */
export function PredictionProvider({ children }: { children: ReactNode }) {
  const { activeQueryId, activeModel, activeVariant } = useApp()
  const result = usePredictions(activeQueryId, activeModel, activeVariant)

  const value = useMemo<PredictionContextValue>(() => {
    const all = result.response?.predictions ?? []
    // The two groups are sliced separately: the model's cap is ten, and
    // reviewer directories are appended after it rather than competing for the
    // same ten places.
    const model = modelCandidates(all).slice(0, MAX_MODEL_CANDIDATES)
    const reviewer = reviewerCandidates(all)
    const key = result.key

    return {
      key,
      generation: result.generation,
      phase: result.phase,
      response: result.response,
      error: result.error,
      sourceStatus: result.sourceStatus,
      exclusion: result.exclusion,
      predictions: all,
      modelPredictions: model,
      reviewerPredictions: reviewer,
      seededDirs: result.response?.seeded_dirs ?? [],
      // `ready` already means "at least one model candidate". A response
      // carrying only reviewer directories is not a model ranking, so it can
      // never make the model's answer look evaluable.
      hasUsableModelRanking: result.phase === 'ready',
      isLoading: result.phase === 'loading',
      isMissingSelection: result.phase === 'idle',
      getByRank: (rank: number) =>
        all.find((prediction) => prediction.rank === rank) ?? null,
      identityOf: (prediction: Prediction) =>
        key === null ? null : candidateIdentity(key, result.generation, prediction),
      hasText: candidateHasText,
      refresh: result.refresh,
    }
  }, [result])

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  )
}

export function usePredictionState(): PredictionContextValue {
  const ctx = useContext(PredictionContext)
  if (!ctx) {
    throw new Error('usePredictionState must be used within a PredictionProvider')
  }
  return ctx
}

/**
 * The same shared state, for a reader that must work without this provider.
 *
 * This is NOT a second source of rankings: it reads the one provider above, or
 * reports that there is none. The live evidence stamp (issue #163) needs the
 * authoritative key, generation and phase from inside `useTokens`, which is
 * also called by components mounted alone in their own tests. Those callers get
 * null and are not gated, rather than throwing.
 */
export function useOptionalPredictionState(): PredictionContextValue | null {
  return useContext(PredictionContext)
}

/**
 * Whether a key still names what is on screen.
 *
 * For consumers holding a key captured earlier (an in-flight save, a pending
 * directory creation): a result is only current if its key still matches.
 */
export function isCurrentPredictionKey(
  value: PredictionContextValue,
  key: PredictionRequestKey | null,
): boolean {
  return samePredictionKey(value.key, key)
}
