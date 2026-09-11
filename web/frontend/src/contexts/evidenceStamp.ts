import { predictionKeyString } from '../api/queries'
import type { PredictionPhase, PredictionRequestKey } from '../api/queries'
import { useOptionalApp } from './AppContext'
import { useOptionalPredictionState } from './PredictionContext'

/**
 * Everything that decides WHICH evidence the app is currently entitled to show.
 *
 * Deliberately not the witness, the rank or the member: those move within one
 * settled view, and the pair identity (`displayedPair`) already tells them
 * apart. This is the coarser question that pair identity cannot answer -- "is
 * the request behind that pair still the request this app is serving?" -- and
 * the two stored strings a publication leaves behind can both describe a
 * superseded request while agreeing perfectly with each other.
 */
export interface EvidenceStampInput {
  queryId: number | null
  model: string
  variant: string
  overrideDir: string | null
  predictionKey: PredictionRequestKey | null
  generation: number
  phase: PredictionPhase
}

/**
 * A string naming the live evidence regime (issue #163).
 *
 * Two regimes, because the two routes into the candidate panel are entitled to
 * different evidence:
 *
 * - A RANKED candidate exists only because the shared ranking says so, so its
 *   stamp carries that request's key, generation and phase. A refresh moves all
 *   three before any new candidate has arrived, which is exactly the window in
 *   which the previous pair's method controls must stop being offered.
 * - A GALLERY inspection carries its own candidate and fetches its own files,
 *   so a ranking refresh underneath it changes nothing it is showing. Binding
 *   its stamp to the ranking's generation would blank controls that are still
 *   describing exactly the pair on screen.
 *
 * Both carry the document, the model and the pipeline, because a token-map
 * artifact is about all three.
 */
export function evidenceStamp(input: EvidenceStampInput): string {
  const view = `${input.queryId ?? 'no-query'}|${input.model}|${input.variant}`
  if (input.overrideDir !== null) {
    return `override|${view}|${input.overrideDir}`
  }
  const key =
    input.predictionKey === null ? 'no-key' : predictionKeyString(input.predictionKey)
  return `ranked|${view}|${key}|g${input.generation}|${input.phase}`
}

/**
 * The live stamp, read from the authoritative shared state.
 *
 * It reuses `usePredictionState`'s provider rather than issuing a second
 * `usePredictions`: a private cache would be a second answer to the question
 * the provider exists to answer once (issue #156), and a gate built on it could
 * disagree with the ranking on screen.
 *
 * Null when either provider is absent, which means "nothing live to check
 * against" and gates nothing. That keeps components that are legitimately
 * mounted alone in their own tests working unchanged.
 */
export function useLiveEvidenceStamp(): string | null {
  const app = useOptionalApp()
  const predictions = useOptionalPredictionState()
  if (app === null || predictions === null) return null
  return evidenceStamp({
    queryId: app.activeQueryId,
    model: app.activeModel,
    variant: app.activeVariant,
    overrideDir: app.overrideCandidateDir,
    predictionKey: predictions.key,
    generation: predictions.generation,
    phase: predictions.phase,
  })
}
