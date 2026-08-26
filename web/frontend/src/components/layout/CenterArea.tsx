import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import {
  useQueryDetail,
  usePredictions,
  useCandidateDirFiles,
} from '../../api/queries'
import { TokenRefProvider } from '../connections/TokenRefRegistry'
import ConnectionOverlay from '../connections/ConnectionOverlay'
import DocumentPanel from '../document/DocumentPanel'
import AwaitingMatchBadge from '../predictions/AwaitingMatchBadge'
import DraggableDivider from './DraggableDivider'
import { buildWordMatchMap } from '../../utils/wordSimilarity'
import { useTokenMap, type TokenMapResponse, type TopMatch } from '../../api/tokenMap'
import { toAttributionVariant } from '../../api/variants'
import { useTokens } from '../../contexts/TokenContext'
import { METHODS } from '../common/AttributionMethodSelector'

export default function CenterArea() {
  const [splitPercent, setSplitPercent] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const queryScrollRef = useRef<HTMLDivElement>(null)
  const candidateScrollRef = useRef<HTMLDivElement>(null)

  const {
    activeQueryId,
    activePredictionRank,
    activeModel,
    activeVariant,
    overrideCandidateDir,
    setOverrideCandidateDir,
  } = useApp()

  // Fetch query detail
  const queryDetail = useQueryDetail(activeQueryId)

  // Fetch predictions
  const predictions = usePredictions(activeQueryId, activeModel, activeVariant)

  // Derive current prediction.
  //
  // The key check is load-bearing, not defensive. Selecting a model updates
  // activeModel during render; usePredictions only clears its data from an
  // effect, one commit later. Without the guard, that in-between render pairs
  // the OLD selection's rank-1 candidate with the NEW one and fires a
  // multi-MB token-map fetch for a (candidate, model) combination the
  // reviewer never selected -- which then paints as the "current" evidence
  // (issue #73). PredictionResponse echoes both `model` and `variant` from
  // the request exactly (web/routers/predictions.py resolves them or raises),
  // so comparing them is an exact key check.
  //
  // Since issue #94 the model is the half a reviewer can actually move: there
  // is one pipeline, so `variant` can no longer diverge. It is still compared
  // because the field is still on the wire and the day a second pipeline is
  // served again, this guard should already be right.
  const currentPrediction = useMemo(() => {
    if (!predictions.data?.predictions) return null
    if (predictions.data.variant !== activeVariant) return null
    if (predictions.data.model !== activeModel) return null
    // Looked up BY RANK, not by array index. Reviewer directories are
    // anchored at rank 11 regardless of how many model candidates came back,
    // so the list can have a gap in it and index arithmetic would pair a rank
    // with the wrong card.
    return (
      predictions.data.predictions.find((p) => p.rank === activePredictionRank) ??
      null
    )
  }, [predictions.data, activePredictionRank, activeVariant, activeModel])

  // Derive candidate info — override wins over the normal prediction path
  const candidateDir = overrideCandidateDir ?? currentPrediction?.dir_name ?? null

  // For override-mode candidates, fetch the directory's files on demand.
  // (Regular predictions already carry candidate_files in their payload.)
  const overrideCandidateFiles = useCandidateDirFiles(overrideCandidateDir)
  const candidateFile = useMemo(() => {
    if (overrideCandidateDir) {
      return overrideCandidateFiles.data?.[0] ?? null
    }
    return currentPrediction?.candidate_files?.[0] ?? null
  }, [overrideCandidateDir, overrideCandidateFiles.data, currentPrediction])

  // Candidate tokens: simple tokenization from candidate file text
  const candidateTokens = useMemo(() => {
    if (candidateFile?.text) {
      return candidateFile.text
        .split(/\s+/)
        .filter((t) => t.length > 0)
        .map((t, i) => ({
          text: t,
          index: i,
          category: /^[.,;:!?()\[\]]+$/.test(t) ? 'punctuation' : 'content',
        }))
    }
    return undefined
  }, [candidateFile])

  // Word-match similarity for cross-panel highlighting
  const wordMatchMap = useMemo(() => {
    if (!queryDetail.data?.tokens || !candidateTokens) return null
    return buildWordMatchMap(queryDetail.data.tokens, candidateTokens)
  }, [queryDetail.data?.tokens, candidateTokens])

  const { selectedMethod, viewMode, setAvailableMethods, clearAllPins } = useTokens()

  // The highlights are computed from the same pipeline as the ranking, so the
  // evidence a reviewer reads always belongs to the ranking they are judging.
  // The artifacts call the uncorrected variant "baseline" where the prediction
  // CSVs call it "raw" -- toAttributionVariant is the one place that bridges
  // the two vocabularies.
  const attributionVariant = toAttributionVariant(activeVariant)
  // The reviewer-facing method name, not the artifact slug: a PI-admin on
  // attention_weighted should read "Attn-W", the label the method picker uses.
  const methodLabel =
    METHODS.find((m) => m.key === selectedMethod)?.label ?? selectedMethod

  const tokenMapResult = useTokenMap(
    activeQueryId,
    candidateDir,
    activeModel || undefined,
    selectedMethod,
    attributionVariant,
  )

  // Keep the AttributionMethodSelector's available list in sync with whatever
  // pair is currently loaded. Also clear any leftover pins from a previous
  // session/pair so each new pair starts with hover-only behavior (no sticky
  // lines from stale pinnedTokens state).
  const availableMethodsKey = tokenMapResult.data?.example_id
  useEffect(() => {
    setAvailableMethods(tokenMapResult.data?.available_methods ?? [])
    clearAllPins()
  }, [availableMethodsKey, tokenMapResult.data, setAvailableMethods, clearAllPins])

  // When an attribution method is selected and a matching matrix is present
  // in the token-map payload, swap similarity_matrix and recompute top_matches
  // so DocumentPanel highlights reflect the selection. Otherwise fall through
  // to the raw token map (or the word-match fallback if the map is absent).
  //
  // Different attribution methods have wildly different value ranges (IG ~±0.03,
  // BERTScore [0,1], OT ~[0,0.07], attention ~[0,0.015], DLA baseline ~[0,0.9]),
  // so we take |value| and divide by the matrix max to put every method on a
  // common [0,1] scale. DocumentPanel's existing thresholding expects this.
  const selectedMatrix = useMemo(() => {
    const data = tokenMapResult.data
    if (!data || !selectedMethod || !data.pair_matrices) return undefined
    return data.pair_matrices[selectedMethod]?.[attributionVariant]
  }, [tokenMapResult.data, selectedMethod, attributionVariant])

  // The payload's own `similarity_matrix` is plain cosine over the raw hidden
  // states: identical for every attribution method and every post-processing
  // variant. Falling through to it when the selected cell of `pair_matrices`
  // is absent is what made the highlights look frozen across a switch (issue
  // #73) -- the view had silently stopped showing attribution at all. Since
  // issue #94 removed the variant picker, the switch a PI-admin can still make
  // is the method one, and the failure mode is identical. Say so instead of
  // showing a grid that cannot answer the question just asked.
  //
  // `available_methods` is the test, NOT the shape of `pair_matrices`. Both
  // are declared with a default factory (web/models.py:117,121), so the API
  // always serialises both keys -- the original `pair_matrices != null` was
  // true for every real response and false only under `npm run dev:mock`.
  // Emptiness does not separate the two cases either: `load_token_map` builds
  // `pair_matrices` with `setdefault(m, {})[v]` only when a matrix is actually
  // added, so an artifact that carries attribution but not for the requested
  // variant emits exactly the same `{}` as one that carries none at all.
  // `available_methods` describes the whole artifact regardless of the
  // ?method=/?variant= filter, which is precisely the distinction wanted:
  // empty means the artifact predates attribution, and for those the cosine
  // map is the honest whole story, so they keep the old behaviour.
  const artifactHasAttribution =
    (tokenMapResult.data?.available_methods?.length ?? 0) > 0

  const attributionUnavailable =
    tokenMapResult.data != null &&
    artifactHasAttribution &&
    selectedMethod != null &&
    selectedMatrix === undefined

  // On current main the notice above is defence in depth: #90's resolver keys
  // artifacts per layer and filters on `variants_available`, so a pair that
  // resolves at all carries the variant it resolved for. What a reviewer does
  // hit is the other branch -- `resolve_example_id` returning None, i.e. a 404
  // -- which until now rendered an unexplained empty evidence panel. Both
  // states mean the same thing to the reader, so they share one message.
  const attributionMissing = tokenMapResult.error != null

  const effectiveTokenMap = useMemo(() => {
    const data = tokenMapResult.data
    if (!data) return wordMatchMap
    if (!selectedMatrix) return attributionUnavailable ? null : data
    const selected = selectedMatrix

    // Per-pair |max| over all cells; abs+normalize to [0,1].
    let absMax = 0
    for (let qi = 0; qi < selected.length; qi++) {
      const row = selected[qi]
      if (!row) continue
      for (let ci = 0; ci < row.length; ci++) {
        const v = Math.abs(row[ci])
        if (v > absMax) absMax = v
      }
    }
    const denom = absMax > 1e-12 ? absMax : 1
    const normalized: number[][] = selected.map((row) =>
      row ? row.map((s) => Math.abs(s) / denom) : [],
    )

    const topMatches: Record<string, TopMatch[]> = {}
    for (let qi = 0; qi < normalized.length; qi++) {
      const row = normalized[qi]
      if (!row) continue
      const indexed: TopMatch[] = row.map((s, ci) => ({
        candidate_idx: ci,
        score: s,
      }))
      indexed.sort((a, b) => b.score - a.score)
      topMatches[String(qi)] = indexed.slice(0, 3)
    }
    const swapped: TokenMapResponse = {
      ...data,
      similarity_matrix: normalized,
      top_matches: topMatches,
    }
    return swapped
  }, [tokenMapResult.data, selectedMatrix, attributionUnavailable, wordMatchMap])

  // Note: we deliberately do NOT auto-pin top-attribution tokens on pair entry.
  // Connection lines are drawn purely on hover (see useConnectionState). Token
  // background shading already encodes the selected attribution matrix via
  // DocumentPanel's row/col-max derivation from the swapped similarity_matrix.

  const handleDrag = useCallback((newPercent: number) => {
    setSplitPercent(newPercent)
  }, [])

  return (
    <TokenRefProvider>
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {(attributionUnavailable || attributionMissing) && (
          <div
            role="status"
            className="px-3 py-1.5 bg-stone-100 dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 text-xs text-stone-600 dark:text-stone-300 flex-shrink-0"
          >
            {attributionUnavailable ? (
              <>
                No {methodLabel} attribution for this pair. Highlights are off
                rather than falling back to a plain similarity grid, which
                would look the same for every method.
              </>
            ) : (
              <>
                No attribution available for this pair. Highlights are off; the
                ranking above is unaffected.
              </>
            )}
          </div>
        )}
        <div
          ref={containerRef}
          className="relative flex-1 flex overflow-hidden"
        >
          {/* Query panel */}
          <div
            data-tour="query-panel"
            style={{ width: `${splitPercent}%` }}
            className="h-full overflow-hidden flex flex-col"
          >
            <DocumentPanel
              side="query"
              filename={queryDetail.data?.filename}
              tokens={queryDetail.data?.tokens}
              tokenMap={effectiveTokenMap}
              loading={queryDetail.loading}
              scrollRef={queryScrollRef}
              // The badge reports the fate of directories *this* document
              // seeded, so it belongs on the query panel, not on a candidate.
              badge={
                <AwaitingMatchBadge
                  seededDirs={predictions.data?.seeded_dirs ?? []}
                />
              }
            />
          </div>

          <DraggableDivider onDrag={handleDrag} />

          {/* Candidate panel */}
          <div
            data-tour="candidate-panel"
            style={{ width: `${100 - splitPercent}%` }}
            className="h-full overflow-hidden flex flex-col"
          >
            {overrideCandidateDir && (
              <div className="px-3 py-1.5 bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800/40 text-xs text-amber-800 dark:text-amber-200 flex items-center justify-between flex-shrink-0">
                <span>Viewing example pair · candidate may be outside top-10</span>
                <button
                  type="button"
                  onClick={() => setOverrideCandidateDir(null)}
                  className="px-2 py-0.5 rounded hover:bg-amber-100 dark:hover:bg-amber-900/40 font-medium"
                >
                  Exit
                </button>
              </div>
            )}
            <AnimatePresence mode="wait">
              <motion.div
                key={candidateDir ?? 'empty'}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="h-full flex flex-col"
              >
                <DocumentPanel
                  side="candidate"
                  filename={candidateFile?.filename}
                  dirLabel={
                    // Reviewer directories show their human label; the opaque
                    // reviewer-dir-N id would tell the reviewer nothing.
                    overrideCandidateDir ??
                    currentPrediction?.label ??
                    currentPrediction?.dir_name
                  }
                  score={overrideCandidateDir ? undefined : currentPrediction?.score}
                  rank={overrideCandidateDir ? undefined : activePredictionRank}
                  tokens={candidateTokens}
                  tokenMap={effectiveTokenMap}
                  loading={
                    overrideCandidateDir
                      ? overrideCandidateFiles.loading
                      : predictions.loading
                  }
                  scrollRef={candidateScrollRef}
                />
              </motion.div>
            </AnimatePresence>
          </div>

          {/* SVG connection overlay */}
          {viewMode !== 'heatmap' && (
            <ConnectionOverlay
              containerRef={containerRef}
              leftPanelRef={queryScrollRef}
              rightPanelRef={candidateScrollRef}
            />
          )}
        </div>
      </div>
    </TokenRefProvider>
  )
}
