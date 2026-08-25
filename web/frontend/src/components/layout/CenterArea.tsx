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
import DraggableDivider from './DraggableDivider'
import { buildWordMatchMap } from '../../utils/wordSimilarity'
import { useTokenMap, type TokenMapResponse, type TopMatch } from '../../api/tokenMap'
import { toAttributionVariant, VARIANT_OPTIONS } from '../../api/variants'
import { useTokens } from '../../contexts/TokenContext'

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
  // The variant check is load-bearing, not defensive. `setActiveVariant`
  // updates activeVariant during render; usePredictions only clears its data
  // from an effect, one commit later. Without the guard, that in-between
  // render pairs the OLD variant's rank-1 candidate with the NEW variant and
  // fires a multi-MB token-map fetch for a (candidate, variant) combination
  // the reviewer never selected -- which then paints as the "current"
  // evidence (issue #73). PredictionResponse.variant echoes the request
  // exactly (web/routers/predictions.py resolves it or raises), so comparing
  // it is an exact key check.
  const currentPrediction = useMemo(() => {
    if (!predictions.data?.predictions) return null
    if (predictions.data.variant !== activeVariant) return null
    return predictions.data.predictions[activePredictionRank - 1] ?? null
  }, [predictions.data, activePredictionRank, activeVariant])

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

  // The reviewer's post-processing choice drives the highlights too, so the
  // evidence they read always belongs to the ranking they are judging. The
  // artifacts call the uncorrected variant "baseline" where the prediction
  // CSVs call it "raw" -- toAttributionVariant is the one place that bridges
  // the two vocabularies.
  const attributionVariant = toAttributionVariant(activeVariant)
  const variantLabel =
    VARIANT_OPTIONS.find((o) => o.key === activeVariant)?.label ?? activeVariant

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
  // states: identical for every post-processing variant. Falling through to it
  // when the selected variant's matrix is absent is what made the highlights
  // look frozen across a variant switch (issue #73) -- the view had silently
  // stopped showing attribution at all. Say so instead of showing a grid that
  // cannot answer the question the reviewer just asked. Artifacts with no
  // `pair_matrices` at all predate attribution entirely, and for those the
  // cosine map is the honest whole story, so they keep the old behaviour.
  const attributionUnavailable =
    tokenMapResult.data != null &&
    tokenMapResult.data.pair_matrices != null &&
    selectedMethod != null &&
    selectedMatrix === undefined

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
        {attributionUnavailable && (
          <div
            role="status"
            className="px-3 py-1.5 bg-stone-100 dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 text-xs text-stone-600 dark:text-stone-300 flex-shrink-0"
          >
            No {selectedMethod} attribution for the {variantLabel} variant on
            this pair. Highlights are off rather than showing another
            variant&apos;s.
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
                    overrideCandidateDir ?? currentPrediction?.dir_name
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
