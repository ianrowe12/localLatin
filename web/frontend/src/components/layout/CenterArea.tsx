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
    overrideCandidateDir,
    setOverrideCandidateDir,
  } = useApp()

  // Fetch query detail
  const queryDetail = useQueryDetail(activeQueryId)

  // Fetch predictions
  const predictions = usePredictions(activeQueryId, activeModel)

  // Derive current prediction
  const currentPrediction = useMemo(() => {
    if (!predictions.data?.predictions) return null
    return predictions.data.predictions[activePredictionRank - 1] ?? null
  }, [predictions.data, activePredictionRank])

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

  const {
    applyAutoHighlights,
    selectedMethod,
    selectedVariant,
    setAvailableMethods,
  } = useTokens()

  const tokenMapResult = useTokenMap(
    activeQueryId,
    candidateDir,
    activeModel || undefined,
  )

  // Keep the AttributionMethodSelector's available list in sync with whatever
  // pair is currently loaded.
  const availableMethodsKey = tokenMapResult.data?.example_id
  useEffect(() => {
    setAvailableMethods(tokenMapResult.data?.available_methods ?? [])
  }, [availableMethodsKey, tokenMapResult.data, setAvailableMethods])

  // When an attribution method is selected and a matching matrix is present
  // in the token-map payload, swap similarity_matrix and recompute top_matches
  // so DocumentPanel highlights reflect the selection. Otherwise fall through
  // to the raw token map (or the word-match fallback if the map is absent).
  const effectiveTokenMap = useMemo(() => {
    const data = tokenMapResult.data
    if (!data) return wordMatchMap
    const selected =
      selectedMethod && data.pair_matrices
        ? data.pair_matrices[selectedMethod]?.[selectedVariant]
        : undefined
    if (!selected) return data
    const topMatches: Record<string, TopMatch[]> = {}
    for (let qi = 0; qi < selected.length; qi++) {
      const row = selected[qi]
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
      similarity_matrix: selected,
      top_matches: topMatches,
    }
    return swapped
  }, [tokenMapResult.data, selectedMethod, selectedVariant, wordMatchMap])

  // Auto-highlights: prefer the attribution-method-specific top_highlights
  // when available, otherwise fall back to the legacy auto_highlights payload.
  const autoHighlightKey = tokenMapResult.data?.example_id
  useEffect(() => {
    const data = tokenMapResult.data
    if (!data) return

    const selectedTop =
      selectedMethod && data.top_highlights
        ? data.top_highlights[selectedMethod]?.[selectedVariant]
        : undefined
    const selectedMatrix =
      selectedMethod && data.pair_matrices
        ? data.pair_matrices[selectedMethod]?.[selectedVariant]
        : undefined

    if (selectedTop && selectedMatrix) {
      const highlights = selectedTop.query.map((qi) => {
        const row = selectedMatrix[qi] ?? []
        const indexed = row
          .map((s, ci) => ({ candidate_idx: ci, score: s }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 2)
        return {
          queryIdx: qi,
          matches: indexed.map((m, rank) => ({
            candidateIdx: m.candidate_idx,
            score: m.score,
            rank: rank + 1,
          })),
        }
      })
      applyAutoHighlights(highlights)
      return
    }

    if (!data.auto_highlights) return
    const highlights = data.auto_highlights.map((ah) => ({
      queryIdx: ah.query_idx,
      matches: ah.matches.map((m, rank) => ({
        candidateIdx: m.candidate_idx,
        score: m.score,
        rank: rank + 1,
      })),
    }))
    applyAutoHighlights(highlights)
  }, [
    autoHighlightKey,
    tokenMapResult.data,
    selectedMethod,
    selectedVariant,
    applyAutoHighlights,
  ])

  const handleDrag = useCallback((newPercent: number) => {
    setSplitPercent(newPercent)
  }, [])

  return (
    <TokenRefProvider>
      <div
        ref={containerRef}
        className="relative flex-1 flex h-full overflow-hidden"
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
        <ConnectionOverlay
          containerRef={containerRef}
          leftPanelRef={queryScrollRef}
          rightPanelRef={candidateScrollRef}
        />
      </div>
    </TokenRefProvider>
  )
}
