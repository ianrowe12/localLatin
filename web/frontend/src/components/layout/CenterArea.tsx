import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import { useQueryDetail, usePredictions } from '../../api/queries'
import { TokenRefProvider } from '../connections/TokenRefRegistry'
import ConnectionOverlay from '../connections/ConnectionOverlay'
import DocumentPanel from '../document/DocumentPanel'
import DraggableDivider from './DraggableDivider'
import { buildWordMatchMap } from '../../utils/wordSimilarity'
import { useTokenMap } from '../../api/tokenMap'
import { useTokens, type PinMatch } from '../../contexts/TokenContext'

export default function CenterArea() {
  const [splitPercent, setSplitPercent] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const queryScrollRef = useRef<HTMLDivElement>(null)
  const candidateScrollRef = useRef<HTMLDivElement>(null)

  const { activeQueryId, activePredictionRank, activeModel } = useApp()

  // Fetch query detail
  const queryDetail = useQueryDetail(activeQueryId)

  // Fetch predictions
  const predictions = usePredictions(activeQueryId, activeModel)

  // Derive current prediction
  const currentPrediction = useMemo(() => {
    if (!predictions.data?.predictions) return null
    return predictions.data.predictions[activePredictionRank - 1] ?? null
  }, [predictions.data, activePredictionRank])

  // Derive candidate info
  const candidateDir = currentPrediction?.dir_name ?? null
  const candidateFile = currentPrediction?.candidate_files?.[0] ?? null

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

  const { applyAutoHighlights } = useTokens()

  const tokenMapResult = useTokenMap(
    activeQueryId,
    candidateDir,
    activeModel || undefined,
  )

  const effectiveTokenMap = tokenMapResult.data ?? wordMatchMap

  const autoHighlightKey = tokenMapResult.data?.example_id
  useEffect(() => {
    if (!tokenMapResult.data?.auto_highlights) return
    const highlights = tokenMapResult.data.auto_highlights.map((ah) => ({
      queryIdx: ah.query_idx,
      matches: ah.matches.map((m, rank) => ({
        candidateIdx: m.candidate_idx,
        score: m.score,
        rank: rank + 1,
      })),
    }))
    applyAutoHighlights(highlights)
  }, [autoHighlightKey, applyAutoHighlights])

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
                dirLabel={currentPrediction?.dir_name}
                score={currentPrediction?.score}
                rank={activePredictionRank}
                tokens={candidateTokens}
                tokenMap={effectiveTokenMap}
                loading={predictions.loading}
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
