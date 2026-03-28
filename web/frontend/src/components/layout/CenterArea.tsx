import { useCallback, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import { useQueryDetail, usePredictions } from '../../api/queries'
import { useTokenMap } from '../../api/tokenMap'
import { TokenRefProvider } from '../connections/TokenRefRegistry'
import ConnectionOverlay from '../connections/ConnectionOverlay'
import DocumentPanel from '../document/DocumentPanel'
import DraggableDivider from './DraggableDivider'

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

  // Candidate tokens: use tokenMap's candidate_tokens when available,
  // since they carry proper token indices
  const tokenMap = useTokenMap(activeQueryId, candidateDir)

  const candidateTokens = useMemo(() => {
    if (tokenMap.data?.candidate_tokens) {
      return tokenMap.data.candidate_tokens.map((t) => ({
        text: t.text,
        index: t.idx,
        category: t.is_content ? 'content' : 'punctuation',
      }))
    }
    return undefined
  }, [tokenMap.data])

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
          style={{ width: `${splitPercent}%` }}
          className="h-full overflow-hidden flex flex-col"
        >
          <DocumentPanel
            side="query"
            filename={queryDetail.data?.filename}
            tokens={queryDetail.data?.tokens}
            tokenMap={tokenMap.data}
            loading={queryDetail.loading}
            scrollRef={queryScrollRef}
          />
        </div>

        <DraggableDivider onDrag={handleDrag} />

        {/* Candidate panel */}
        <div
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
                tokens={candidateTokens}
                tokenMap={tokenMap.data}
                loading={tokenMap.loading || predictions.loading}
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
