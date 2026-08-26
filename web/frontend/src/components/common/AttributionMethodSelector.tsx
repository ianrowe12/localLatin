import { motion } from 'framer-motion'
import { useTokens } from '../../contexts/TokenContext'
import type { AttributionMethod } from '../../api/tokenMap'

const METHODS: { key: AttributionMethod; label: string; title: string }[] = [
  { key: 'ig', label: 'IG', title: 'Integrated gradients attribution' },
  { key: 'bertscore', label: 'BERTScore', title: 'Token similarity score' },
  { key: 'ot', label: 'OT', title: 'Optimal transport alignment' },
  { key: 'attention_weighted', label: 'Attn-W', title: 'Attention-weighted alignment' },
  { key: 'dla', label: 'DLA', title: 'Directional layer attribution' },
  { key: 'attention_standalone', label: 'Attn-S', title: 'Standalone attention' },
  { key: 'retrieval_mark', label: 'MarK', title: 'Retrieval marker attribution' },
]

/**
 * PI/admin-only picker for WHICH attribution method drives the evidence
 * highlighting.
 *
 * The Raw/ABTT variant row this component used to carry is gone, and so is the
 * reviewer-facing variant picker that replaced it: since issue #94 the app
 * serves one pipeline (SIF+ABTT) everywhere, so the evidence highlighting has
 * no variant to choose. The method choice stays gated, preserving the evidence
 * gating from commit e71a7ad.
 */
export default function AttributionMethodSelector() {
  const { selectedMethod, availableMethods, setSelectedMethod } = useTokens()

  // Hide the whole selector when the current pair has no attribution data.
  if (availableMethods.length === 0) {
    return null
  }

  return (
    <div data-tour="attribution-controls" className="flex flex-col gap-1">
      <div
        className="flex flex-wrap bg-stone-100 dark:bg-stone-800 rounded-lg p-0.5"
        role="radiogroup"
        aria-label="Attribution method"
      >
        {METHODS.map((method) => {
          const isAvailable = availableMethods.includes(method.key)
          const isActive = selectedMethod === method.key
          return (
            <button
              key={method.key}
              type="button"
              role="radio"
              aria-checked={isActive}
              aria-label={method.title}
              title={method.title}
              disabled={!isAvailable}
              onClick={() => {
                if (isAvailable) setSelectedMethod(method.key)
              }}
              className={`relative px-3 py-1 text-xs font-medium rounded-md transition-all ${
                !isAvailable
                  ? 'opacity-40 cursor-not-allowed text-stone-500'
                  : isActive
                    ? 'text-stone-800 dark:text-stone-100'
                    : 'text-stone-500 hover:text-stone-700 dark:hover:text-stone-300'
              }`}
            >
              {isActive && isAvailable && (
                <motion.div
                  layoutId="attr-method-indicator"
                  className="absolute inset-0 bg-white dark:bg-stone-700 rounded-md shadow-sm"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
              <span className="relative z-10">{method.label}</span>
            </button>
          )
        })}
      </div>

    </div>
  )
}
