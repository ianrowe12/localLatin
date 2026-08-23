import { useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import { useModels } from '../../api/models'
import { VARIANT_OPTIONS, type PredictionVariant } from '../../api/variants'

/**
 * The single 4-way post-processing selector (issue #48).
 *
 * Deliberately NOT PI-gated: reviewers compare variants themselves, so this
 * control is part of the ordinary review surface. (The attribution *method*
 * picker stays PI-only — see LeftSidebar.)
 *
 * Choosing a variant re-drives the ranked prediction list, the candidate texts,
 * the feedback draft/prefill and the token highlights, and is recorded on any
 * feedback submitted afterwards.
 */
export default function VariantSelector() {
  const { activeModel, activeVariant, setActiveVariant, applyDefaultVariant } = useApp()
  const { data: models } = useModels()

  const model = useMemo(
    () => models?.find((m) => m.slug === activeModel) ?? models?.[0] ?? null,
    [models, activeModel],
  )

  // Variants this deployment can actually serve. Before /api/models resolves,
  // treat every variant as available so the control is never briefly empty.
  const available = model?.available_variants ?? null

  // Adopt the deployment's default once, unless the reviewer already chose.
  const serverDefault = model?.default_variant
  useEffect(() => {
    if (serverDefault) applyDefaultVariant(serverDefault)
  }, [serverDefault, applyDefaultVariant])

  // If the selected variant is not served here (e.g. a stale localStorage
  // value, or a deployment that ships fewer CSVs), fall back to the first
  // variant that is.
  useEffect(() => {
    if (!available || available.length === 0) return
    if (available.includes(activeVariant)) return
    setActiveVariant(available[0])
  }, [available, activeVariant, setActiveVariant])

  const isAvailable = (key: PredictionVariant): boolean =>
    available === null || available.length === 0 || available.includes(key)

  return (
    <div
      data-tour="variant-selector"
      className="flex bg-stone-100 dark:bg-stone-800 rounded-lg p-0.5"
      role="radiogroup"
      aria-label="Post-processing variant"
    >
      {VARIANT_OPTIONS.map((option) => {
        const enabled = isAvailable(option.key)
        const isActive = activeVariant === option.key
        const title = `${option.label} — ${option.description}`
        return (
          <button
            key={option.key}
            type="button"
            role="radio"
            aria-checked={isActive}
            aria-label={title}
            title={
              enabled ? title : `${title} (not available in this deployment)`
            }
            disabled={!enabled}
            onClick={() => {
              if (enabled) setActiveVariant(option.key)
            }}
            className={`relative flex-1 px-1.5 py-1 text-xs font-medium rounded-md transition-all ${
              !enabled
                ? 'opacity-40 cursor-not-allowed text-stone-500'
                : isActive
                  ? 'text-stone-800 dark:text-stone-100'
                  : 'text-stone-500 hover:text-stone-700 dark:hover:text-stone-300'
            }`}
          >
            {isActive && enabled && (
              <motion.div
                layoutId="prediction-variant-indicator"
                className="absolute inset-0 bg-white dark:bg-stone-700 rounded-md shadow-sm"
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
            <span className="relative z-10">{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}
