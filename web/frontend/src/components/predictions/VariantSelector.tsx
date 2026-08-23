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
  const { activeModel, activeVariant, setActiveVariant, resolveVariant } = useApp()
  const { data: models } = useModels()

  const model = useMemo(
    () => models?.find((m) => m.slug === activeModel) ?? models?.[0] ?? null,
    [models, activeModel],
  )

  // Variants this deployment can actually serve. Empty until /api/models
  // resolves, which is also the signal not to resolve anything yet.
  const available = useMemo(
    () => model?.available_variants ?? [],
    [model],
  )
  const serverDefault = model?.default_variant ?? null

  // ONE resolution point. Deliberately a single effect rather than an
  // "adopt the default" effect plus an "is it available" effect: two effects
  // commit together and the second would read the pre-default activeVariant
  // from the same render closure, so its fallback would win over the
  // deployment's default. AppContext.resolveVariant does the whole ordering
  // (reviewer's choice -> server default -> first served) in one pass and
  // never persists what it picks.
  useEffect(() => {
    if (!models) return
    resolveVariant(serverDefault, available)
  }, [models, serverDefault, available, resolveVariant])

  const isAvailable = (key: PredictionVariant): boolean =>
    available.length === 0 || available.includes(key)

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
