import { useEffect, useRef } from 'react'
import {
  useFloating,
  offset,
  flip,
  shift,
  autoUpdate,
  type Placement,
} from '@floating-ui/react'
import { AnimatePresence, motion } from 'framer-motion'
import type { TourStep } from './tourSteps'

interface TourTooltipProps {
  step: TourStep
  stepIndex: number
  totalSteps: number
  onNext: () => void
  onPrev: () => void
  onSkip: () => void
}

export default function TourTooltip({
  step,
  stepIndex,
  totalSteps,
  onNext,
  onPrev,
  onSkip,
}: TourTooltipProps) {
  const targetRef = useRef<Element | null>(null)
  const isFirst = stepIndex === 0
  const isLast = stepIndex === totalSteps - 1

  const { refs, floatingStyles } = useFloating({
    placement: (step.placement ?? 'bottom') as Placement,
    whileElementsMounted: autoUpdate,
    middleware: [offset(12), flip(), shift({ padding: 16 })],
  })

  // Sync the virtual reference to the actual DOM target element
  useEffect(() => {
    const el = document.querySelector(`[data-tour="${step.target}"]`)
    targetRef.current = el
    if (el) {
      refs.setReference(el)
    }
  }, [step.target, refs])

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={stepIndex}
        ref={refs.setFloating}
        style={floatingStyles}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 8 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className="z-[9999] glass-panel border-2 border-accent/30 p-4 max-w-sm"
        role="dialog"
        aria-label={`Tour step ${stepIndex + 1} of ${totalSteps}: ${step.title}`}
      >
        <h3 className="font-ui text-base font-semibold text-stone-800 dark:text-stone-100">
          {step.title}
        </h3>
        <p className="font-ui text-sm text-stone-600 dark:text-stone-300 mt-1">
          {step.description}
        </p>

        {/* Footer */}
        <div className="flex items-center justify-between mt-4">
          <span className="font-ui text-xs text-stone-400 dark:text-stone-500">
            Step {stepIndex + 1} of {totalSteps}
          </span>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onSkip}
              className="text-stone-500 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200 text-sm font-ui transition-colors"
            >
              Skip tour
            </button>

            {!isFirst && (
              <button
                type="button"
                onClick={onPrev}
                className="text-stone-500 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200 text-sm font-ui transition-colors"
              >
                Back
              </button>
            )}

            <button
              type="button"
              onClick={onNext}
              className="bg-accent text-white px-3 py-1.5 rounded-lg text-sm font-ui font-medium hover:bg-accent-dark transition-colors"
            >
              {isLast ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
