import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import { FEEDBACK_UPDATED_EVENT } from '../../api/feedback'

// After this many consecutive submitted reviews on the same model, gently
// suggest trying a different model for comparison.
const THRESHOLD = 5

/**
 * Gentle, non-blocking nudge encouraging reviewers to vary the model they
 * review with. Counts consecutive feedback submissions while `activeModel`
 * stays unchanged; the counter (and any dismissal) reset when the reviewer
 * switches models. State is in-memory / per session — no backend, no storage.
 */
export default function ModelDiversityNudge() {
  const { activeModel } = useApp()
  const [count, setCount] = useState(0)
  const [dismissed, setDismissed] = useState(false)

  // Reset whenever the reviewer changes model — this is both the "reset on
  // model change" requirement and the self-clearing behavior.
  useEffect(() => {
    setCount(0)
    setDismissed(false)
  }, [activeModel])

  // Each feedback submission (submit or skip) dispatches this event. Since
  // feedback is always submitted with the current activeModel, an event here
  // means one more review on the model being tracked.
  useEffect(() => {
    const handler = () => setCount((c) => c + 1)
    window.addEventListener(FEEDBACK_UPDATED_EVENT, handler)
    return () => window.removeEventListener(FEEDBACK_UPDATED_EVENT, handler)
  }, [])

  const show = Boolean(activeModel) && count >= THRESHOLD && !dismissed

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="mt-1.5 flex items-start gap-2 rounded-lg border border-highlight/40
                     bg-highlight-light/10 px-2.5 py-2 text-xs text-stone-600 dark:text-stone-300"
          role="status"
          aria-live="polite"
        >
          <span className="flex-1 leading-snug">
            You have reviewed several items with this model. Try another model for comparison.
          </span>
          <button
            type="button"
            onClick={() => setDismissed(true)}
            className="-mr-0.5 -mt-0.5 shrink-0 rounded p-0.5 text-stone-400
                       hover:text-stone-600 dark:hover:text-stone-200 transition-colors"
            aria-label="Dismiss model diversity suggestion"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
