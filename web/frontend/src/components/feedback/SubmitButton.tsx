import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface SubmitButtonProps {
  onSubmit: () => void
  onSkip: () => void
  disabled?: boolean
  skipDisabled?: boolean
  /**
   * What the primary button says. Default: the ordinary save.
   *
   * It is a prop rather than a constant because the same button sometimes only
   * MOVES the reviewer on (issue #221): once a none-of-top-N answer is recorded
   * by the blue action, pressing this writes nothing, and a button still
   * reading "Submit" over an answer already in an append-only log is an
   * invitation to record it twice.
   */
  label?: string
  /**
   * Whether pressing it is a save, and may therefore flash the tick.
   *
   * The tick is the app saying "recorded". On the move-only press nothing is
   * recorded, so it stays an arrow: a confirmation of a write nobody made is
   * exactly the receipt a reviewer would act on.
   */
  confirmsSave?: boolean
}

export default function SubmitButton({
  onSubmit,
  onSkip,
  disabled,
  skipDisabled,
  label = 'Submit & Next',
  confirmsSave = true,
}: SubmitButtonProps) {
  const [showCheck, setShowCheck] = useState(false)

  const handleSubmit = () => {
    onSubmit()
    if (!confirmsSave) return
    setShowCheck(true)
    setTimeout(() => setShowCheck(false), 500)
  }

  return (
    <div data-tour="submit-skip" className="flex gap-2">
      <button
        type="button"
        data-testid="assessment-primary-action"
        onClick={handleSubmit}
        disabled={disabled}
        className="bg-accent hover:bg-accent-dark text-white px-4 py-2 rounded-lg
                   text-sm font-medium flex items-center gap-2 transition-all
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {label}
        <AnimatePresence mode="wait" initial={false}>
          {showCheck ? (
            <motion.span
              key="check"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ duration: 0.15 }}
              aria-hidden="true"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </motion.span>
          ) : (
            <motion.span
              key="arrow"
              initial={{ x: -4, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 4, opacity: 0 }}
              transition={{ duration: 0.15 }}
              aria-hidden="true"
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
              >
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </motion.span>
          )}
        </AnimatePresence>
      </button>

      <button
        type="button"
        onClick={onSkip}
        disabled={skipDisabled}
        className="text-stone-500 hover:text-stone-700 dark:hover:text-stone-300
                   hover:bg-stone-100 dark:hover:bg-stone-800
                   px-3 py-2 rounded-lg text-sm transition-colors
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Skip
      </button>
    </div>
  )
}
