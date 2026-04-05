import { useState, type FormEvent } from 'react'
import { motion } from 'framer-motion'
import { useReviewer } from '../../contexts/ReviewerContext'

export default function ReviewerLoginModal() {
  const { setReviewerName } = useReviewer()
  const [input, setInput] = useState('')

  const canSubmit = input.trim().length > 0

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (canSubmit) {
      setReviewerName(input)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className="w-full max-w-sm mx-4 rounded-xl
                   bg-white dark:bg-surface-800
                   border border-stone-200/60 dark:border-stone-700/60
                   shadow-glass dark:shadow-glass-dark
                   p-6"
      >
        <h2 className="font-latin text-xl font-semibold text-stone-800 dark:text-stone-100 mb-1">
          Welcome to LocalLatin
        </h2>
        <p className="font-ui text-sm text-stone-500 dark:text-stone-400 mb-5">
          Enter your name to begin reviewing.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Your name"
            autoFocus
            className="w-full h-10 px-3 rounded-lg
                       border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900
                       text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Reviewer name"
          />

          <button
            type="submit"
            disabled={!canSubmit}
            className="bg-accent hover:bg-accent-dark text-white
                       px-4 py-2.5 rounded-lg text-sm font-medium font-ui
                       transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Start Reviewing
          </button>
        </form>
      </motion.div>
    </div>
  )
}
