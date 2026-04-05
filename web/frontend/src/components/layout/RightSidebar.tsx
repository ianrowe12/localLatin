import { useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useFeedback } from '../../contexts/FeedbackContext'
import ModelSelector from '../predictions/ModelSelector'
import PredictionList from '../predictions/PredictionList'
import FeedbackPanel from '../feedback/FeedbackPanel'
import ViewModeToggle from '../common/ViewModeToggle'
import Toast from '../common/Toast'

interface RightSidebarProps {
  isOpen: boolean
  onToggle: () => void
}

export default function RightSidebar({ isOpen, onToggle }: RightSidebarProps) {
  const { lastSubmittedKey, undoLastSubmit } = useFeedback()

  const handleToastClose = useCallback(() => {
    // Toast auto-dismisses; lastSubmittedKey will be cleared on next submit or undo
  }, [])

  return (
    <motion.aside
      className="h-full flex-shrink-0 overflow-hidden border-l border-stone-200/60 dark:border-stone-700/60
                 bg-white/60 dark:bg-surface-800/60 backdrop-blur-sm"
      animate={{ width: isOpen ? 320 : 48 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      aria-label="Predictions sidebar"
    >
      <div className="flex flex-col h-full overflow-hidden">
        {/* Toggle button row */}
        <div className="flex items-center justify-start h-10 px-2 flex-shrink-0">
          <button
            onClick={onToggle}
            type="button"
            className="w-7 h-7 rounded flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-500 dark:text-stone-400"
            aria-label={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
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
              className={`transition-transform duration-200 ${isOpen ? '' : 'rotate-180'}`}
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        </div>

        {isOpen && (
          <div className="flex flex-col flex-1 overflow-hidden px-3 pb-3 gap-3">
            {/* Model selection */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1.5 block">
                Model Selection
              </label>
              <ModelSelector />
            </div>

            {/* Visualization mode */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-1.5 block">
                Visualization Mode
              </label>
              <ViewModeToggle />
            </div>

            {/* Prediction list — scrollable */}
            <div className="flex-1 overflow-y-auto rounded-lg bg-stone-50 dark:bg-surface-900/50">
              <PredictionList />
            </div>

            {/* Divider */}
            <div className="border-t border-stone-200 dark:border-stone-700" />

            {/* Feedback area */}
            <FeedbackPanel />
          </div>
        )}

        {/* Collapsed: just the toggle button is shown above */}
      </div>

      {/* Undo toast */}
      <AnimatePresence>
        {lastSubmittedKey && (
          <Toast
            message="Feedback submitted"
            onUndo={undoLastSubmit}
            onClose={handleToastClose}
          />
        )}
      </AnimatePresence>
    </motion.aside>
  )
}
