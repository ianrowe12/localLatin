import { useEffect } from 'react'
import { motion } from 'framer-motion'

interface ToastProps {
  message: string
  /** A second line, for what the message must not be read as promising. */
  detail?: string
  /** Omitted when there is no honest action to offer. */
  actionLabel?: string
  onAction?: () => void
  duration?: number
  onClose: () => void
}

export default function Toast({
  message,
  detail,
  actionLabel,
  onAction,
  duration = 5000,
  onClose,
}: ToastProps) {
  useEffect(() => {
    const timer = setTimeout(onClose, duration)
    return () => clearTimeout(timer)
  }, [duration, onClose])

  return (
    <motion.div
      initial={{ y: 40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 40, opacity: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50
                 bg-stone-800 dark:bg-stone-700 text-white rounded-xl
                 px-4 py-3 flex items-center gap-3 shadow-xl"
      role="status"
      aria-live="polite"
    >
      <span className="flex flex-col">
        <span className="text-sm">{message}</span>
        {detail !== undefined && (
          <span className="text-xs text-stone-300">{detail}</span>
        )}
      </span>
      {actionLabel !== undefined && onAction !== undefined && (
        <button
          type="button"
          onClick={onAction}
          className="text-accent-light hover:text-white font-medium text-sm underline
                     transition-colors"
        >
          {actionLabel}
        </button>
      )}
    </motion.div>
  )
}
