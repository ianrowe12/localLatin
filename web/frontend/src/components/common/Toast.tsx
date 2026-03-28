import { useEffect } from 'react'
import { motion } from 'framer-motion'

interface ToastProps {
  message: string
  onUndo: () => void
  duration?: number
  onClose: () => void
}

export default function Toast({ message, onUndo, duration = 5000, onClose }: ToastProps) {
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
      <span className="text-sm">{message}</span>
      <button
        type="button"
        onClick={onUndo}
        className="text-accent-light hover:text-white font-medium text-sm underline
                   transition-colors"
      >
        Undo
      </button>
    </motion.div>
  )
}
