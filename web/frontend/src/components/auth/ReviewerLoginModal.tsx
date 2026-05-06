import { useState, type FormEvent } from 'react'
import { motion } from 'framer-motion'
import { useReviewer } from '../../contexts/ReviewerContext'

type Mode = 'signin' | 'register'

export default function ReviewerLoginModal() {
  const { signIn, register, error } = useReviewer()
  const [mode, setMode] = useState<Mode>('signin')
  const [username, setUsername] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [adminCode, setAdminCode] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const canSubmit =
    username.trim().length >= 2 &&
    password.length >= 12 &&
    (mode === 'signin' || displayName.trim().length > 0)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!canSubmit || submitting) return
    setSubmitting(true)
    try {
      if (mode === 'signin') {
        await signIn({ username, password })
      } else {
        await register({
          username,
          display_name: displayName,
          password,
          admin_code: adminCode.trim() || undefined,
        })
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className="w-full max-w-md mx-4 rounded-xl
                   bg-white dark:bg-surface-800
                   border border-stone-200/60 dark:border-stone-700/60
                   shadow-glass dark:shadow-glass-dark
                   p-6"
      >
        <h2 className="font-latin text-xl font-semibold text-stone-800 dark:text-stone-100 mb-1">
          LocalLatin Access
        </h2>
        <p className="font-ui text-sm text-stone-500 dark:text-stone-400 mb-5">
          Sign in to review manuscripts. Display names and initials are used for
          attribution; account sign-in controls access.
        </p>

        <div className="grid grid-cols-2 gap-2 mb-4">
          <button
            type="button"
            onClick={() => setMode('signin')}
            className={`h-9 rounded-lg text-sm font-ui transition-colors ${
              mode === 'signin'
                ? 'bg-accent text-white'
                : 'bg-stone-100 dark:bg-surface-900 text-stone-600 dark:text-stone-300'
            }`}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={() => setMode('register')}
            className={`h-9 rounded-lg text-sm font-ui transition-colors ${
              mode === 'register'
                ? 'bg-accent text-white'
                : 'bg-stone-100 dark:bg-surface-900 text-stone-600 dark:text-stone-300'
            }`}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            autoFocus
            className="w-full h-10 px-3 rounded-lg border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Username"
          />

          {mode === 'register' && (
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Display name for attribution"
              className="w-full h-10 px-3 rounded-lg border border-stone-200 dark:border-stone-700
                         bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                         placeholder:text-stone-400 dark:placeholder:text-stone-500
                         focus:outline-none focus:ring-2 focus:ring-accent/30"
              aria-label="Display name"
            />
          )}

          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full h-10 px-3 rounded-lg border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Password"
          />

          {mode === 'register' && (
            <input
              type="password"
              value={adminCode}
              onChange={(e) => setAdminCode(e.target.value)}
              placeholder="Admin code (optional)"
              className="w-full h-10 px-3 rounded-lg border border-stone-200 dark:border-stone-700
                         bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                         placeholder:text-stone-400 dark:placeholder:text-stone-500
                         focus:outline-none focus:ring-2 focus:ring-accent/30"
              aria-label="Admin code"
            />
          )}

          {error && (
            <p className="font-ui text-sm text-red-600 dark:text-red-400">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={!canSubmit || submitting}
            className="bg-accent hover:bg-accent-dark text-white
                       px-4 py-2.5 rounded-lg text-sm font-medium font-ui
                       transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting
              ? 'Working...'
              : mode === 'signin'
                ? 'Sign in'
                : 'Create account'}
          </button>
        </form>
      </motion.div>
    </div>
  )
}
