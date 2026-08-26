import { useState, type FormEvent } from 'react'
import { motion } from 'framer-motion'
import { MIN_PASSWORD_LENGTH } from '../../api/auth'
import { useReviewer } from '../../contexts/ReviewerContext'

interface Props {
  /** Forced mode is the post-reset gate: no way out except changing it. */
  forced?: boolean
  onClose?: () => void
}

const inputClass = `w-full h-10 px-3 rounded-lg border border-stone-200 dark:border-stone-700
   bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
   placeholder:text-stone-400 dark:placeholder:text-stone-500
   focus:outline-none focus:ring-2 focus:ring-accent/30`

export default function ChangePasswordModal({ forced = false, onClose }: Props) {
  const { changePassword, clearReviewer } = useReviewer()
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)

  const tooShort = newPassword.length > 0 && newPassword.length < MIN_PASSWORD_LENGTH
  const mismatch = confirmPassword.length > 0 && confirmPassword !== newPassword
  const canSubmit =
    currentPassword.length > 0 &&
    newPassword.length >= MIN_PASSWORD_LENGTH &&
    newPassword === confirmPassword &&
    newPassword !== currentPassword

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!canSubmit || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
      setDone(true)
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      if (!forced) onClose?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Password change failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Change password"
    >
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
          {forced ? 'Set a New Password' : 'Change Password'}
        </h2>
        <p className="font-ui text-sm text-stone-500 dark:text-stone-400 mb-5">
          {forced
            ? 'A PI/admin reset your password. Choose your own before continuing.'
            : `Enter your current password, then a new one of at least ${MIN_PASSWORD_LENGTH} characters. Your other sessions are signed out.`}
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <input
            type="password"
            value={currentPassword}
            onChange={(event) => setCurrentPassword(event.target.value)}
            placeholder={forced ? 'Temporary password' : 'Current password'}
            autoFocus
            autoComplete="current-password"
            className={inputClass}
            aria-label={forced ? 'Temporary password' : 'Current password'}
          />
          <input
            type="password"
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
            placeholder="New password"
            autoComplete="new-password"
            className={inputClass}
            aria-label="New password"
          />
          <input
            type="password"
            value={confirmPassword}
            onChange={(event) => setConfirmPassword(event.target.value)}
            placeholder="Confirm new password"
            autoComplete="new-password"
            className={inputClass}
            aria-label="Confirm new password"
          />

          {tooShort && (
            <p className="font-ui text-sm text-stone-500 dark:text-stone-400">
              Use at least {MIN_PASSWORD_LENGTH} characters.
            </p>
          )}
          {mismatch && (
            <p className="font-ui text-sm text-stone-500 dark:text-stone-400">
              The two new passwords do not match.
            </p>
          )}
          {error && (
            <p className="font-ui text-sm text-red-600 dark:text-red-400" role="alert">
              {error}
            </p>
          )}
          {done && forced && (
            <p className="font-ui text-sm text-accent dark:text-accent-light">
              Password updated.
            </p>
          )}

          <div className="flex items-center gap-2">
            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="flex-1 bg-accent hover:bg-accent-dark text-white
                         px-4 py-2.5 rounded-lg text-sm font-medium font-ui
                         transition-all
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? 'Saving...' : 'Change password'}
            </button>
            {forced ? (
              <button
                type="button"
                onClick={() => void clearReviewer()}
                className="px-4 py-2.5 rounded-lg text-sm font-medium font-ui
                           bg-stone-100 dark:bg-surface-900 text-stone-600 dark:text-stone-300
                           hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors"
              >
                Sign out
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2.5 rounded-lg text-sm font-medium font-ui
                           bg-stone-100 dark:bg-surface-900 text-stone-600 dark:text-stone-300
                           hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </form>
      </motion.div>
    </div>
  )
}
