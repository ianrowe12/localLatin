import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  getCurrentUser,
  register as registerAccount,
  signIn as signInAccount,
  signOut as signOutAccount,
  type AuthUser,
  type RegisterPayload,
  type RegistrationResponse,
  type SignInPayload,
} from '../api/auth'

export interface ReviewerContextValue {
  user: AuthUser | null
  reviewerName: string | null
  loading: boolean
  error: string | null
  isPiAdmin: boolean
  signIn: (payload: SignInPayload) => Promise<void>
  register: (payload: RegisterPayload) => Promise<RegistrationResponse>
  clearReviewer: () => Promise<void>
}

const ReviewerContext = createContext<ReviewerContextValue | null>(null)

export function ReviewerProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getCurrentUser()
      .then((currentUser) => {
        if (cancelled) return
        setUser(currentUser)
        setError(null)
      })
      .catch(() => {
        if (cancelled) return
        setUser(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const signIn = useCallback(async (payload: SignInPayload) => {
    setError(null)
    try {
      setUser(await signInAccount(payload))
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Sign-in failed'
      setError(message)
      throw err
    }
  }, [])

  const register = useCallback(async (payload: RegisterPayload) => {
    setError(null)
    try {
      const result = await registerAccount(payload)
      setUser(result.status === 'approved' ? result.account : null)
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed'
      setError(message)
      throw err
    }
  }, [])

  const clearReviewer = useCallback(async () => {
    await signOutAccount()
    setUser(null)
  }, [])

  const value = useMemo<ReviewerContextValue>(
    () => ({
      user,
      reviewerName: user?.display_name ?? null,
      loading,
      error,
      isPiAdmin: user?.role === 'pi_admin',
      signIn,
      register,
      clearReviewer,
    }),
    [clearReviewer, error, loading, register, signIn, user],
  )

  return (
    <ReviewerContext.Provider value={value}>{children}</ReviewerContext.Provider>
  )
}

export function useReviewer(): ReviewerContextValue {
  const ctx = useContext(ReviewerContext)
  if (!ctx) {
    throw new Error('useReviewer must be used within a ReviewerProvider')
  }
  return ctx
}
