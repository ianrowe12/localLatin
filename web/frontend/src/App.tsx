import { useEffect } from 'react'
import { AppProvider, useApp } from './contexts/AppContext'
import { TokenProvider } from './contexts/TokenContext'
import { FeedbackProvider } from './contexts/FeedbackContext'
import { TourProvider } from './components/onboarding/TourProvider'
import { ReviewerProvider, useReviewer } from './contexts/ReviewerContext'
import { fetchNextQuery } from './api/queries'
import AppShell from './components/layout/AppShell'
import ReviewerLoginModal from './components/auth/ReviewerLoginModal'
import ChangePasswordModal from './components/auth/ChangePasswordModal'

function AppContent() {
  const { user, loading, mustChangePassword } = useReviewer()
  const {
    activeQueryId,
    currentView,
    navigateToQuery,
    setCurrentView,
  } = useApp()

  useEffect(() => {
    if (mustChangePassword) return
    if (user?.role === 'reviewer' && currentView === 'dashboard') {
      setCurrentView('review')
    }
  }, [currentView, mustChangePassword, setCurrentView, user])

  useEffect(() => {
    if (mustChangePassword) return
    if (user?.role !== 'reviewer' || activeQueryId !== null) return

    let cancelled = false
    fetchNextQuery()
      .then((next) => {
        if (cancelled) return
        if (next.file_id !== null) {
          navigateToQuery(next.file_id)
        } else {
          setCurrentView('review')
        }
      })
      .catch(() => {
        if (!cancelled) setCurrentView('review')
      })

    return () => {
      cancelled = true
    }
  }, [activeQueryId, mustChangePassword, navigateToQuery, setCurrentView, user])

  if (loading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-surface-50 dark:bg-surface-900">
        <div className="font-ui text-sm text-stone-500 dark:text-stone-400">
          Loading...
        </div>
      </div>
    )
  }

  if (user === null) {
    return <ReviewerLoginModal />
  }

  // Backend refuses every other route until this is done; the modal is the
  // only thing on screen so the reviewer cannot get stuck against 403s.
  if (mustChangePassword) {
    return <ChangePasswordModal forced />
  }

  return (
    <AppShell />
  )
}

export default function App() {
  return (
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <FeedbackProvider>
            <TourProvider>
              <AppContent />
            </TourProvider>
          </FeedbackProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}
