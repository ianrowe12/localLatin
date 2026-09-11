import { useEffect, type ReactNode } from 'react'
import { AppProvider, useApp } from './contexts/AppContext'
import { TokenProvider } from './contexts/TokenContext'
import { FeedbackProvider } from './contexts/FeedbackContext'
import { PredictionProvider } from './contexts/PredictionContext'
import { SavedDirectoryProvider } from './contexts/SavedDirectoryContext'
import SavedDirectoryObservations from './contexts/SavedDirectoryObservations'
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

/**
 * Saved-directory state, scoped to the signed-in account (issue #161).
 *
 * Separated out only because `accountKey` has to come from `useReviewer`, which
 * is not readable in `App` itself.
 *
 * Reviewer directories are GLOBAL. `web/services/reviewer_dirs.py` serves the
 * same rows to every reviewer, so a document grouped for one of them is grouped
 * for all of them, and the same groups are expected to reappear here as soon as
 * the next account's own request answers. What must not survive a sign-out is
 * the provenance of the record: it holds what the database said to one
 * authenticated session, together with that session's unfinished writes and
 * unanswered lookups. Carrying it across would make another session's evidence
 * durable under this one's name, which is exactly the weight the record is
 * given everywhere else. So the provider derives a NEW store during the render
 * in which `accountKey` changes -- the old records become unreachable rather
 * than emptied, not readable for even one frame -- and the new account
 * re-establishes the same facts for itself.
 */
export function AccountScopedSavedDirectories({ children }: { children: ReactNode }) {
  const { user } = useReviewer()
  return (
    <SavedDirectoryProvider accountKey={user?.id ?? null}>
      {children}
    </SavedDirectoryProvider>
  )
}

export default function App() {
  return (
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          {/* Above PredictionProvider, and so above every loading, error,
              empty and ready branch of the prediction list: creating a
              directory broadcasts a refetch that replaces that subtree, and the
              acknowledgement of a permanent write must outlive it (issue
              #161). */}
          <AccountScopedSavedDirectories>
            {/* Above FeedbackProvider and the review views, so the list, the
                candidate panel and the assessment panel all read one ranking for
                the current query/model/variant (issue #156). It issues no request
                until a query and a model are chosen. */}
            <PredictionProvider>
              {/* The single admission point for `seeded_dirs`: the shared
                  prediction key carries no account, so this decides which
                  responses are this session's own answer before any of them
                  reaches the durable record (issue #161). Mounted here, not in
                  the list, so it is not reset by a view change. */}
              <SavedDirectoryObservations />
              <FeedbackProvider>
                <TourProvider>
                  <AppContent />
                </TourProvider>
              </FeedbackProvider>
            </PredictionProvider>
          </AccountScopedSavedDirectories>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}
