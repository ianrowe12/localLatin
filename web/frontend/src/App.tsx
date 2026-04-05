import { AppProvider } from './contexts/AppContext'
import { TokenProvider } from './contexts/TokenContext'
import { FeedbackProvider } from './contexts/FeedbackContext'
import { TourProvider } from './components/onboarding/TourProvider'
import { ReviewerProvider, useReviewer } from './contexts/ReviewerContext'
import AppShell from './components/layout/AppShell'
import ReviewerLoginModal from './components/auth/ReviewerLoginModal'

function AppContent() {
  const { reviewerName } = useReviewer()

  return (
    <>
      {reviewerName === null && <ReviewerLoginModal />}
      <AppShell />
    </>
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
