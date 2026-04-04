import { AppProvider } from './contexts/AppContext'
import { TokenProvider } from './contexts/TokenContext'
import { FeedbackProvider } from './contexts/FeedbackContext'
import { TourProvider } from './components/onboarding/TourProvider'
import AppShell from './components/layout/AppShell'

export default function App() {
  return (
    <AppProvider>
      <TokenProvider>
        <FeedbackProvider>
          <TourProvider>
            <AppShell />
          </TourProvider>
        </FeedbackProvider>
      </TokenProvider>
    </AppProvider>
  )
}
