import { AppProvider } from './contexts/AppContext'
import { TokenProvider } from './contexts/TokenContext'
import { FeedbackProvider } from './contexts/FeedbackContext'
import AppShell from './components/layout/AppShell'

export default function App() {
  return (
    <AppProvider>
      <TokenProvider>
        <FeedbackProvider>
          <AppShell />
        </FeedbackProvider>
      </TokenProvider>
    </AppProvider>
  )
}
