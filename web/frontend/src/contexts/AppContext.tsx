import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'light' | 'dark'

export type View = 'dashboard' | 'review'

export interface AppContextValue {
  theme: Theme
  toggleTheme: () => void
  currentView: View
  setCurrentView: (view: View) => void
  activeQueryId: number | null
  setActiveQueryId: (id: number | null) => void
  activeModel: string
  setActiveModel: (slug: string) => void
  activePredictionRank: number
  setActivePredictionRank: (rank: number) => void
  navigateToQuery: (fileId: number) => void
}

const AppContext = createContext<AppContextValue | null>(null)

const THEME_KEY = 'locallatin-theme'

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark'
  return 'light'
}

function applyThemeClass(theme: Theme): void {
  document.documentElement.classList.remove('light', 'dark')
  document.documentElement.classList.add(theme)
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  const [currentView, setCurrentView] = useState<View>('dashboard')
  const [activeQueryId, setActiveQueryId] = useState<number | null>(null)
  const [activeModel, setActiveModel] = useState<string>('')
  const [activePredictionRank, setActivePredictionRank] = useState<number>(1)

  useEffect(() => {
    applyThemeClass(theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))
  }, [])

  const navigateToQuery = useCallback((fileId: number) => {
    setActiveQueryId(fileId)
    setActivePredictionRank(1)
    setCurrentView('review')
  }, [])

  const value: AppContextValue = {
    theme,
    toggleTheme,
    currentView,
    setCurrentView,
    activeQueryId,
    setActiveQueryId,
    activeModel,
    setActiveModel,
    activePredictionRank,
    setActivePredictionRank,
    navigateToQuery,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) {
    throw new Error('useApp must be used within an AppProvider')
  }
  return ctx
}
