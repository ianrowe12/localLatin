import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'light' | 'dark'

export interface AppContextValue {
  theme: Theme
  toggleTheme: () => void
  activeQueryId: number | null
  setActiveQueryId: (id: number | null) => void
  activeModel: string
  setActiveModel: (slug: string) => void
  activePredictionRank: number
  setActivePredictionRank: (rank: number) => void
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

  const value: AppContextValue = {
    theme,
    toggleTheme,
    activeQueryId,
    setActiveQueryId,
    activeModel,
    setActiveModel,
    activePredictionRank,
    setActivePredictionRank,
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
