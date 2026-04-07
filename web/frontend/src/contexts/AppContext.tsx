import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'light' | 'dark'

export type View = 'dashboard' | 'review' | 'examples'

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
  overrideCandidateDir: string | null
  setOverrideCandidateDir: (dir: string | null) => void
  navigateToExample: (
    model: string,
    queryFileId: number,
    candidateDir: string,
  ) => void
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
  const [overrideCandidateDir, setOverrideCandidateDirState] = useState<
    string | null
  >(null)
  // Tracks the queryId that the current override was set for. When
  // activeQueryId diverges from this (e.g. user picks a different query),
  // the override is auto-cleared.
  const overrideQueryIdRef = useRef<number | null>(null)

  const setOverrideCandidateDir = useCallback((dir: string | null) => {
    if (dir === null) {
      overrideQueryIdRef.current = null
    }
    setOverrideCandidateDirState(dir)
  }, [])

  useEffect(() => {
    applyThemeClass(theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  // If the override belongs to a different query than the currently active
  // one, clear it. This guarantees the override never leaks across unrelated
  // query navigations, while still allowing navigateToExample to set it for
  // the query it is navigating to.
  useEffect(() => {
    if (
      overrideCandidateDir !== null &&
      overrideQueryIdRef.current !== null &&
      overrideQueryIdRef.current !== activeQueryId
    ) {
      overrideQueryIdRef.current = null
      setOverrideCandidateDirState(null)
    }
  }, [activeQueryId, overrideCandidateDir])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))
  }, [])

  const navigateToQuery = useCallback((fileId: number) => {
    setActiveQueryId(fileId)
    setActivePredictionRank(1)
    setCurrentView('review')
  }, [])

  const navigateToExample = useCallback(
    (model: string, queryFileId: number, candidateDir: string) => {
      overrideQueryIdRef.current = queryFileId
      setActiveModel(model)
      setActiveQueryId(queryFileId)
      setActivePredictionRank(1)
      setOverrideCandidateDir(candidateDir)
      setCurrentView('review')
    },
    [],
  )

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
    overrideCandidateDir,
    setOverrideCandidateDir,
    navigateToExample,
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
