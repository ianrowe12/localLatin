import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  DEFAULT_VARIANT,
  isPredictionVariant,
  type PredictionVariant,
} from '../api/variants'

export type Theme = 'light' | 'dark'

export type View = 'dashboard' | 'review' | 'files'

export interface AppContextValue {
  theme: Theme
  toggleTheme: () => void
  currentView: View
  setCurrentView: (view: View) => void
  activeQueryId: number | null
  setActiveQueryId: (id: number | null) => void
  activeModel: string
  setActiveModel: (slug: string) => void
  /** Post-processing variant every view is computed from (issue #48). */
  activeVariant: PredictionVariant
  /** Records an explicit reviewer choice: persisted and never overridden. */
  setActiveVariant: (variant: PredictionVariant) => void
  /**
   * Machine resolution of which variant to *display*, given what
   * GET /api/models says this deployment serves. Never persists and never
   * counts as a reviewer choice. See the implementation for the ordering.
   */
  resolveVariant: (
    serverDefault: PredictionVariant | null,
    available: PredictionVariant[],
  ) => void
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
const VARIANT_KEY = 'locallatin-variant'

/** The reviewer's persisted variant choice, or null if they never made one. */
function getStoredVariant(): PredictionVariant | null {
  try {
    const stored = localStorage.getItem(VARIANT_KEY)
    return isPredictionVariant(stored) ? stored : null
  } catch {
    return null
  }
}

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
  // The reviewer's own choice, or null if they have never made one. Seeded
  // from localStorage so it survives a reload, and updated by
  // setActiveVariant. This is the ONLY thing resolveVariant treats as a
  // preference, which is what keeps a machine fallback from masquerading as
  // one. Until GET /api/models reports what this deployment serves,
  // activeVariant shows the stored choice or the compile-time default.
  const chosenVariant = useRef<PredictionVariant | null>(getStoredVariant())
  const [activeVariant, setActiveVariantState] = useState<PredictionVariant>(
    () => chosenVariant.current ?? DEFAULT_VARIANT,
  )
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

  const setActiveVariant = useCallback((variant: PredictionVariant) => {
    chosenVariant.current = variant
    setActiveVariantState(variant)
    // Every variant re-ranks the candidates, so rank 1 of the old variant is
    // meaningless under the new one -- restart at the new top hit.
    setActivePredictionRank(1)
    try {
      localStorage.setItem(VARIANT_KEY, variant)
    } catch {
      // Private-mode / quota failures just cost persistence, not the choice.
    }
  }, [])

  /**
   * Decide what to display once the deployment has described itself, in one
   * ordered pass: the reviewer's own choice if it is served here, else the
   * deployment's declared `default_variant`, else the first variant served.
   *
   * Two invariants this exists to protect, both of which a separate
   * "adopt the default" / "fall back to available[0]" pair got wrong:
   *
   *  - The server default beats "first served". Picking available[0] when the
   *    compile-time default is unserved would strand a reviewer on Raw in a
   *    deployment that declared, say, sif as its default.
   *  - A machine fallback is never persisted and never becomes
   *    `chosenVariant`. A stored preference therefore survives its variant
   *    being temporarily unserved and comes back when the CSV returns.
   *
   * Idempotent: re-running it after an explicit choice re-picks that choice.
   */
  const resolveVariant = useCallback(
    (serverDefault: PredictionVariant | null, available: PredictionVariant[]) => {
      const isServed = (variant: PredictionVariant | null | undefined): boolean =>
        variant != null && (available.length === 0 || available.includes(variant))
      const target = [chosenVariant.current, serverDefault, available[0]].find(isServed)
      if (target) setActiveVariantState(target)
    },
    [],
  )

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
    activeVariant,
    setActiveVariant,
    resolveVariant,
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
