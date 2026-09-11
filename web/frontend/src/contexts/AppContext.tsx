import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { DEFAULT_VARIANT, type PredictionVariant } from '../api/variants'

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
  /**
   * The single post-processing pipeline every view is computed from.
   *
   * Constant since issue #94: the reviewer-facing variant picker is gone and
   * the app serves SIF+ABTT only, so predictions, candidate texts, highlights
   * and feedback rows all speak of one variant. The backend still serves the
   * other three (and feedback still records which variant a row belongs to),
   * so this stays a typed value threaded through the API calls rather than a
   * hardcoded string at each call site.
   */
  activeVariant: PredictionVariant
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
  // One pipeline, no choice to make and nothing to persist (issue #94).
  const activeVariant: PredictionVariant = DEFAULT_VARIANT
  /**
   * The gallery override, bound to the query it was chosen for (issue #156).
   *
   * A gallery example is a pair: "this candidate directory, for THIS query".
   * Storing only the directory made the two halves separable, and the query
   * half was the one that moved -- navigating to another document left the old
   * example's directory in place, and the passive effect below only cleared it
   * a commit later. The first render for the new query therefore inspected the
   * previous example's candidate and asked the token-map endpoint for evidence
   * about a pair no reviewer had selected.
   *
   * `queryId` is captured when the override is set, never inferred afterwards.
   */
  const [override, setOverrideState] = useState<{
    dir: string
    queryId: number | null
  } | null>(null)

  // Mirror of the rendered query id, so a caller that sets an override without
  // naming a query (the picker, as opposed to the gallery) binds it to the
  // document actually on screen. A plain mirror, never derived state.
  const activeQueryIdRef = useRef(activeQueryId)
  activeQueryIdRef.current = activeQueryId

  const setOverrideCandidateDir = useCallback((dir: string | null) => {
    setOverrideState(
      dir === null ? null : { dir, queryId: activeQueryIdRef.current },
    )
  }, [])

  // Render-time identity check, deliberately not an effect. Consumers derive
  // the candidate directory, its file request and its token evidence straight
  // from this value, so it has to be right on the FIRST render after the query
  // changes -- one commit late is one painted screen of another document's
  // evidence.
  const overrideCandidateDir =
    override !== null && override.queryId === activeQueryId ? override.dir : null

  useEffect(() => {
    applyThemeClass(theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  // Housekeeping only: the guard above has already made a stale override
  // invisible, and this drops the dead state so nothing can resurrect it by
  // navigating back to the original query.
  useEffect(() => {
    if (override !== null && override.queryId !== activeQueryId) {
      setOverrideState(null)
    }
  }, [activeQueryId, override])

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
      setActiveModel(model)
      setActiveQueryId(queryFileId)
      setActivePredictionRank(1)
      // Bound to the query being navigated to, not to the one still on screen:
      // this is the one call that legitimately sets an override for a document
      // other than the current one.
      setOverrideState({ dir: candidateDir, queryId: queryFileId })
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

/**
 * The same value, for a reader that must work without this provider.
 *
 * Used by the live evidence stamp (issue #163), which is consulted from
 * `useTokens` -- a hook that legitimately runs in isolated component tests
 * mounted under `TokenProvider` alone. Those callers get null and are not
 * gated, rather than throwing.
 */
export function useOptionalApp(): AppContextValue | null {
  return useContext(AppContext)
}
