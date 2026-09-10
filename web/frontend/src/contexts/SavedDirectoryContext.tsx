import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from 'react'
import {
  SavedDirectoryStore,
  type SavedDirectoryRecord,
} from './savedDirectoryStore'

/**
 * React access to the durable saved-directory store (issue #161).
 *
 * The store lives outside React on purpose (see `savedDirectoryStore.ts`), so
 * this file is only subscription and lifecycle:
 *
 * - `useSavedDirectory(queryId)` re-renders on that query's record alone, which
 *   is what lets a late completion for query A update A's record while B is on
 *   screen without ever painting A's success onto B.
 * - The provider outlives `PredictionList`'s loading, error and empty branches,
 *   so an acknowledgement no longer disappears when a refresh unmounts the
 *   component that created the directory.
 *
 * `accountKey` scopes the whole store to one signed-in account on one
 * deployment. Reviewer directories are named by numeric query id, so state left
 * over from another account — or another deployment's database — must not be
 * able to claim that this query is already saved. Pass the current account id
 * (for example `useReviewer().user?.id`) when mounting this provider.
 */

interface SavedDirectoryContextValue {
  store: SavedDirectoryStore
}

const SavedDirectoryContext = createContext<SavedDirectoryContextValue | null>(null)

/**
 * The store used when no provider is mounted.
 *
 * App composition is externally owned until issue #156 hands it over, so the
 * creation form has to be durable BEFORE `SavedDirectoryProvider` is wired in.
 * Throwing instead would take the running app down; giving each consumer its
 * own store would silently reinstate the bug, since the whole point is that the
 * acknowledgement outlives the component that made it.
 *
 * It is process-wide and therefore NOT account-scoped: mounting the provider
 * with an `accountKey` is what adds sign-out clearing, and doing so is the
 * remaining integration step. The claim it holds ("this document seeds a
 * grouping") is a global fact about the database rather than a per-reviewer
 * one, which is why an unscoped fallback is tolerable in the meantime.
 */
let defaultStore: SavedDirectoryStore | null = null

function getDefaultStore(): SavedDirectoryStore {
  if (defaultStore === null) defaultStore = new SavedDirectoryStore()
  return defaultStore
}

/** Test isolation for the fallback, like `__resetModelsCache` for models. */
export function __resetDefaultSavedDirectoryStore(): void {
  defaultStore = null
}

export function SavedDirectoryProvider({
  children,
  accountKey = null,
  store: injected,
}: {
  children: ReactNode
  accountKey?: string | number | null
  /** Test seam: supply a pre-populated store instead of a fresh one. */
  store?: SavedDirectoryStore
}) {
  const storeRef = useRef<SavedDirectoryStore | null>(injected ?? null)
  if (storeRef.current === null) storeRef.current = new SavedDirectoryStore()
  const store = storeRef.current

  const previousAccount = useRef<string | number | null>(accountKey)
  useEffect(() => {
    if (previousAccount.current === accountKey) return
    previousAccount.current = accountKey
    // Sign-out, sign-in as somebody else, or a forced re-authentication: what
    // the previous session could see says nothing about this one.
    store.clear()
  }, [accountKey, store])

  const value = useMemo<SavedDirectoryContextValue>(() => ({ store }), [store])
  return (
    <SavedDirectoryContext.Provider value={value}>
      {children}
    </SavedDirectoryContext.Provider>
  )
}

export function useSavedDirectoryStore(): SavedDirectoryStore {
  const ctx = useContext(SavedDirectoryContext)
  return ctx?.store ?? getDefaultStore()
}

/** Subscribe to one query's saved/creation record. */
export function useSavedDirectory(queryId: number | null): SavedDirectoryRecord {
  const store = useSavedDirectoryStore()
  return useSyncExternalStore(
    store.subscribe,
    () => store.getRecord(queryId),
    () => store.getRecord(queryId),
  )
}

/**
 * Read this query's record and resolve it from the database while it is still
 * unknown. Model is passed only because the endpoint scores the returned
 * records under one; identity itself does not depend on it, so switching model
 * does not re-ask the question.
 *
 * Keyed on the status rather than run once per mount, so a record left unknown
 * by a create whose outcome could not be established re-checks itself instead
 * of sitting there unresolved. It cannot loop: every answer moves the status
 * off `unknown`.
 */
export function useSavedDirectoryFor(
  queryId: number | null,
  options: { model?: string } = {},
): SavedDirectoryRecord {
  const store = useSavedDirectoryStore()
  const record = useSavedDirectory(queryId)
  const { model } = options
  const status = record.identity.status
  useEffect(() => {
    if (queryId === null) return
    if (status !== 'unknown') return
    void store.ensureLookup(queryId, { model })
  }, [queryId, model, store, status])
  return record
}
