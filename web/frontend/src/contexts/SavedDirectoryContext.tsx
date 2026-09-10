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
  if (!ctx) {
    throw new Error(
      'useSavedDirectoryStore must be used within a SavedDirectoryProvider',
    )
  }
  return ctx.store
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
 * Read this query's record and resolve it from the database if it has never
 * been resolved. Model is passed only because the endpoint scores the returned
 * records under one; identity itself does not depend on it, and a lookup is not
 * repeated when the reviewer switches model.
 */
export function useSavedDirectoryFor(
  queryId: number | null,
  options: { model?: string } = {},
): SavedDirectoryRecord {
  const store = useSavedDirectoryStore()
  const record = useSavedDirectory(queryId)
  const { model } = options
  useEffect(() => {
    if (queryId === null) return
    void store.ensureLookup(queryId, { model })
  }, [queryId, model, store])
  return record
}
