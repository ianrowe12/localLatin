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
 * deployment. Reviewer directories are global -- the same rows are served to
 * every reviewer, and they may legitimately reappear for the next account -- but
 * this record holds what the database said to ONE authenticated session, along
 * with that session's unfinished writes and unanswered lookups. Replaying it
 * into another session would make evidence nobody re-established durable under
 * a new name, so pass the current account id (for example
 * `useReviewer().user?.id`) when mounting this provider.
 *
 * MOUNTING IT IS NOT OPTIONAL: the hooks throw without it rather than falling
 * back to a store of their own. A per-component fallback would type-check and
 * silently reinstate the bug this module exists to fix; a process-wide one
 * would leak one reviewer's state into the next session. `App` mounts it above
 * `PredictionProvider`; any other surface that renders `NewDirectoryCta` has to
 * wrap it too.
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
  // Scoped DURING RENDER, not in an effect. An effect runs after the children
  // have already painted, so a sign-out or a switch to another account would
  // show the previous session's acknowledgements for one frame, sourced from a
  // session that is over. A new account therefore gets a new store rather than
  // a cleaned one: the old records are unreachable instead of merely emptied,
  // and a request still in flight from the old session resolves into a store
  // nothing is subscribed to. What the new account can see, it re-establishes
  // from its own lookup and its own ranking -- see
  // `SavedDirectoryObservations`, which is what keeps the previous session's
  // prediction snapshot out of this one.
  const scopedRef = useRef<{
    accountKey: string | number | null
    store: SavedDirectoryStore
  } | null>(null)
  if (scopedRef.current === null) {
    scopedRef.current = { accountKey, store: injected ?? new SavedDirectoryStore() }
  } else if (scopedRef.current.accountKey !== accountKey) {
    scopedRef.current = { accountKey, store: new SavedDirectoryStore() }
  }
  const store = scopedRef.current.store

  const value = useMemo<SavedDirectoryContextValue>(() => ({ store }), [store])
  return (
    <SavedDirectoryContext.Provider value={value}>
      {children}
    </SavedDirectoryContext.Provider>
  )
}

/**
 * REQUIRES a mounted `SavedDirectoryProvider`, and says so rather than
 * inventing a store.
 *
 * A component-lifetime fallback would satisfy every type and reinstate exactly
 * the bug issue #161 exists to fix, silently. A process-wide one would satisfy
 * the tests and leave one reviewer's account state readable by the next. So the
 * durability of a permanent write is not allowed to depend on remembering to
 * mount a provider: forgetting it is loud.
 */
export function useSavedDirectoryStore(): SavedDirectoryStore {
  const ctx = useContext(SavedDirectoryContext)
  if (ctx === null) {
    throw new Error(
      'useSavedDirectory* requires a <SavedDirectoryProvider>. Mount it above the prediction list so a created directory survives the refresh that unmounts the component which created it.',
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
