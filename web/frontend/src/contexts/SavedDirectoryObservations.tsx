import { useEffect, useRef } from 'react'
import { usePredictionState } from './PredictionContext'
import { useSavedDirectoryFor, useSavedDirectoryStore } from './SavedDirectoryContext'
import type { SavedDirectoryStore } from './savedDirectoryStore'

/**
 * The one bridge from the shared prediction state into the saved-directory
 * store (issue #161).
 *
 * WHY IT IS NOT IN `PredictionList`
 *
 * Scoping the store to an account isolates the store's OWN asynchronous work:
 * a lookup or a create issued by the previous session resolves into a store
 * nothing is subscribed to. It says nothing about evidence that arrives from
 * somewhere else. `PredictionProvider` is mounted above the auth gates and its
 * request key is query/model/variant, with no account in it, so a sign-out
 * followed by a sign-in leaves the PREVIOUS session's settled ranking, its
 * `seeded_dirs` and its cache in place. A component that simply folds
 * `seeded_dirs` into "the current store" therefore hands the new account the
 * old one's snapshot the moment the store is replaced.
 *
 * What is wrong with that is provenance, not privacy. Reviewer directories are
 * global: `web/services/reviewer_dirs.py` serves the same rows to everybody,
 * so a document grouped for one reviewer is grouped for all of them, and the
 * same groups may and should reappear here as soon as this account's own
 * request answers. The objection is that the durable record is supposed to hold
 * what the DATABASE said to THIS session. Replaying another session's response
 * into it makes an unverified claim durable, and durable is the whole point of
 * the record: it outlives the response it came from, survives failures, and is
 * what the UI cites when it refuses to offer Create again.
 *
 * HOW ADMISSION WORKS
 *
 * On adopting a store -- at mount, and again during the render in which the
 * account changes -- this component does two things:
 *
 * 1. It notes the generation on display. Everything at or below it was fetched
 *    before this account was established.
 * 2. It calls #156's own `refresh()`, once. That clears the shared prediction
 *    cache and supersedes anything in flight, so no request issued from here on
 *    can be answered out of the previous session's cache.
 *
 * After that, a settled response with a HIGHER generation is necessarily this
 * account's own answer, and its positive `seeded_dirs` are admitted. Nothing
 * else is: an inherited snapshot is simply not ingested, and a late response
 * from before the switch is dropped by the shared hook itself.
 *
 * This is not a second prediction cache, a second generation counter or a
 * second request. It reads #156's key, generation, phase and `seeded_dirs`, and
 * uses #156's refresh contract. Everything downstream -- the CTA, the
 * acknowledgement, the badge -- reads the store, so there is exactly one
 * admission point rather than one per component that happens to render a
 * directory.
 *
 * It renders nothing, and is mounted for the whole authenticated lifetime, so
 * navigating to another view or losing the prediction list does not reset
 * admission or drop evidence.
 *
 * It also never announces anything. Only the store announces an owned write,
 * exactly once, after the identity is recorded; feeding it observations cannot
 * manufacture or consume that event, and a repeated observation of the same
 * rows is a no-op, so ordinary reads stay quiet and cannot drive a refresh
 * loop.
 */

interface Admission {
  readonly store: SavedDirectoryStore
  /**
   * Generation exposed when this store was adopted. A settled response at or
   * below it predates this account and is not evidence for it.
   */
  readonly baseline: number
  /** Whether the cache-clearing refresh for this store has been issued. */
  invalidated: boolean
}

export default function SavedDirectoryObservations() {
  const store = useSavedDirectoryStore()
  const { key, generation, phase, seededDirs, refresh } = usePredictionState()

  /**
   * ASKING THE QUESTION, which is separate from admitting an answer to it.
   *
   * "Does this document already seed a grouping?" is answered by its own
   * endpoint, and that endpoint has nothing to do with a ranking. Until now the
   * only production caller that started the lookup was inside `NewDirectoryCta`,
   * so the question was asked only where a creation offer happened to be
   * rendered. After a reload that makes the answer depend on the ranking: the
   * record starts unknown, neither CTA position mounts while predictions are
   * loading or have failed, nothing asks, and the durable acknowledgement for a
   * directory that plainly exists stays invisible until some model's ranking
   * recovers -- with the directory endpoint healthy the whole time.
   *
   * Asking from here instead fixes that at the root. This component is mounted
   * for the whole authenticated lifetime, above the request branches, so the
   * lookup follows the reviewer's current document rather than the visibility
   * of a button, and survives the list unmounting.
   *
   * It does not relax anything about OFFERING creation. Whether Create is
   * available is still decided downstream from evidence -- a settled ranking
   * and an `absent` identity -- and this only ensures the identity half is
   * actually established. The two outcomes it adds are both honest: a saved
   * grouping is acknowledged through a failed ranking, and a failed lookup is
   * reported as unresolved with a retry instead of as an empty question.
   *
   * `useSavedDirectoryFor` is the existing shared mechanism for this and is
   * re-entrant: it asks only while the status is `unknown`, and `ensureLookup`
   * drops a call that races one already in flight. So this does not duplicate
   * the CTA's request, and it cannot loop, because every answer moves the
   * status off `unknown`.
   */
  useSavedDirectoryFor(key?.queryId ?? null, { model: key?.model })

  // Adopted DURING RENDER, like the store itself. An effect would leave one
  // commit in which the new account's store is current and the old account's
  // baseline is not, which is the commit the stale snapshot would be ingested
  // in.
  const admissionRef = useRef<Admission | null>(null)
  if (admissionRef.current === null || admissionRef.current.store !== store) {
    admissionRef.current = { store, baseline: generation, invalidated: false }
  }
  const admission = admissionRef.current

  // Declared before the observation effect so it runs before it: within one
  // commit React runs a component's effects in order, and admission must be
  // closed until this has run.
  useEffect(() => {
    if (admission.invalidated) return
    admission.invalidated = true
    // Free when nothing is selected, which is the normal sign-in path: the
    // shared hook issues no request for a null key and only the cache is
    // dropped. It costs one refetch when an account changes with a document
    // already on screen, which is precisely the case that needs one.
    refresh()
  }, [admission, refresh])

  // `loading` and `error` are statements about a request, not about the
  // database, and the store is told nothing during either.
  const settled = phase === 'ready' || phase === 'empty' || phase === 'excluded'
  // Read from the ref rather than from state on purpose: the effect above sets
  // it and immediately calls `refresh()`, whose update re-renders this
  // component, and the second half of the test cannot be satisfied before a
  // post-refresh response settles several renders later. There is no render in
  // between that could read it stale and matter.
  const admitted = admission.invalidated && generation > admission.baseline
  // Recorded against the query THE RESPONSE speaks for, never against whatever
  // is on screen by the time it lands.
  const observedQueryId = admitted && settled && key !== null ? key.queryId : null

  useEffect(() => {
    if (observedQueryId === null) return
    // The store ignores an empty list and rows seeded by another query, so a
    // failed, empty or stale ranking can never erase a saved identity, its
    // historical groups or their membership. Only positive evidence moves it.
    store.observeSeededDirs(observedQueryId, seededDirs)
  }, [store, observedQueryId, generation, seededDirs])

  return null
}
