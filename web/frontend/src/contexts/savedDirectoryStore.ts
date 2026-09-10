import {
  createReviewerDir,
  fetchReviewerDirs,
  notifyReviewerDirsUpdated,
  type CreateReviewerDirPayload,
  type ReviewerDir,
} from '../api/reviewerDirs'

/**
 * Durable "does this document already seed a grouping?" state (issue #161).
 *
 * Creating a reviewer directory writes a PERMANENT record: both backend tables
 * are append-only and there is no rename or removal. So the acknowledgement of
 * that write cannot be a by-product of a prediction request. Today it is —
 * `PredictionList` reads `data.seeded_dirs` from the per-model predictions
 * response, and the create itself broadcasts a refresh that clears that cache.
 * If the refresh then fails, or the reviewer switches model or reloads, the app
 * says "not saved" about a row that exists, and offers to create it again.
 *
 * Three deliberate properties:
 *
 * 1. **Identity is keyed by the SEED QUERY, not by a request.** A directory
 *    belongs to the document that seeded it. Which model is selected, whether
 *    its q-q matrix loaded and whether the last predictions call succeeded are
 *    all irrelevant to whether the row exists.
 * 2. **"I don't know" is a state.** `unknown`, `checking`, `absent` and
 *    `unresolved` are distinct. A pending or failed lookup is not an empty
 *    lookup, and neither is an authorisation to offer a fresh Create.
 * 3. **Confirmation is sticky.** Once a POST returns 201, or a seed-filtered
 *    GET returns rows, that fact outlives later empty or failed responses. A
 *    stale empty list cannot un-save a saved grouping.
 *
 * This is NOT another prediction cache. It never reads predictions, never
 * writes feedback and never changes an assessment outcome; issue #156 owns the
 * shared prediction request/error contract and this module deliberately does
 * not duplicate it.
 *
 * It is a plain observable store rather than `useState` because the ordering
 * matters: the saved identity is committed SYNCHRONOUSLY, before
 * `REVIEWER_DIRS_UPDATED_EVENT` invalidates the prediction caches and unmounts
 * the component that issued the create.
 */

/** What we know about groupings seeded by one query. */
export type DirectoryIdentity =
  | { readonly status: 'unknown' }
  /** A seed-filtered GET is in flight and nothing has been confirmed yet. */
  | { readonly status: 'checking' }
  | {
      readonly status: 'saved'
      /**
       * Every directory the server reports for this seed, oldest first.
       * Historical duplicates (created before the atomic-creation safeguard of
       * issue #160) are all listed rather than silently collapsed to one.
       */
      readonly dirs: readonly ReviewerDir[]
      /** Oldest recorded group, matching the backend's own seed lookup. */
      readonly primary: ReviewerDir
      readonly confirmedBy: SavedConfirmation
      /** A refresh of an already-confirmed identity is in flight. */
      readonly refreshing: boolean
      /** The last refresh failed; the identity below it is still known good. */
      readonly refreshError: string | null
    }
  /** The server answered, for this seed, with no directories at all. */
  | { readonly status: 'absent' }
  /** The lookup failed. Whether a grouping exists is genuinely unknown. */
  | { readonly status: 'unresolved'; readonly error: string }

export type SavedConfirmation =
  /** This session's POST returned 201. */
  | 'created'
  /** A create failed and the reconciling GET found the seed already spoken for. */
  | 'recovered'
  /** A seed-filtered GET listed it (reload, remount, explicit refresh). */
  | 'server-list'
  /** A predictions response carried it in `seeded_dirs`. */
  | 'predictions'

/** The naming form, which outlives the component that renders it. */
export type CreationState =
  | { readonly status: 'idle' }
  /** Named but not sent: Cancel here genuinely cancels. */
  | { readonly status: 'editing'; readonly proposedLabel: string }
  /**
   * A POST has been issued. There is nothing left to cancel: aborting the fetch
   * cannot roll back a write the server may already have committed.
   */
  | {
      readonly status: 'pending'
      readonly proposedLabel: string
      readonly operationId: number
    }
  | {
      readonly status: 'failed'
      /** Kept so a retry does not silently rename the reviewer's grouping. */
      readonly proposedLabel: string
      readonly error: string
      /**
       * `not-created`: reconciled against the server, no directory exists for
       * this seed, so retrying creates rather than duplicates.
       * `unknown`: reconciliation itself failed. The write may have landed;
       * the UI must not claim it did not.
       */
      readonly outcome: 'not-created' | 'unknown'
    }

export interface SavedDirectoryRecord {
  readonly identity: DirectoryIdentity
  readonly creation: CreationState
}

export const BLANK_RECORD: SavedDirectoryRecord = Object.freeze({
  identity: Object.freeze({ status: 'unknown' as const }),
  creation: Object.freeze({ status: 'idle' as const }),
})

export type CreateResult =
  | { readonly outcome: 'created'; readonly dir: ReviewerDir }
  /** The create failed but the seed already has a grouping; it is returned. */
  | { readonly outcome: 'recovered'; readonly dirs: readonly ReviewerDir[] }
  /** Reconciled: nothing was written. The proposed name is preserved. */
  | { readonly outcome: 'failed'; readonly error: string }
  /** Neither the write nor its absence could be established. */
  | { readonly outcome: 'unresolved'; readonly error: string }
  /** A POST for this query is already in flight; no second one was sent. */
  | { readonly outcome: 'already-pending' }
  /** This seed is already known saved; creating again is not possible. */
  | {
      readonly outcome: 'already-saved'
      readonly dirs: readonly ReviewerDir[]
    }

export interface CreateDirectoryInput {
  readonly label: string
  readonly model?: string
  readonly variant?: CreateReviewerDirPayload['variant']
}

function messageOf(err: unknown): string {
  if (err instanceof Error && err.message) return err.message
  return 'Could not create the directory'
}

/**
 * Oldest first, ties broken by dir_id, so "the group for this seed" is the same
 * record the backend's own `get_reviewer_dir_by_seed` resolves and does not
 * shuffle between reloads. The rest of the list stays visible either way.
 */
function orderBySeniority(dirs: readonly ReviewerDir[]): ReviewerDir[] {
  return [...dirs].sort((a, b) => {
    if (a.created_at !== b.created_at) return a.created_at < b.created_at ? -1 : 1
    return a.dir_id < b.dir_id ? -1 : a.dir_id > b.dir_id ? 1 : 0
  })
}

export class SavedDirectoryStore {
  private records = new Map<number, SavedDirectoryRecord>()
  private listeners = new Set<() => void>()
  /** Newest lookup per query; older resolutions are ignored. */
  private lookupGeneration = new Map<number, number>()
  private lookupSeq = 0
  private operationSeq = 0

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Stable-identity read, safe to use as a `useSyncExternalStore` snapshot. */
  getRecord = (queryId: number | null): SavedDirectoryRecord => {
    if (queryId === null) return BLANK_RECORD
    return this.records.get(queryId) ?? BLANK_RECORD
  }

  /**
   * Sign-out / account switch. Saved identity is a claim about "the directories
   * this deployment's database holds, as visible to this account", so it must
   * not survive into another account's session. Nothing here is persisted to
   * browser storage for the same reason: a bare numeric query id from another
   * deployment must never be able to assert that this query is saved. Reload
   * recovery is the server's job, not localStorage's.
   */
  clear(): void {
    this.records.clear()
    this.lookupGeneration.clear()
    this.emit()
  }

  // --- lookup --------------------------------------------------------------

  /**
   * Resolve this seed's directories from the database, once.
   *
   * Re-entrant by design: mounting a consumer, remounting it after a prediction
   * refresh and switching model all call this, and only the first does I/O.
   * `force` re-checks an already-answered query without discarding what is
   * already known.
   */
  async ensureLookup(
    queryId: number,
    options: { force?: boolean; model?: string } = {},
  ): Promise<void> {
    const record = this.getRecord(queryId)
    const inFlight =
      record.identity.status === 'checking' ||
      (record.identity.status === 'saved' && record.identity.refreshing)
    if (inFlight) return
    if (!options.force) {
      // An `unresolved` identity is a failed lookup, not an answer: retrying it
      // is the point. `unknown` has never been asked.
      if (record.identity.status === 'saved') return
      if (record.identity.status === 'absent') return
    }

    const generation = ++this.lookupSeq
    this.lookupGeneration.set(queryId, generation)
    this.update(queryId, (current) => ({
      ...current,
      identity:
        current.identity.status === 'saved'
          ? { ...current.identity, refreshing: true }
          : { status: 'checking' },
    }))

    try {
      const dirs = await fetchReviewerDirs({
        seedQueryId: queryId,
        model: options.model,
      })
      if (this.lookupGeneration.get(queryId) !== generation) return
      if (dirs.length > 0) {
        this.recordSaved(queryId, dirs, 'server-list')
        return
      }
      this.recordAbsent(queryId)
    } catch (err) {
      if (this.lookupGeneration.get(queryId) !== generation) return
      const error = err instanceof Error && err.message ? err.message : 'Lookup failed'
      this.update(queryId, (current) => {
        if (current.identity.status === 'saved') {
          // Known-saved survives a failed refresh. This is the case that
          // currently regresses to a fresh Create button.
          return {
            ...current,
            identity: { ...current.identity, refreshing: false, refreshError: error },
          }
        }
        return { ...current, identity: { status: 'unresolved', error } }
      })
    }
  }

  /**
   * Positive evidence from elsewhere — today `PredictionResponse.seeded_dirs`.
   *
   * Deliberately one-directional: a non-empty list confirms, an empty one says
   * nothing. Predictions can be for another model, can fail, and can be served
   * from a cache that predates the write, so absence there is not absence in
   * the database.
   */
  observeSeededDirs(queryId: number, dirs: readonly ReviewerDir[]): void {
    if (dirs.length === 0) return
    const seeded = dirs.filter((dir) => dir.seed_query_id === queryId)
    if (seeded.length === 0) return
    this.recordSaved(queryId, seeded, 'predictions')
  }

  // --- creation ------------------------------------------------------------

  /** Open the naming form, or re-open it on the name already proposed. */
  beginNaming(queryId: number, suggestion: string): void {
    const record = this.getRecord(queryId)
    if (record.creation.status === 'pending') return
    const proposedLabel =
      record.creation.status === 'editing' || record.creation.status === 'failed'
        ? record.creation.proposedLabel
        : suggestion
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'editing', proposedLabel },
    }))
  }

  /** Type into the name field. Ignored once the write has been issued. */
  setProposedLabel(queryId: number, proposedLabel: string): void {
    const record = this.getRecord(queryId)
    if (record.creation.status === 'pending') return
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'editing', proposedLabel },
    }))
  }

  /**
   * Cancel the naming form. Returns false — and changes nothing — once a POST
   * has been issued, because at that point there is no client-side action that
   * undoes the write. The caller must disable the control or relabel it as
   * closing the form rather than cancelling the creation.
   */
  cancelNaming(queryId: number): boolean {
    const record = this.getRecord(queryId)
    if (record.creation.status === 'pending') return false
    this.update(queryId, (current) => ({ ...current, creation: { status: 'idle' } }))
    return true
  }

  /**
   * Close the form over an in-flight write WITHOUT claiming a rollback. The
   * request keeps running and still records its result against this query; only
   * the form is dismissed, and a late failure will not re-open it.
   */
  dismissPendingForm(queryId: number): void {
    const record = this.getRecord(queryId)
    if (record.creation.status !== 'pending') return
    this.update(queryId, (current) => ({ ...current, creation: { status: 'idle' } }))
  }

  /**
   * Create the grouping for `queryId`.
   *
   * The result is recorded against THAT query, whatever is on screen when the
   * response lands: a reviewer who navigates away mid-flight has still created
   * a permanent directory, and dropping the response (what the CTA does today)
   * loses the only acknowledgement of it.
   *
   * On any failure the seed is reconciled with a GET rather than by reading the
   * 409's prose. The message text is not a contract, an aborted or lost
   * response is indistinguishable from a refusal at this layer, and the
   * database is the only authority on whether the row exists. (When issue #156
   * lands a typed HTTP status on the shared client, that status can shortcut
   * the reconciliation for failures which certainly wrote nothing; it cannot
   * replace it, because a lost 201 has no status at all.)
   */
  async createDirectory(
    queryId: number,
    input: CreateDirectoryInput,
  ): Promise<CreateResult> {
    const record = this.getRecord(queryId)
    if (record.creation.status === 'pending') return { outcome: 'already-pending' }
    if (record.identity.status === 'saved') {
      return { outcome: 'already-saved', dirs: record.identity.dirs }
    }

    const proposedLabel = input.label
    const operationId = ++this.operationSeq
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'pending', proposedLabel, operationId },
    }))

    try {
      const dir = await createReviewerDir(
        {
          query_file_id: queryId,
          label: proposedLabel.trim() || undefined,
          model_slug: input.model || undefined,
          variant: input.variant,
        },
        // Record first, broadcast second. The broadcast clears every prediction
        // cache and re-renders the list into its loading branch, so anything
        // that only lived in the creating component is gone by the time the
        // refresh result (successful or not) arrives.
        { notify: false },
      )
      this.recordSaved(queryId, [dir], 'created')
      this.settleOperation(queryId, operationId, { status: 'idle' })
      notifyReviewerDirsUpdated()
      return { outcome: 'created', dir }
    } catch (err) {
      const error = messageOf(err)
      let reconciled: ReviewerDir[] | null = null
      try {
        reconciled = await fetchReviewerDirs({
          seedQueryId: queryId,
          model: input.model,
        })
      } catch {
        reconciled = null
      }

      if (reconciled && reconciled.length > 0) {
        const wasKnown = this.getRecord(queryId).identity.status === 'saved'
        this.recordSaved(queryId, reconciled, 'recovered')
        this.settleOperation(queryId, operationId, { status: 'idle' })
        // A recovered directory the app had not seen is new to every candidate
        // list in this tab. Already-known ones broadcast nothing, so recovery
        // cannot feed a refresh loop.
        if (!wasKnown) notifyReviewerDirsUpdated()
        return { outcome: 'recovered', dirs: reconciled }
      }

      if (reconciled) {
        // The database says this seed is free, so the write certainly did not
        // land: a retry creates rather than duplicates.
        this.recordAbsent(queryId)
        this.settleOperation(queryId, operationId, {
          status: 'failed',
          proposedLabel,
          error,
          outcome: 'not-created',
        })
        return { outcome: 'failed', error }
      }

      this.settleOperation(queryId, operationId, {
        status: 'failed',
        proposedLabel,
        error,
        outcome: 'unknown',
      })
      return { outcome: 'unresolved', error }
    }
  }

  /** Clear a failed attempt's error without discarding the proposed name. */
  retryNaming(queryId: number): void {
    const creation = this.getRecord(queryId).creation
    if (creation.status !== 'failed') return
    const { proposedLabel } = creation
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'editing', proposedLabel },
    }))
  }

  // --- internals -----------------------------------------------------------

  /**
   * A confirmed save is a durable fact about the database, so unlike the
   * transient form state it is applied whatever generation it arrives in. The
   * label stored is the one the SERVER holds, which for a recovery is somebody
   * else's wording rather than the name this reviewer proposed.
   */
  private recordSaved(
    queryId: number,
    dirs: readonly ReviewerDir[],
    confirmedBy: SavedConfirmation,
  ): void {
    const ordered = orderBySeniority(dirs)
    const primary = ordered[0]
    if (!primary) return
    this.update(queryId, (current) => ({
      ...current,
      identity: {
        status: 'saved',
        dirs: ordered,
        primary,
        confirmedBy,
        refreshing: false,
        refreshError: null,
      },
    }))
  }

  /**
   * "The server answered, and this seed has no directories."
   *
   * Sticky in the other direction: a confirmed save is never downgraded by a
   * later empty answer, because an empty answer can be stale (a lookup issued
   * before the write) while a 201 cannot be.
   */
  private recordAbsent(queryId: number): void {
    this.update(queryId, (current) => {
      if (current.identity.status === 'saved') {
        return {
          ...current,
          identity: { ...current.identity, refreshing: false, refreshError: null },
        }
      }
      return { ...current, identity: { status: 'absent' } }
    })
  }

  /**
   * Transient creation state belongs to one attempt. A late failure from an
   * abandoned or superseded attempt must not overwrite a newer pending write,
   * a dismissed form, or a save that has since been confirmed.
   */
  private settleOperation(
    queryId: number,
    operationId: number,
    creation: CreationState,
  ): void {
    const current = this.getRecord(queryId)
    if (
      current.creation.status !== 'pending' ||
      current.creation.operationId !== operationId
    ) {
      return
    }
    this.update(queryId, (record) => ({ ...record, creation }))
  }

  private update(
    queryId: number,
    mutate: (current: SavedDirectoryRecord) => SavedDirectoryRecord,
  ): void {
    const current = this.records.get(queryId) ?? BLANK_RECORD
    const next = mutate(current)
    if (next === current) return
    this.records.set(queryId, next)
    this.emit()
  }

  private emit(): void {
    for (const listener of this.listeners) listener()
  }
}
