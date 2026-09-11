import {
  createReviewerDir,
  fetchReviewerDirs,
  isCompleteReviewerDir,
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

/**
 * The naming form's OPERATION, which outlives the form itself.
 *
 * Deliberately not the same thing as "is the form on screen". A reviewer who
 * closes a form has hidden a panel; they have not recalled a write, cancelled a
 * permanent record or forgotten the name they chose. Conflating the two is how
 * closing a pending form released the single-write lock and let a second POST
 * go out for one document (issue #161 review, finding 1).
 */
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
       * the UI must not claim it did not, and must not offer to send another.
       */
      readonly outcome: 'not-created' | 'unknown'
    }

/**
 * True while the outcome of an issued write is genuinely unknown. Editing the
 * name, submitting again and discarding the proposal are all refused in this
 * state, because each of them acts on an answer nobody has.
 */
export function isWriteUnsettled(creation: CreationState): boolean {
  return (
    creation.status === 'pending' ||
    (creation.status === 'failed' && creation.outcome === 'unknown')
  )
}

export interface SavedDirectoryRecord {
  readonly identity: DirectoryIdentity
  readonly creation: CreationState
  /** Whether the naming panel is on screen. Never a fact about the write. */
  readonly formOpen: boolean
}

export const BLANK_RECORD: SavedDirectoryRecord = Object.freeze({
  identity: Object.freeze({ status: 'unknown' as const }),
  creation: Object.freeze({ status: 'idle' as const }),
  formOpen: false,
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
  /**
   * A previous write's outcome is still unknown, so a second one is refused:
   * it could be the duplicate that a lost 201 already created.
   */
  | { readonly outcome: 'blocked-unresolved' }
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

/**
 * Fold new evidence about one directory into what is already confirmed, never
 * subtracting.
 *
 * Both backend tables are append-only, so every fact this app has ever been
 * told about a directory stays true: a group that existed still exists, a
 * document filed into it is still filed into it, and `matched` -- which
 * `web/services/reviewer_dirs.py` derives from a human having filed a second
 * distinct witness -- cannot revert to `awaiting_match`. Responses, by
 * contrast, are snapshots that can arrive out of order: an older seed lookup
 * can resolve after a newer predictions payload, and a predictions payload can
 * be served from a cache built before the second witness was filed. Overwriting
 * with the newest arrival therefore deletes history for no reason (review
 * finding 4).
 *
 * The score fields are the exception and take the incoming value: they are
 * recomputed server-side against the selected model's q-q matrix, are
 * informational, and never decide status.
 */
function mergeDir(known: ReviewerDir, incoming: ReviewerDir): ReviewerDir {
  const members = new Set<number>([
    ...known.member_query_ids,
    ...incoming.member_query_ids,
  ])
  return {
    ...known,
    ...incoming,
    status:
      known.status === 'matched' || incoming.status === 'matched'
        ? 'matched'
        : 'awaiting_match',
    member_query_ids: [...members].sort((a, b) => a - b),
    best_match_score: incoming.best_match_score ?? known.best_match_score,
  }
}

/** Union by `dir_id`: a group this app has confirmed is never dropped. */
function mergeDirs(
  known: readonly ReviewerDir[],
  incoming: readonly ReviewerDir[],
): ReviewerDir[] {
  const byId = new Map<string, ReviewerDir>()
  for (const dir of known) byId.set(dir.dir_id, dir)
  for (const dir of incoming) {
    const existing = byId.get(dir.dir_id)
    byId.set(dir.dir_id, existing ? mergeDir(existing, dir) : dir)
  }
  return orderBySeniority([...byId.values()])
}

/**
 * Whether a merge actually told us anything new.
 *
 * The predictions response repeats a query's `seeded_dirs` on every settled
 * request, so the same positive evidence is re-observed after each refresh and
 * each model switch. Folding it in is correct; re-publishing an identical
 * record afterwards is not, because every subscriber re-renders for it. Compared
 * field by field rather than by reference, because `mergeDirs` always builds a
 * new array.
 */
function sameDirs(a: readonly ReviewerDir[], b: readonly ReviewerDir[]): boolean {
  if (a.length !== b.length) return false
  return a.every((dir, index) => {
    const other = b[index]
    return (
      dir.dir_id === other.dir_id &&
      dir.label === other.label &&
      dir.status === other.status &&
      dir.created_by === other.created_by &&
      dir.created_at === other.created_at &&
      dir.seed_query_id === other.seed_query_id &&
      dir.best_match_score === other.best_match_score &&
      dir.member_query_ids.length === other.member_query_ids.length &&
      dir.member_query_ids.every((id, i) => id === other.member_query_ids[i])
    )
  })
}

/**
 * `created` and `recovered` are first-hand facts about THIS reviewer's attempt,
 * so a later routine refresh does not overwrite them: the acknowledgement would
 * otherwise stop saying where the grouping came from halfway through a session.
 */
function mergeConfirmation(
  known: SavedConfirmation,
  incoming: SavedConfirmation,
): SavedConfirmation {
  if (known === 'created' || known === 'recovered') return known
  return incoming
}

export class SavedDirectoryStore {
  private records = new Map<number, SavedDirectoryRecord>()
  private listeners = new Set<() => void>()
  /** Newest lookup per query; older resolutions are ignored. */
  private lookupGeneration = new Map<number, number>()
  private lookupSeq = 0
  private operationSeq = 0
  /**
   * How many write attempts have been STARTED for a query, and how many are
   * still in flight.
   *
   * A lookup's empty answer only authorises a fresh Create if it describes a
   * world in which no write of ours could have been racing it. Generations
   * alone cannot see that: they only change when another LOOKUP starts, so a
   * pre-write GET happily resolved a post-write uncertainty as "no directory
   * here" (review finding 5).
   */
  private writeSeq = new Map<number, number>()
  private writesInFlight = new Map<number, number>()
  /**
   * Has the write this store issued for `queryId` been announced yet?
   *
   * The broadcast is a fact about THIS STORE'S OWN HISTORY -- did the refresh
   * event for the directory we wrote go out -- and nothing about the identity
   * record answers it. `observeSeededDirs` deliberately records a directory
   * without announcing anything, so "the client can already name this row" is
   * not evidence that the caches were told about it; reading the announcement
   * out of the identity is what let a seeded observation arriving before the
   * 201 swallow the only notification the write was ever going to send.
   *
   * A query is added when a write is issued and removed when that write is
   * announced, or when the write is proven not to have landed.
   */
  private readonly unannouncedWrites = new Set<number>()

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
   * Sign-out / account switch. Saved identity is a claim about what this
   * deployment's database told ONE authenticated session, so it must not
   * survive into another one. The rows themselves are global and may
   * legitimately reappear; what may not carry over is the previous session's
   * evidence for them, its unfinished writes and its unanswered lookups.
   * Nothing here is persisted to browser storage for the same reason: a bare
   * numeric query id from another deployment must never be able to assert that
   * this query is saved. Reload recovery is the server's job, not
   * localStorage's.
   *
   * `SavedDirectoryProvider` does not call this on an account change; it builds
   * a new store instead, so the old records are unreachable rather than emptied
   * and a request still in flight from the previous session cannot land in the
   * new one. This stays for a host that owns one store across a lifecycle event
   * of its own.
   */
  clear(): void {
    this.records.clear()
    this.lookupGeneration.clear()
    this.writeSeq.clear()
    this.writesInFlight.clear()
    this.unannouncedWrites.clear()
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
    // Captured before the request goes out. An empty answer is only permission
    // to create if no write of ours could have been in flight beside it.
    const writesAtStart = this.writeSeq.get(queryId) ?? 0
    const racedAWrite = (this.writesInFlight.get(queryId) ?? 0) > 0
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
        // Positive evidence is always usable: a row that exists, exists,
        // whenever the response was built. If this lookup is the last step of a
        // creation -- typically a `Check again` after a lost response -- that is
        // `confirmSaved`'s business, not this branch's.
        this.confirmSaved(queryId, dirs, 'server-list')
        return
      }
      // Only a well-formed, empty list gets here: `fetchReviewerDirs` throws on
      // a body it cannot read, so absence is always something the server said
      // rather than something this client inferred.
      const writesSince = (this.writeSeq.get(queryId) ?? 0) !== writesAtStart
      const stillWriting = (this.writesInFlight.get(queryId) ?? 0) > 0
      if (racedAWrite || writesSince || stillWriting) {
        // This answer describes a world the reviewer has since acted on. It is
        // not proof that the write did not land, so it must not put a Create
        // button back in front of them.
        this.markUnresolved(
          queryId,
          'The app checked before your last attempt finished, so this answer cannot settle it.',
          { onlyWhileChecking: true },
        )
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
   *
   * The rows still have to be READABLE as directories. They arrive through the
   * shared prediction validator (issue #156), whose contract for the same rows
   * is deliberately laxer than this one: it tolerates a missing
   * `member_query_ids`, `created_at` or `created_by` and fills in `[]` and `''`
   * so an older backend still renders a candidate in a ranking. Those
   * substitutes are fine for a candidate and are not evidence of a permanent
   * record: promoting them here writes a blank creator and a blank timestamp
   * into the durable identity, where the blank timestamp sorts first and
   * silently renames which group the document is filed under, and `[]` reads as
   * "no human has filed a second witness" for a group where one has.
   *
   * So the same completeness contract the lookup and the 201 body are held to
   * is applied here, and a list that fails it is NOT evidence in either
   * direction. It confirms nothing, erases nothing, settles no outstanding
   * write and announces nothing -- the question simply stays open, and every
   * recovery path (`Check again`, the next lookup, the next good response)
   * stays exactly as reachable as it was.
   */
  observeSeededDirs(queryId: number, dirs: readonly ReviewerDir[]): void {
    if (dirs.length === 0) return
    const seeded = dirs.filter((dir) => dir.seed_query_id === queryId)
    if (seeded.length === 0) return
    // Rejected whole rather than filtered, exactly as `assertReviewerDirList`
    // rejects a list containing one bad row. Confirming the readable subset
    // would record a partial view of this document's groups as though it were
    // the answer.
    if (!seeded.every(isCompleteReviewerDir)) return
    // Positive, seed-matched, complete and server-sourced: the same class of
    // evidence a lookup returns, so it finishes an outstanding write of ours
    // exactly as a lookup would. Recording the identity and stopping short of
    // that is what stranded a reviewer with an acknowledgement, no `Check
    // again` and a refresh nobody would ever send.
    this.confirmSaved(queryId, seeded, 'predictions')
  }

  // --- creation ------------------------------------------------------------

  /**
   * Open the naming panel, on the name already proposed if there is one.
   *
   * Never resets a proposal. A reviewer who closed the panel over a failure and
   * re-opened it is continuing one act of naming, and retyping a name is how a
   * second grouping ends up under a near-duplicate of the first one's label.
   */
  beginNaming(queryId: number, suggestion: string): void {
    const { creation } = this.getRecord(queryId)
    const proposedLabel =
      creation.status === 'idle' ? suggestion : creation.proposedLabel
    this.update(queryId, (current) => ({
      ...current,
      formOpen: true,
      creation:
        current.creation.status === 'idle'
          ? { status: 'editing', proposedLabel }
          : current.creation,
    }))
  }

  /**
   * Type into the name field.
   *
   * Refused while the outcome of an issued write is unknown. Editing there
   * would replace the failure with a clean `editing` state, taking the
   * uncertainty explanation off screen and re-arming a submit that might
   * duplicate a directory the server has already stored.
   */
  setProposedLabel(queryId: number, proposedLabel: string): void {
    const { creation } = this.getRecord(queryId)
    if (isWriteUnsettled(creation)) return
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'editing', proposedLabel },
    }))
  }

  /**
   * Cancel the naming form.
   *
   * Only an unsent form can be cancelled. Once a write has been issued this
   * closes the panel and returns false, keeping the operation and the proposed
   * name: there is no client-side action that undoes a write, and discarding
   * the name while the outcome is unknown loses the one piece of context a
   * reviewer needs to decide what to do next.
   */
  cancelNaming(queryId: number): boolean {
    const { creation } = this.getRecord(queryId)
    if (isWriteUnsettled(creation)) {
      this.closeForm(queryId)
      return false
    }
    this.update(queryId, (current) => ({
      ...current,
      formOpen: false,
      creation: { status: 'idle' },
    }))
    return true
  }

  /**
   * Close the panel over an issued write WITHOUT claiming a rollback.
   *
   * The request keeps running and still records its result against this query.
   * Crucially the operation itself is untouched, so the single-write lock holds
   * and the proposed name survives: hiding a panel is not settling a write
   * (review finding 1).
   */
  dismissPendingForm(queryId: number): void {
    this.closeForm(queryId)
  }

  /** Hide the naming panel, leaving every fact about the write in place. */
  private closeForm(queryId: number): void {
    this.update(queryId, (current) =>
      current.formOpen ? { ...current, formOpen: false } : current,
    )
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
    if (isWriteUnsettled(record.creation)) {
      // The previous attempt may already have written the row. Sending another
      // one on the strength of not knowing is exactly how a document ends up
      // with two permanent groupings; the reviewer has to resolve the question
      // first, which `ensureLookup(force)` does.
      return { outcome: 'blocked-unresolved' }
    }

    const proposedLabel = input.label
    const operationId = ++this.operationSeq
    this.writeSeq.set(queryId, (this.writeSeq.get(queryId) ?? 0) + 1)
    this.writesInFlight.set(queryId, (this.writesInFlight.get(queryId) ?? 0) + 1)
    // From here until this write is announced or disproved, whichever path
    // learns it landed owes the rest of the app a refresh.
    this.unannouncedWrites.add(queryId)
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
      this.writeSettled(queryId)
      this.confirmSaved(queryId, [dir], 'created')
      return { outcome: 'created', dir }
    } catch (err) {
      const error = messageOf(err)
      // Counted as settled before the reconciling GET goes out, so that GET is
      // allowed to establish absence: this request is over, whatever it did.
      this.writeSettled(queryId)
      let reconciled: ReviewerDir[] | null = null
      try {
        // A malformed answer throws here, which is what keeps it out of the
        // `reconciled.length === 0` branch below: that branch is a claim that
        // the write did not land.
        reconciled = await fetchReviewerDirs({
          seedQueryId: queryId,
          model: input.model,
        })
      } catch {
        reconciled = null
      }

      if (reconciled && reconciled.length > 0) {
        this.confirmSaved(queryId, reconciled, 'recovered')
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

      this.markUnresolved(queryId, error)
      this.settleOperation(queryId, operationId, {
        status: 'failed',
        proposedLabel,
        error,
        outcome: 'unknown',
      })
      return { outcome: 'unresolved', error }
    }
  }

  /**
   * Clear a RESOLVED failure's error without discarding the proposed name.
   *
   * Refused while the outcome is unknown, for the same reason editing is: a
   * clean form invites a second write that could duplicate a stored directory.
   */
  retryNaming(queryId: number): void {
    const creation = this.getRecord(queryId).creation
    if (creation.status !== 'failed' || creation.outcome !== 'not-created') return
    const { proposedLabel } = creation
    this.update(queryId, (current) => ({
      ...current,
      creation: { status: 'editing', proposedLabel },
    }))
  }

  // --- internals -----------------------------------------------------------

  /** One issued write is over, whatever it managed to do. */
  private writeSettled(queryId: number): void {
    const inFlight = this.writesInFlight.get(queryId) ?? 0
    if (inFlight <= 1) this.writesInFlight.delete(queryId)
    else this.writesInFlight.set(queryId, inFlight - 1)
  }

  /**
   * Emit the refresh for an owned write, at most once per write.
   *
   * Callers must have recorded the identity first: the refresh clears every
   * prediction cache and re-renders the list into its loading branch, so a
   * listener has to be able to see the acknowledgement the moment it wakes up.
   */
  private announceWrite(queryId: number): void {
    if (!this.unannouncedWrites.delete(queryId)) return
    notifyReviewerDirsUpdated()
  }

  /**
   * The one way this store accepts positive evidence: record it, finish any
   * write of ours it answers, tell the rest of the app.
   *
   * Every source of a confirmed directory goes through here -- the 201, the
   * reconciliation after a failed POST, an explicit lookup, and a `seeded_dirs`
   * observation -- because they are four ways of learning the SAME thing, and a
   * reviewer's recovery must not depend on which one happens to arrive. When a
   * seed observation was allowed to record identity alone, it produced a state
   * with no exit: the CTA's saved branch hid `Check again`, the identity was
   * no longer unknown so nothing would look it up again, and the write stayed
   * unannounced with no reachable action left to announce it. The write was
   * acknowledged on screen while every other cached query went on omitting it.
   *
   * The two gates are unchanged and both live in one place now. Settling only
   * touches a write that is genuinely unsettled, and `announceWrite` only
   * speaks if this store owes a refresh, so an ordinary read or observation --
   * no write of ours outstanding -- still passes through silently.
   */
  private confirmSaved(
    queryId: number,
    dirs: readonly ReviewerDir[],
    confirmedBy: SavedConfirmation,
  ): void {
    this.recordSaved(queryId, dirs, confirmedBy)
    // The outcome nobody could establish is established: it landed. A late
    // failure from that same attempt can no longer reopen the form, because
    // `settleOperation` only writes over a pending operation of its own id.
    this.update(queryId, (current) =>
      isWriteUnsettled(current.creation)
        ? { ...current, creation: { status: 'idle' }, formOpen: false }
        : current,
    )
    this.announceWrite(queryId)
  }

  /**
   * A confirmed save is a durable fact about the database, so unlike the
   * transient form state it is applied whatever generation it arrives in. The
   * label stored is the one the SERVER holds, which for a recovery is somebody
   * else's wording rather than the name this reviewer proposed.
   *
   * Evidence is FOLDED IN, never substituted (review finding 4): responses
   * arrive out of order and some of them are partial views, so a snapshot that
   * mentions one group is not a statement that the others were removed -- and
   * nothing can remove them. See `mergeDirs`.
   */
  private recordSaved(
    queryId: number,
    dirs: readonly ReviewerDir[],
    confirmedBy: SavedConfirmation,
  ): void {
    if (dirs.length === 0) return
    this.update(queryId, (current) => {
      const known = current.identity.status === 'saved' ? current.identity.dirs : []
      const merged = mergeDirs(known, dirs)
      const primary = merged[0]
      if (!primary) return current
      const confirmation =
        current.identity.status === 'saved'
          ? mergeConfirmation(current.identity.confirmedBy, confirmedBy)
          : confirmedBy
      if (
        current.identity.status === 'saved' &&
        current.identity.confirmedBy === confirmation &&
        current.identity.refreshing === false &&
        current.identity.refreshError === null &&
        sameDirs(current.identity.dirs, merged)
      ) {
        // Nothing new. Re-publishing the record here would re-render every
        // subscriber for each repetition of the same `seeded_dirs`.
        return current
      }
      return {
        ...current,
        identity: {
          status: 'saved',
          dirs: merged,
          primary,
          confirmedBy: confirmation,
          refreshing: false,
          refreshError: null,
        },
      }
    })
  }

  /**
   * "The server answered, for this seed, with no directories."
   *
   * Sticky in two directions. A confirmed save is never downgraded, because an
   * empty answer can be stale (a lookup issued before the write) while a 201
   * cannot be. And an unsettled write is never resolved by it: the caller is
   * responsible for not calling this with an answer that raced a write, but the
   * guard is repeated here because "absent" is the one state that hands out
   * permission to write again.
   *
   * A resolved absence does settle a previous failure whose outcome was
   * unknown: the database has now been read cleanly with nothing in flight, so
   * that write demonstrably did not land and the reviewer can retry the name
   * they still have.
   */
  private recordAbsent(queryId: number): void {
    // Measured in outstanding REQUESTS, not in form state: a write that has
    // returned an error is over even though its record still says `pending`
    // until the reconciliation it triggered comes back.
    if ((this.writesInFlight.get(queryId) ?? 0) > 0) return
    if (this.getRecord(queryId).identity.status !== 'saved') {
      // The server has been read cleanly with nothing in flight and this seed
      // has no directories, so any write of ours did not land. There is no new
      // row for the caches to hear about.
      this.unannouncedWrites.delete(queryId)
    }
    this.update(queryId, (current) => {
      if (current.identity.status === 'saved') {
        return {
          ...current,
          identity: { ...current.identity, refreshing: false, refreshError: null },
        }
      }
      const creation: CreationState =
        current.creation.status === 'failed' && current.creation.outcome === 'unknown'
          ? { ...current.creation, outcome: 'not-created' }
          : current.creation
      return { ...current, identity: { status: 'absent' }, creation }
    })
  }

  /**
   * "The question is open again."
   *
   * Used when a write may or may not have landed and the follow-up check could
   * not say which. Whatever the app believed a moment ago -- typically `absent`,
   * the state that offers a Create button -- is now stale, and acting on it is
   * how a second permanent directory gets created for one document. A confirmed
   * save is not downgraded: that one cannot become false.
   *
   * `onlyWhileChecking` is for a lookup that has been overtaken by a write: it
   * has to release the `checking` state it put the record into, but it must not
   * disturb a more specific answer that arrived while it was out.
   */
  private markUnresolved(
    queryId: number,
    error: string,
    options: { onlyWhileChecking?: boolean } = {},
  ): void {
    this.update(queryId, (current) => {
      if (current.identity.status === 'saved') {
        return {
          ...current,
          identity: { ...current.identity, refreshing: false, refreshError: error },
        }
      }
      if (options.onlyWhileChecking && current.identity.status !== 'checking') {
        return current
      }
      return { ...current, identity: { status: 'unresolved', error } }
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
