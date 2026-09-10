import {
  useSavedDirectoryFor,
  useSavedDirectoryStore,
} from '../../contexts/SavedDirectoryContext'
import {
  DIRECTORY_CREATION_COPY,
  RECOVERED_INSTEAD_OF_CREATED,
  savedByNote,
} from '../../utils/reviewerDirectoryCopy'
import DirectoryCreationGuidance from './DirectoryCreationGuidance'
import DirectorySavedNotice from './DirectorySavedNotice'

interface NewDirectoryCtaProps {
  queryId: number
  model: string
  /** Rendered prominently when the top candidate is below the no-match band. */
  emphasised: boolean
  filename?: string
  /**
   * Rendered inside the no-match callout rather than standing alone: drops the
   * outer padding and adopts the callout's red, so the two read as one block.
   */
  inline?: boolean
  /**
   * Acknowledgement owned by the parent (issue #156).
   *
   * Creation is permanent and unrepeatable, and creating one triggers a
   * prediction refetch that can unmount this component. When a parent that
   * outlives the refetch passes the created label, it wins over the local copy
   * so the confirmation survives loading and failed refreshes. Left undefined,
   * the component keeps its own state exactly as before.
   */
  createdLabel?: string | null
  /** Called once, with the created label, on a success still on screen. */
  onCreated?: (label: string) => void
}

/**
 * "This is a new source" -- the creation flow for a reviewer directory.
 *
 * Renders as the default top option inside `NoMatchCallout` below the no-match
 * band (issue #94), and un-emphasised at the foot of the list above it, since
 * "this is a new source" is a judgement a reviewer can reach at any score.
 * `emphasised` and `inline` are styling knobs only.
 *
 * NAMING IS PART OF CREATION, so the button opens a one-field form rather than
 * posting immediately. Both tables are append-only with no rename, so a
 * directory's label is permanent from the moment it exists and a reviewer gets
 * one chance at it. The field is pre-filled with the seed's filename, so
 * accepting the default is still just a second click. Every sentence here lives
 * in `utils/reviewerDirectoryCopy` and the blocks around the field are their own
 * components (issue #162).
 *
 * WHAT THIS COMPONENT NO LONGER OWNS (issue #161). It used to keep the created
 * label in `useState` and drop any response that arrived after the reviewer
 * moved on. Both were wrong in the same way: creation writes a permanent record,
 * and the acknowledgement of it was living in a component that its own success
 * unmounts -- `createReviewerDir` broadcasts a prediction refresh, and
 * `PredictionList` returns a different subtree while that refresh is in flight.
 * A refresh that then failed took the only evidence of the write with it and
 * offered a fresh Create button for a directory that already existed.
 *
 * So the durable half lives in `contexts/savedDirectoryStore`, keyed by the SEED
 * QUERY rather than by this component's lifetime, and this file is the view of
 * one query's record:
 *
 *   saved       -> the acknowledgement, whatever the prediction request is doing
 *   absent      -> the create button, the only state in which it is offered
 *   checking    -> a quiet line; the question has not been answered yet
 *   unresolved  -> an honest "could not check", with a retry, never a Create
 *
 * A late completion is recorded against the query it was started on even if
 * this instance is showing another one, so navigating away no longer throws the
 * acknowledgement away, and cannot paint it onto the next fragment either.
 *
 * This component therefore REQUIRES a `SavedDirectoryProvider` above it and
 * throws without one. Mounting it in App composition is deferred to issue
 * #156's handoff; every surface that renders this CTA must wrap it until then.
 */
export default function NewDirectoryCta({
  queryId,
  model,
  emphasised,
  filename,
  inline = false,
  createdLabel,
  onCreated,
}: NewDirectoryCtaProps) {
  const store = useSavedDirectoryStore()
  const { identity, creation } = useSavedDirectoryFor(queryId, { model })

  const suggestion = filename
    ? `New directory from ${filename.replace(/\.txt$/, '')}`
    : 'New directory'

  const block = inline ? 'mt-2' : 'px-2 mt-2'

  // --- saved: the acknowledgement ------------------------------------------
  if (identity.status === 'saved') {
    const { primary, confirmedBy } = identity
    return (
      <div className={inline ? 'mt-2' : 'mt-2 mx-2'}>
        <DirectorySavedNotice label={primary.label} />
        {confirmedBy === 'recovered' && (
          <p
            data-testid="new-directory-recovered"
            role="status"
            className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1"
          >
            {RECOVERED_INSTEAD_OF_CREATED}
          </p>
        )}
        {confirmedBy !== 'created' && (
          <p
            data-testid="new-directory-attribution"
            className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1"
          >
            {savedByNote(primary.created_by)}
          </p>
        )}
      </div>
    )
  }

  // --- the naming form ------------------------------------------------------
  // Ahead of the identity states below: an issued write is the one thing a
  // reviewer must not lose sight of while the lookup catches up.
  if (creation.status !== 'idle') {
    const pending = creation.status === 'pending'
    return (
      <form
        onSubmit={(event) => {
          event.preventDefault()
          // The store refuses a second write for the same attempt, so a
          // double-click or a stray Enter cannot post twice.
          void store.createDirectory(queryId, {
            label: creation.proposedLabel,
            model: model || undefined,
          })
        }}
        className={block}
        data-testid="new-directory-form"
        aria-busy={pending}
      >
        <label
          className="block text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-1"
          htmlFor="reviewer-dir-label"
        >
          {DIRECTORY_CREATION_COPY.fieldLabel}
        </label>
        <input
          id="reviewer-dir-label"
          type="text"
          // The proposed name survives an error, a re-open and an outcome
          // nobody could establish: retyping it is how a second directory ends
          // up under a slightly different name.
          value={creation.proposedLabel}
          onChange={(event) => store.setProposedLabel(queryId, event.target.value)}
          placeholder={suggestion}
          maxLength={200}
          disabled={pending}
          autoFocus
          className="w-full rounded-md border border-stone-300 dark:border-stone-600 bg-white dark:bg-surface-800 px-2 py-1.5 text-sm disabled:opacity-60"
        />
        {creation.status === 'failed' && (
          <>
            <p
              role="alert"
              data-testid="new-directory-error"
              className="text-xs text-red-600 dark:text-red-400 mt-1"
            >
              {creation.error}
            </p>
            <p
              data-testid="new-directory-outcome"
              className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300 mt-1"
            >
              {creation.outcome === 'not-created'
                ? DIRECTORY_CREATION_COPY.failedNotCreated
                : DIRECTORY_CREATION_COPY.failedUnknown}
            </p>
          </>
        )}
        {pending && (
          <p
            data-testid="new-directory-pending-note"
            role="status"
            className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300 mt-1"
          >
            {DIRECTORY_CREATION_COPY.pendingNote}
          </p>
        )}
        <DirectoryCreationGuidance />
        <div className="flex gap-2 mt-2">
          <button
            type="submit"
            disabled={pending}
            data-testid="new-directory-submit"
            className="flex-1 rounded-md bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-xs font-medium px-3 py-1.5"
          >
            {pending
              ? DIRECTORY_CREATION_COPY.submitting
              : DIRECTORY_CREATION_COPY.submit}
          </button>
          {creation.status === 'failed' && creation.outcome === 'unknown' && (
            <button
              type="button"
              data-testid="new-directory-check-again"
              onClick={() => {
                void store.ensureLookup(queryId, { force: true, model })
              }}
              className="rounded-md border border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 text-xs px-3 py-1.5"
            >
              {DIRECTORY_CREATION_COPY.checkAgain}
            </button>
          )}
          <button
            type="button"
            data-testid="new-directory-cancel"
            onClick={() => {
              // Two different actions behind one position. Before the request
              // Cancel abandons an unsent form; afterwards there is nothing to
              // cancel, so the control says what it really does.
              if (pending) store.dismissPendingForm(queryId)
              else store.cancelNaming(queryId)
            }}
            className="rounded-md border border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 text-xs px-3 py-1.5"
          >
            {pending
              ? DIRECTORY_CREATION_COPY.closePending
              : DIRECTORY_CREATION_COPY.cancel}
          </button>
        </div>
      </form>
    )
  }

  // --- the question has not been answered yet -------------------------------
  if (identity.status === 'checking' || identity.status === 'unknown') {
    return (
      <p
        data-testid="new-directory-checking"
        className={`${block} font-ui text-xs leading-snug text-stone-500 dark:text-stone-400`}
      >
        {DIRECTORY_CREATION_COPY.checking}
      </p>
    )
  }

  // --- the question could not be answered -----------------------------------
  if (identity.status === 'unresolved') {
    return (
      <div className={block} data-testid="new-directory-unresolved">
        <p
          role="status"
          className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-300"
        >
          {DIRECTORY_CREATION_COPY.unresolvedNote}
        </p>
        <button
          type="button"
          data-testid="new-directory-check-again"
          onClick={() => {
            void store.ensureLookup(queryId, { force: true, model })
          }}
          className="mt-1 rounded-md border border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 text-xs px-3 py-1.5"
        >
          {DIRECTORY_CREATION_COPY.checkAgain}
        </button>
      </div>
    )
  }

  // --- absent: the only state in which creating is on offer -----------------
  return (
    <div className={block}>
      <button
        type="button"
        data-testid="new-directory-cta"
        onClick={() => store.beginNaming(queryId, suggestion)}
        className={
          emphasised
            ? `w-full rounded-lg bg-incorrect px-2 py-1.5 text-xs font-semibold
               text-white transition-colors hover:bg-incorrect-light
               focus:outline-none focus:ring-2 focus:ring-incorrect/40`
            : 'w-full rounded-lg border border-dashed border-stone-300 dark:border-stone-600 text-stone-600 dark:text-stone-400 hover:border-indigo-400 hover:text-indigo-600 text-xs font-ui px-3 py-2 transition-colors'
        }
      >
        {emphasised
          ? DIRECTORY_CREATION_COPY.openEmphasised
          : DIRECTORY_CREATION_COPY.openQuiet}
      </button>
      <p
        data-testid="new-directory-caption"
        className="font-ui text-[11px] leading-snug text-stone-600 dark:text-stone-300 mt-1"
      >
        {DIRECTORY_CREATION_COPY.buttonCaption}
      </p>
    </div>
  )
}
