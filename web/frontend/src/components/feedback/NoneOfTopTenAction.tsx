import { useState } from 'react'
import { recordNoneOfTopK, type NoneOfTopKReceipt } from '../../api/cclKey'
import { FEEDBACK_UPDATED_EVENT } from '../../api/feedback'
import { notifyReviewerDirsUpdated } from '../../api/reviewerDirs'
import { toApiErrorInfo } from '../../api/client'
import type { PredictionVariant } from '../../api/variants'
import { CCL_KEY_COPY } from '../../utils/reviewerDirectoryCopy'

interface NoneOfTopTenActionProps {
  queryId: number
  model: string
  variant: PredictionVariant
  /** The prose in the shared notes box, saved with this answer. */
  notes: string
  /** Whether the reviewer has pressed the blue control. */
  open: boolean
  /** False when the ranking cannot be rejected (an unreadable candidate). */
  available: boolean
  /** Called once the server has acknowledged, so the panel can clear itself. */
  onRecorded: () => void
}

type State =
  | { status: 'idle' }
  | { status: 'saving' }
  | { status: 'recorded'; receipt: NoneOfTopKReceipt }
  | { status: 'failed'; message: string; certain: boolean }

/**
 * The form behind the blue "None of the top N" control (issue #196).
 *
 * ONE optional field and ONE submit, which is the whole of Prof. Firey's
 * request: the evaluator says that none of the ranked candidates match and, if
 * they happen to know it, names the CCL key of the source. What that key means
 * is the server's decision, not this component's -- it may name a directory the
 * labelled corpus already holds, a group a colleague already made, or nothing
 * yet -- so the field is not validated here beyond a length bound, and the
 * receipt reports what actually happened rather than what was hoped for.
 *
 * It submits on its own rather than through the panel's Save button, and that
 * is deliberate twice over. For the reviewer it is one action with one result:
 * press, type or not, record. For the code it keeps this write out of the
 * assessment save path, which issue #158/#172 is repairing in parallel; the two
 * changes touch different files and can land in either order.
 *
 * The three outcomes are told apart, because they are not the same fact:
 * recorded (with the branch the server took), certainly-not-recorded, and
 * "could not tell". Only the first clears the panel. Saying "nothing was saved"
 * after an answer nobody heard would invite a second row in an append-only log.
 */
export default function NoneOfTopTenAction({
  queryId,
  model,
  variant,
  notes,
  open,
  available,
  onRecorded,
}: NoneOfTopTenActionProps) {
  const [key, setKey] = useState('')
  const [state, setState] = useState<State>({ status: 'idle' })

  if (!open) return null

  const saving = state.status === 'saving'

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault()
    if (saving) return
    setState({ status: 'saving' })
    try {
      const receipt = await recordNoneOfTopK({
        queryId,
        model,
        variant,
        cclKey: key,
        notes,
      })
      setState({ status: 'recorded', receipt })
      setKey('')
      // A created or joined group changes what every other document is offered,
      // and the answer itself changes this document's review status.
      window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
      if (receipt.ccl_key_action !== null) notifyReviewerDirsUpdated()
      onRecorded()
    } catch (error) {
      const info = toApiErrorInfo(
        error,
        'The app could not reach the server, so this answer was not recorded.',
      )
      // A refusal the server made before writing is certain; a network failure
      // or an unreadable answer is not, and the difference decides whether the
      // reviewer is told to press again.
      const certain =
        info.kind === 'http' && info.status !== null && info.status < 500
      setState({
        status: 'failed',
        message: info.message,
        certain,
      })
    }
  }

  return (
    <form
      onSubmit={submit}
      data-testid="none-of-top-k-form"
      className="mt-3 rounded-xl border border-indigo-200 dark:border-indigo-400/30 bg-indigo-50/50 dark:bg-indigo-500/5 px-3 py-3 flex flex-col gap-2"
      aria-busy={saving}
    >
      <label
        htmlFor="ccl-key"
        className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-300"
      >
        {CCL_KEY_COPY.fieldLabel}
      </label>
      <input
        id="ccl-key"
        type="text"
        value={key}
        onChange={(event) => setKey(event.target.value)}
        placeholder={CCL_KEY_COPY.placeholder}
        maxLength={200}
        disabled={saving}
        data-testid="ccl-key-input"
        className="w-full rounded-lg border border-stone-300 dark:border-stone-600 bg-white dark:bg-surface-800 px-3 py-2 text-sm disabled:opacity-60"
      />
      <p className="font-ui text-xs leading-snug text-stone-500 dark:text-stone-400">
        {CCL_KEY_COPY.optionalNote}
      </p>
      <p className="font-ui text-xs leading-snug text-stone-500 dark:text-stone-400">
        {CCL_KEY_COPY.outcomeNote}
      </p>

      {state.status === 'recorded' && (
        <p
          role="status"
          data-testid="none-of-top-k-receipt"
          data-action={state.receipt.ccl_key_action ?? 'none'}
          className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200"
        >
          {receiptCopy(state.receipt)}
        </p>
      )}
      {state.status === 'failed' && (
        <div role="alert" data-testid="none-of-top-k-error">
          <p className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200">
            {state.message}
          </p>
          <p className="mt-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300">
            {state.certain ? CCL_KEY_COPY.failed : CCL_KEY_COPY.uncertain}
          </p>
        </div>
      )}

      <button
        type="submit"
        disabled={saving || !available}
        data-testid="none-of-top-k-submit"
        title={
          available
            ? undefined
            : 'Some candidates cannot be read on this screen, so they cannot be rejected.'
        }
        className="mt-1 w-full rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-sm font-semibold px-3 py-2.5"
      >
        {saving ? CCL_KEY_COPY.submitting : CCL_KEY_COPY.submit}
      </button>
    </form>
  )
}

/** The receipt sentence for the branch the server actually took. */
function receiptCopy(receipt: NoneOfTopKReceipt): string {
  const key = receipt.ccl_key ?? ''
  if (receipt.ccl_key_action === null || key === '') {
    return CCL_KEY_COPY.recorded.none
  }
  return CCL_KEY_COPY.recorded[receipt.ccl_key_action](key)
}
