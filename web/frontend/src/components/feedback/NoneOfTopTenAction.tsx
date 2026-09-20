import { useState } from 'react'
import {
  recordNoneOfTopK,
  type NoneOfTopKReceipt,
  type RecordedKeyAnswer,
} from '../../api/cclKey'
import { FEEDBACK_UPDATED_EVENT } from '../../api/feedback'
import { notifyReviewerDirsUpdated } from '../../api/reviewerDirs'
import { toApiErrorInfo } from '../../api/client'
import type { PredictionVariant } from '../../api/variants'
import { CCL_KEY_COPY, recordedSentence } from '../../utils/reviewerDirectoryCopy'

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
  /**
   * What this reviewer already recorded for this document, from the server's
   * own prefill. Null when they have recorded nothing here.
   */
  recorded: RecordedKeyAnswer | null
  /** Called once the server has acknowledged, so the panel can close the form. */
  onRecorded: () => void
  /** Called by "Change this answer", so the panel can reopen the form. */
  onChange: () => void
}

type State =
  | { status: 'idle' }
  | { status: 'saving' }
  | { status: 'recorded'; receipt: NoneOfTopKReceipt }
  | { status: 'failed'; message: string; certain: boolean }

/**
 * The blue "None of the top N" control's form, and the answer it records.
 *
 * ONE optional field and ONE submit, which is the whole of Prof. Firey's
 * request: the evaluator says that none of the ranked candidates match and, if
 * they happen to know it, names the CCL key of the source. What that key means
 * is the server's decision, not this component's -- it may name a directory the
 * labelled corpus already holds, a group a colleague already made, or nothing
 * yet -- so the field is not validated here beyond a length bound, and the
 * receipt reports what actually happened rather than what was hoped for.
 *
 * THE RECEIPT OUTLIVES THE FORM, and that is the point of the three-way render
 * below. The first version returned null whenever `open` was false, while the
 * panel's `onRecorded` closed the form on success: the component set
 * `status: 'recorded'` and was unmounted in the same breath, so the sentence
 * naming the branch -- the only place the four outcomes are distinguishable to
 * an evaluator -- never reached the screen. The component tests passed because
 * they rendered with `open` hard-coded and a no-op `onRecorded`, which is not
 * how the panel wires it; `NoneOfTopTenPanel.test.tsx` now drives the real
 * wiring.
 *
 * So there are three states, in this order of precedence:
 *
 *   this session's receipt  -> the sentence for the branch just taken, plus
 *                              Change. Shown whether the form is open or not.
 *   a stored answer         -> the same sentence, rebuilt from the row the
 *                              server returns on revisit, plus Change. This is
 *                              what stops a returning reviewer meeting an empty
 *                              form with a live Record button over an answer
 *                              they already gave (the log is append-only).
 *   the form                -> only when the reviewer has asked for it.
 *
 * Recording again is possible and is never hidden: it appends a new answer, it
 * does not revise the old one, and the copy says so. An identical repeat is a
 * no-op server-side, which is what makes "press Record, wonder, press again"
 * harmless.
 *
 * The three outcomes of a submit are told apart, because they are not the same
 * fact: recorded (with the branch), certainly-not-recorded, and "could not
 * tell". Saying "nothing was saved" after an answer nobody heard would invite a
 * second row in an append-only log.
 */
export default function NoneOfTopTenAction({
  queryId,
  model,
  variant,
  notes,
  open,
  available,
  recorded,
  onRecorded,
  onChange,
}: NoneOfTopTenActionProps) {
  const [key, setKey] = useState('')
  const [state, setState] = useState<State>({ status: 'idle' })

  const saving = state.status === 'saving'
  const justRecorded = state.status === 'recorded' ? state.receipt : null

  // What to show back, this session's answer first. A stored answer is only
  // read when this session has not recorded one, so a fresh receipt is never
  // overwritten by a prefill fetched before it.
  const answer: RecordedKeyAnswer | null =
    justRecorded !== null
      ? {
          key: justRecorded.ccl_key ?? '',
          action: justRecorded.ccl_key_action,
          dir: justRecorded.ccl_key_dir,
          rank: justRecorded.ccl_key_rank,
        }
      : recorded

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

  if (!open && answer !== null) {
    return (
      <div
        data-testid="none-of-top-k-recorded"
        data-action={answer.action ?? 'none'}
        className="mt-3 rounded-xl border border-indigo-200 dark:border-indigo-400/30 bg-indigo-50/50 dark:bg-indigo-500/5 px-3 py-3 flex flex-col gap-1"
      >
        <p className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-300">
          {CCL_KEY_COPY.recordedHeading}
        </p>
        <p
          role="status"
          data-testid="none-of-top-k-receipt"
          className="font-ui text-xs leading-snug text-stone-700 dark:text-stone-200"
        >
          {recordedSentence(answer.action, answer.key, answer.rank)}
        </p>
        <button
          type="button"
          data-testid="none-of-top-k-change"
          onClick={() => {
            // Back to a blank field: the stored answer stays on the server and
            // is shown again if this one is abandoned.
            setState({ status: 'idle' })
            setKey('')
            onChange()
          }}
          className="mt-1 self-start rounded-lg border border-stone-300 dark:border-stone-600 px-3 py-1.5 font-ui text-xs text-stone-600 dark:text-stone-300 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
        >
          {CCL_KEY_COPY.change}
        </button>
        <p className="font-ui text-[11px] leading-snug text-stone-500 dark:text-stone-400">
          {CCL_KEY_COPY.changeNote}
        </p>
      </div>
    )
  }

  if (!open) return null

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

      {/* An answer already on the server, while the reviewer is typing another.
          It is not replaced by what they are about to record, so it stays
          visible and says so. */}
      {recorded !== null && (
        <p
          data-testid="none-of-top-k-previous"
          className="font-ui text-xs leading-snug text-stone-500 dark:text-stone-400"
        >
          {recordedSentence(recorded.action, recorded.key, recorded.rank)}{' '}
          {CCL_KEY_COPY.changeNote}
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
