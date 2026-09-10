import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export type SubmitActionKind = 'submit' | 'skip'

/**
 * What a Save or Skip callback tells the button about the write it performed.
 *
 * Returning nothing keeps the older void contract: the callback finished, but
 * nothing confirmed that anything was written, so the button acknowledges
 * neither success nor failure. A caller that knows the answer returns it, which
 * is what lets the acknowledged check mean "this save landed" rather than "the
 * click was received". The object is deliberately open for the rest of issue
 * #158 to carry the saved revision through without touching call sites again.
 */
export interface SubmitActionOutcome {
  ok: boolean
}

export type SubmitActionResult = void | SubmitActionOutcome

export type SubmitAction = () =>
  | SubmitActionResult
  | Promise<SubmitActionResult>

interface SubmitButtonProps {
  onSubmit: SubmitAction
  onSkip: SubmitAction
  disabled?: boolean
  skipDisabled?: boolean
  /**
   * Identifies the assessment these controls act on. When it changes, the
   * button forgets the previous assessment's pending run and its
   * acknowledgement instead of showing either against the new one. The rest of
   * issue #158 supplies it; the per-assessment lock that survives a remount
   * belongs to the provider, not here.
   */
  operationKey?: string | number
}

interface Acknowledgement {
  kind: SubmitActionKind
  ok: boolean
}

/** How long an acknowledged success stays on screen before the button resets. */
const ACK_VISIBLE_MS = 1500

const PENDING_TEXT: Record<SubmitActionKind, string> = {
  submit: 'Saving your assessment…',
  skip: 'Recording your skip…',
}

const SUCCESS_TEXT: Record<SubmitActionKind, string> = {
  submit: 'Assessment saved.',
  skip: 'Skip recorded.',
}

// Neither wording claims what reached the server: the button only knows that
// its own operation was not confirmed. Whatever detail exists -- which request
// failed, whether the draft survived, whether a retry is safe -- belongs to the
// caller's own error region, not to this notice.
const FAILURE_TEXT: Record<SubmitActionKind, string> = {
  submit: 'Submit was not confirmed.',
  skip: 'Skip was not confirmed.',
}

export default function SubmitButton({
  onSubmit,
  onSkip,
  disabled,
  skipDisabled,
  operationKey,
}: SubmitButtonProps) {
  const [pending, setPending] = useState<SubmitActionKind | null>(null)
  const [acknowledgement, setAcknowledgement] =
    useState<Acknowledgement | null>(null)

  // The second of two clicks in one tick runs before React has re-rendered with
  // the disabled attribute, so the guard has to be synchronous. It is
  // button-local: it stops this pair of controls from starting two operations at
  // once, and is not exactly-once behaviour for the write itself.
  const activeRunRef = useRef<number | null>(null)
  const runIdRef = useRef(0)
  const mountedRef = useRef(true)
  const ackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearAckTimer = useCallback(() => {
    if (ackTimerRef.current !== null) {
      clearTimeout(ackTimerRef.current)
      ackTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      clearAckTimer()
    }
  }, [clearAckTimer])

  // A callback that resolves after the button has gone, after a newer run
  // started, or after the reviewer moved to another assessment must not
  // resurrect a stale state.
  const finish = useCallback(
    (runId: number, result: Acknowledgement | null) => {
      if (!mountedRef.current || runId !== runIdRef.current) return
      setPending(null)
      setAcknowledgement(result)
      if (result !== null && result.ok) {
        ackTimerRef.current = setTimeout(() => {
          ackTimerRef.current = null
          if (mountedRef.current) setAcknowledgement(null)
        }, ACK_VISIBLE_MS)
      }
    },
    [],
  )

  const run = useCallback(
    async (kind: SubmitActionKind, action: SubmitAction) => {
      if (activeRunRef.current !== null) return
      const runId = ++runIdRef.current
      activeRunRef.current = runId
      clearAckTimer()
      setAcknowledgement(null)
      setPending(kind)

      let outcome: SubmitActionResult
      try {
        outcome = await action()
      } catch (error) {
        // A rejection the caller did not handle would otherwise vanish with the
        // promise, leaving only this short notice; log it so the real failure is
        // still recoverable from the console.
        console.error(`SubmitButton: the ${kind} action failed`, error)
        finish(runId, { kind, ok: false })
        return
      } finally {
        // Only the run that still holds the lock may release it: an abandoned
        // run finishing late must not unlock a newer one.
        if (activeRunRef.current === runId) activeRunRef.current = null
      }

      finish(runId, outcome === undefined ? null : { kind, ok: outcome.ok })
    },
    [clearAckTimer, finish],
  )

  const seenKeyRef = useRef(operationKey)
  useEffect(() => {
    if (seenKeyRef.current === operationKey) return
    seenKeyRef.current = operationKey
    activeRunRef.current = null
    runIdRef.current += 1
    clearAckTimer()
    setPending(null)
    setAcknowledgement(null)
  }, [clearAckTimer, operationKey])

  const busy = pending !== null
  const showCheck =
    acknowledgement !== null &&
    acknowledgement.ok &&
    acknowledgement.kind === 'submit'
  const failed = acknowledgement !== null && !acknowledgement.ok

  const statusText =
    pending !== null
      ? PENDING_TEXT[pending]
      : acknowledgement === null
        ? ''
        : acknowledgement.ok
          ? SUCCESS_TEXT[acknowledgement.kind]
          : FAILURE_TEXT[acknowledgement.kind]

  const opState = busy
    ? 'pending'
    : acknowledgement === null
      ? 'idle'
      : acknowledgement.ok
        ? 'acknowledged'
        : 'failed'

  return (
    <div
      data-tour="submit-skip"
      data-op-state={opState}
      data-op-kind={pending ?? acknowledgement?.kind}
      className="flex flex-wrap items-center gap-2"
    >
      <button
        type="button"
        onClick={() => {
          void run('submit', onSubmit)
        }}
        // Either operation in flight blocks both controls; `disabled` on its own
        // stays the caller's evaluation-eligibility policy.
        disabled={disabled === true || busy}
        aria-busy={pending === 'submit'}
        className="bg-accent hover:bg-accent-dark text-white px-4 py-2 rounded-lg
                   text-sm font-medium flex items-center gap-2 transition-all
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Submit &amp; Next
        <AnimatePresence mode="wait" initial={false}>
          {pending === 'submit' ? (
            <motion.span
              key="spinner"
              data-testid="submit-spinner"
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.6, opacity: 0 }}
              transition={{ duration: 0.15 }}
              aria-hidden="true"
            >
              <svg
                className="animate-spin"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
              >
                <path d="M12 3a9 9 0 1 0 9 9" />
              </svg>
            </motion.span>
          ) : showCheck ? (
            <motion.span
              key="check"
              data-testid="submit-check"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ duration: 0.15 }}
              aria-hidden="true"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </motion.span>
          ) : (
            <motion.span
              key="arrow"
              initial={{ x: -4, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 4, opacity: 0 }}
              transition={{ duration: 0.15 }}
              aria-hidden="true"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </motion.span>
          )}
        </AnimatePresence>
      </button>

      <button
        type="button"
        onClick={() => {
          void run('skip', onSkip)
        }}
        // Skip keeps its own eligibility policy: it is a deliberate deferral,
        // not a weaker Submit, so `disabled` never reaches it.
        disabled={skipDisabled === true || busy}
        aria-busy={pending === 'skip'}
        className="text-stone-500 hover:text-stone-700 dark:hover:text-stone-300
                   hover:bg-stone-100 dark:hover:bg-stone-800
                   px-3 py-2 rounded-lg text-sm transition-colors
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Skip
      </button>

      {/* Announced politely, and kept separate from the caller's own error
          region so a detailed explanation is never duplicated or replaced by
          this short notice. */}
      <p
        role="status"
        aria-live="polite"
        data-testid="submit-button-status"
        className={`font-ui text-xs leading-snug ${
          failed
            ? 'text-incorrect'
            : 'text-stone-500 dark:text-stone-400'
        }`}
      >
        {statusText}
      </p>
    </div>
  )
}
