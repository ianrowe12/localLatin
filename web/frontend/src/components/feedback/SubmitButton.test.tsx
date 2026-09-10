import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SubmitButton from './SubmitButton'
import type { SubmitActionOutcome } from './SubmitButton'

/**
 * The button's own asynchronous contract (issue #158, first slice): a click
 * starts one operation, the acknowledgement follows the awaited result, and a
 * failure is never dressed up as a save. The cross-remount, per-assessment lock
 * and the draft/navigation guarantees live in the provider and are not claimed
 * here.
 */

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function row(): HTMLElement {
  const element = screen.getByTestId('submit-button-status').parentElement
  if (element === null) throw new Error('status has no container')
  return element
}

function submitButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /submit & next/i }) as HTMLButtonElement
}

function skipButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: 'Skip' }) as HTMLButtonElement
}

function statusText(): string {
  return screen.getByTestId('submit-button-status').textContent ?? ''
}

function opState(): string | null {
  return row().getAttribute('data-op-state')
}

/** Flush the microtasks a resolved or rejected callback is waiting on. */
async function settle(): Promise<void> {
  await act(async () => {
    await Promise.resolve()
  })
}

/**
 * Two clicks React has had no chance to re-render between: this is what a
 * double click or a Save/Skip race actually looks like, and the reason the
 * button needs a synchronous guard rather than the disabled attribute.
 */
function clickTogether(...targets: HTMLElement[]): void {
  act(() => {
    for (const target of targets) {
      target.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    }
  })
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('SubmitButton', () => {
  it('stays pending until the save resolves, and only then acknowledges it', async () => {
    const user = userEvent.setup()
    const save = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn(() => save.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    await user.click(submitButton())

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(opState()).toBe('pending')
    expect(statusText()).toBe('Saving your assessment…')
    expect(submitButton().getAttribute('aria-busy')).toBe('true')
    expect(await screen.findByTestId('submit-spinner')).toBeTruthy()
    // The acknowledgement must not run ahead of the write.
    expect(screen.queryByTestId('submit-check')).toBeNull()

    await act(async () => {
      save.resolve({ ok: true })
    })

    expect(opState()).toBe('acknowledged')
    expect(statusText()).toBe('Assessment saved.')
    expect(await screen.findByTestId('submit-check')).toBeTruthy()
    expect(submitButton().getAttribute('aria-busy')).toBe('false')
    expect(submitButton().disabled).toBe(false)
  })

  it('acknowledges a resolved skip separately from a save', async () => {
    const user = userEvent.setup()
    const skip = deferred<SubmitActionOutcome>()
    const onSkip = vi.fn(() => skip.promise)

    render(<SubmitButton onSubmit={vi.fn()} onSkip={onSkip} />)
    await user.click(skipButton())

    expect(statusText()).toBe('Recording your skip…')
    expect(skipButton().getAttribute('aria-busy')).toBe('true')

    await act(async () => {
      skip.resolve({ ok: true })
    })

    expect(opState()).toBe('acknowledged')
    expect(statusText()).toBe('Skip recorded.')
    // The check belongs to Submit & Next; a skip is not a saved assessment.
    expect(screen.queryByTestId('submit-check')).toBeNull()
  })

  it('never shows a check when the save rejects, and does not swallow the error', async () => {
    const user = userEvent.setup()
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const save = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn(() => save.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    await user.click(submitButton())

    const failure = new Error('network down')
    await act(async () => {
      save.reject(failure)
    })

    expect(opState()).toBe('failed')
    expect(screen.queryByTestId('submit-check')).toBeNull()
    // Wording that does not claim the server rejected the write.
    expect(statusText()).toBe('Submit was not confirmed.')
    expect(consoleError).toHaveBeenCalledWith(
      'SubmitButton: the submit action failed',
      failure,
    )
  })

  it('reports a failure the caller handled itself without logging it again', async () => {
    const user = userEvent.setup()
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const onSubmit = vi.fn(async (): Promise<SubmitActionOutcome> => ({ ok: false }))

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    await user.click(submitButton())
    await settle()

    expect(opState()).toBe('failed')
    expect(statusText()).toBe('Submit was not confirmed.')
    expect(screen.queryByTestId('submit-check')).toBeNull()
    // The caller already owns the detailed message, so this is not an
    // unhandled failure.
    expect(consoleError).not.toHaveBeenCalled()
  })

  it('recovers after a rejection and can acknowledge the retry', async () => {
    const user = userEvent.setup()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const first = deferred<SubmitActionOutcome>()
    const second = deferred<SubmitActionOutcome>()
    const onSubmit = vi
      .fn<() => Promise<SubmitActionOutcome>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    await user.click(submitButton())
    await act(async () => {
      first.reject(new Error('network down'))
    })
    expect(opState()).toBe('failed')
    expect(submitButton().disabled).toBe(false)
    expect(skipButton().disabled).toBe(false)

    await user.click(submitButton())
    expect(onSubmit).toHaveBeenCalledTimes(2)
    // The stale failure notice clears as soon as the retry starts.
    expect(statusText()).toBe('Saving your assessment…')

    await act(async () => {
      second.resolve({ ok: true })
    })
    expect(opState()).toBe('acknowledged')
    expect(await screen.findByTestId('submit-check')).toBeTruthy()
  })

  it('leaves the button neutral when a callback reports no outcome', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn(async () => {})

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    await user.click(submitButton())
    await settle()

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(opState()).toBe('idle')
    expect(statusText()).toBe('')
    // Nothing confirmed the write, so nothing is acknowledged.
    expect(screen.queryByTestId('submit-check')).toBeNull()
  })

  it('starts one save for two clicks in the same tick', async () => {
    const save = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn(() => save.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    clickTogether(submitButton(), submitButton())

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(opState()).toBe('pending')

    await act(async () => {
      save.resolve({ ok: true })
    })
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('refuses a skip raced against a save in the same tick', async () => {
    const save = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn(() => save.promise)
    const onSkip = vi.fn()

    render(<SubmitButton onSubmit={onSubmit} onSkip={onSkip} />)
    clickTogether(submitButton(), skipButton())

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSkip).not.toHaveBeenCalled()

    await act(async () => {
      save.resolve({ ok: true })
    })
    expect(onSkip).not.toHaveBeenCalled()
  })

  it('disables both controls while either operation is in flight', async () => {
    const user = userEvent.setup()
    const skip = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn()
    const onSkip = vi.fn(() => skip.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={onSkip} />)
    await user.click(skipButton())

    expect(submitButton().disabled).toBe(true)
    expect(skipButton().disabled).toBe(true)

    await act(async () => {
      skip.resolve({ ok: true })
    })

    expect(submitButton().disabled).toBe(false)
    expect(skipButton().disabled).toBe(false)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps the Submit and Skip eligibility policies distinct', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    const onSkip = vi.fn()

    const { rerender } = render(
      <SubmitButton onSubmit={onSubmit} onSkip={onSkip} disabled />,
    )
    expect(submitButton().disabled).toBe(true)
    // A deferral stays available when an evaluation is not eligible.
    expect(skipButton().disabled).toBe(false)
    await user.click(skipButton())
    expect(onSkip).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalled()

    rerender(<SubmitButton onSubmit={onSubmit} onSkip={onSkip} skipDisabled />)
    expect(submitButton().disabled).toBe(false)
    expect(skipButton().disabled).toBe(true)
    await user.click(submitButton())
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSkip).toHaveBeenCalledTimes(1)
  })

  it('clears the acknowledgement on its own, and keeps a failure on screen', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const first = deferred<SubmitActionOutcome>()
    const second = deferred<SubmitActionOutcome>()
    const onSubmit = vi
      .fn<() => Promise<SubmitActionOutcome>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)

    render(<SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} />)
    fireEvent.click(submitButton())
    await act(async () => {
      first.resolve({ ok: true })
    })
    expect(opState()).toBe('acknowledged')

    act(() => {
      vi.advanceTimersByTime(1500)
    })
    expect(opState()).toBe('idle')
    expect(statusText()).toBe('')

    fireEvent.click(submitButton())
    await act(async () => {
      second.reject(new Error('network down'))
    })
    act(() => {
      vi.advanceTimersByTime(10_000)
    })
    // A failure has no expiry: it waits for the reviewer's next attempt.
    expect(opState()).toBe('failed')
    expect(statusText()).toBe('Submit was not confirmed.')
  })

  it('drops an acknowledgement that belongs to a previous assessment', async () => {
    const user = userEvent.setup()
    const save = deferred<SubmitActionOutcome>()
    const onSubmit = vi.fn(() => save.promise)

    const { rerender } = render(
      <SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} operationKey="q7" />,
    )
    await user.click(submitButton())
    expect(opState()).toBe('pending')

    rerender(
      <SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} operationKey="q8" />,
    )
    expect(opState()).toBe('idle')

    await act(async () => {
      save.resolve({ ok: true })
    })

    // The save may well have succeeded; it just did not save what is on screen
    // now, so the new assessment is not marked as saved.
    expect(opState()).toBe('idle')
    expect(statusText()).toBe('')
    expect(screen.queryByTestId('submit-check')).toBeNull()
  })

  it('lets the new assessment be saved while the abandoned one is still running', async () => {
    const user = userEvent.setup()
    const abandoned = deferred<SubmitActionOutcome>()
    const current = deferred<SubmitActionOutcome>()
    const onSubmit = vi
      .fn<() => Promise<SubmitActionOutcome>>()
      .mockReturnValueOnce(abandoned.promise)
      .mockReturnValueOnce(current.promise)

    const { rerender } = render(
      <SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} operationKey="q7" />,
    )
    await user.click(submitButton())
    rerender(
      <SubmitButton onSubmit={onSubmit} onSkip={vi.fn()} operationKey="q8" />,
    )

    await user.click(submitButton())
    expect(onSubmit).toHaveBeenCalledTimes(2)
    expect(opState()).toBe('pending')

    // The abandoned run completing must not unlock or acknowledge the new one.
    await act(async () => {
      abandoned.resolve({ ok: true })
    })
    expect(opState()).toBe('pending')
    clickTogether(submitButton())
    expect(onSubmit).toHaveBeenCalledTimes(2)

    await act(async () => {
      current.resolve({ ok: true })
    })
    expect(opState()).toBe('acknowledged')
  })

  it('cancels the acknowledgement timer when it unmounts', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    const save = deferred<SubmitActionOutcome>()
    const { unmount } = render(
      <SubmitButton onSubmit={() => save.promise} onSkip={vi.fn()} />,
    )

    fireEvent.click(submitButton())
    await act(async () => {
      save.resolve({ ok: true })
    })
    expect(vi.getTimerCount()).toBeGreaterThan(0)

    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('ignores a callback that completes after the button is gone', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const save = deferred<SubmitActionOutcome>()
    const { unmount } = render(
      <SubmitButton onSubmit={() => save.promise} onSkip={vi.fn()} />,
    )

    fireEvent.click(submitButton())
    unmount()

    await act(async () => {
      save.resolve({ ok: true })
    })
    await settle()

    expect(screen.queryByTestId('submit-button-status')).toBeNull()
    // No update on an unmounted tree, so React has nothing to complain about.
    expect(consoleError).not.toHaveBeenCalled()
  })

  it('ignores a rejection that arrives after the button is gone', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const save = deferred<SubmitActionOutcome>()
    const { unmount } = render(
      <SubmitButton onSubmit={() => save.promise} onSkip={vi.fn()} />,
    )

    fireEvent.click(submitButton())
    unmount()

    await act(async () => {
      save.reject(new Error('network down'))
    })
    await settle()

    expect(screen.queryByTestId('submit-button-status')).toBeNull()
    // The failure is still reported, it just has no surface left to render on.
    expect(consoleError).toHaveBeenCalledTimes(1)
  })
})
