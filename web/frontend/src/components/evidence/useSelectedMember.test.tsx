import { useState } from 'react'
import { act, render, renderHook, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { useSelectedMember } from './useSelectedMember'

/**
 * Every (identity, choice) pair this component has ever RENDERED, not just the
 * ones that survived to the end of a test.
 *
 * A member choice that outlives its prediction is a painting bug: the reviewer
 * reads the screen, and one commit of a previous group's witness beside a new
 * group's number is exactly the confusion issue #163 is about. An end-state
 * assertion cannot see it, because a reset in an effect repairs the value one
 * commit later and every `waitFor` then agrees.
 */
let painted: string[] = []
/** Whether each attempted choice was taken. */
let accepted: boolean[] = []

/**
 * Renders the control bound to `owner`, which is the key the control itself
 * was rendered for -- not whatever key the hook is being given now. A bar
 * animating away is exactly this: a control whose owner is no longer current.
 */
function Probe({
  identityKey,
  owner,
}: {
  identityKey: string
  owner?: string
}) {
  const selection = useSelectedMember(identityKey)
  const boundOwner = owner ?? identityKey
  painted.push(`${identityKey}:${selection.filename ?? '-'}`)
  return (
    <>
      <button
        type="button"
        onClick={() => {
          accepted.push(selection.select('query-2.txt', boundOwner))
        }}
      >
        Choose query-2
      </button>
      <button
        type="button"
        onClick={() => {
          accepted.push(selection.select('query-0.txt', boundOwner))
        }}
      >
        Choose query-0
      </button>
      <span data-testid="selected">{selection.filename ?? '-'}</span>
    </>
  )
}

function Host({
  initialKey,
  ownerOverride,
}: {
  initialKey: string
  ownerOverride?: string
}) {
  const [identityKey, setIdentityKey] = useState(initialKey)
  return (
    <>
      <button type="button" onClick={() => setIdentityKey('key-b')}>
        Switch identity
      </button>
      <button type="button" onClick={() => setIdentityKey(initialKey)}>
        Restore identity
      </button>
      <Probe identityKey={identityKey} owner={ownerOverride} />
    </>
  )
}

const selected = () => screen.getByTestId('selected').textContent

beforeEach(() => {
  painted = []
  accepted = []
})

describe('useSelectedMember', () => {
  it('starts with no manual choice', () => {
    render(<Host initialKey="key-a" />)
    expect(selected()).toBe('-')
  })

  it('keeps a choice while the identity is unchanged', async () => {
    render(<Host initialKey="key-a" />)
    await userEvent.click(screen.getByText('Choose query-2'))
    expect(selected()).toBe('query-2.txt')
    await userEvent.click(screen.getByText('Choose query-0'))
    expect(selected()).toBe('query-0.txt')
  })

  it('never paints an old choice under a new identity', async () => {
    render(<Host initialKey="key-a" />)
    await userEvent.click(screen.getByText('Choose query-2'))
    expect(painted).toContain('key-a:query-2.txt')

    await userEvent.click(screen.getByText('Switch identity'))

    expect(selected()).toBe('-')
    // The point of the test: not one render of key-b carried key-a's witness.
    expect(painted.filter((entry) => entry.startsWith('key-b:'))).not.toContain(
      'key-b:query-2.txt',
    )
    expect(new Set(painted.filter((e) => e.startsWith('key-b:')))).toEqual(
      new Set(['key-b:-']),
    )
  })

  it('refuses a choice made against a group that is no longer current', async () => {
    // The shape that made this necessary: a member strip still on screen for
    // its exit animation, whose group has already been replaced by one that
    // happens to contain a file of the same name.
    render(<Host initialKey="key-a" ownerOverride="stale-key" />)
    await userEvent.click(screen.getByText('Choose query-2'))

    expect(accepted).toEqual([false])
    expect(selected()).toBe('-')
    expect(painted).not.toContain('key-a:query-2.txt')
  })

  it('does not restore a choice when an identity comes back', async () => {
    render(<Host initialKey="key-a" />)
    await userEvent.click(screen.getByText('Choose query-2'))
    await userEvent.click(screen.getByText('Switch identity'))
    await userEvent.click(screen.getByText('Restore identity'))

    // The group behind key-a may have changed membership meanwhile; only a
    // fresh, validated default is honest here.
    expect(selected()).toBe('-')
  })
})

/**
 * The callback a departing control is actually holding.
 *
 * The cases above hand a stale owner to the CURRENT callback, which is a
 * different and easier shape. A member strip retained by `AnimatePresence`
 * holds the callback it was rendered with, and that callback's captured key is
 * its own old group -- so the event and the closure agree, and agreement
 * between two values captured together establishes nothing. `renderHook` is
 * used because the capture has to be real: the old function object itself,
 * invoked after the hook has moved on.
 */
describe('useSelectedMember, called through a callback its group left behind', () => {
  it('refuses the stale action and leaves the current choice standing', () => {
    const hook = renderHook(({ owner }) => useSelectedMember(owner), {
      initialProps: { owner: 'old-group' },
    })
    const captured = hook.result.current.select
    hook.rerender({ owner: 'current-group' })

    act(() => {
      expect(hook.result.current.select('chosen.txt', 'current-group')).toBe(true)
    })

    let accepted: boolean | undefined
    act(() => {
      accepted = captured('obsolete.txt', 'old-group')
    })

    // Refused BEFORE anything is written, so there is no state to hide
    // afterwards and the caller has nothing to announce. Returning true and
    // writing old-key state that a later render filters out is not a refusal:
    // the caller believes the choice was taken and publishes the old pair.
    expect(accepted).toBe(false)
    expect(hook.result.current.filename).toBe('chosen.txt')
  })

  it('still takes an action from the callback the current group rendered', () => {
    // The over-fix guard: a live check that refused everything would satisfy
    // the case above and break the selector entirely.
    const hook = renderHook(({ owner }) => useSelectedMember(owner), {
      initialProps: { owner: 'old-group' },
    })
    hook.rerender({ owner: 'current-group' })
    const current = hook.result.current.select

    act(() => {
      expect(current('chosen.txt', 'current-group')).toBe(true)
    })
    expect(hook.result.current.filename).toBe('chosen.txt')
  })
})
