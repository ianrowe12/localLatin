import { useState } from 'react'
import { render, screen } from '@testing-library/react'
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

function Probe({ identityKey }: { identityKey: string }) {
  const selection = useSelectedMember(identityKey)
  painted.push(`${identityKey}:${selection.filename ?? '-'}`)
  return (
    <>
      <button type="button" onClick={() => selection.select('query-2.txt')}>
        Choose query-2
      </button>
      <button type="button" onClick={() => selection.select('query-0.txt')}>
        Choose query-0
      </button>
      <span data-testid="selected">{selection.filename ?? '-'}</span>
    </>
  )
}

function Host({ initialKey }: { initialKey: string }) {
  const [identityKey, setIdentityKey] = useState(initialKey)
  return (
    <>
      <button type="button" onClick={() => setIdentityKey('key-b')}>
        Switch identity
      </button>
      <button type="button" onClick={() => setIdentityKey(initialKey)}>
        Restore identity
      </button>
      <Probe identityKey={identityKey} />
    </>
  )
}

const selected = () => screen.getByTestId('selected').textContent

beforeEach(() => {
  painted = []
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
