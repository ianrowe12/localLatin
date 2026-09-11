import { useEffect, useState } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import {
  TokenProvider,
  WitnessTokenScope,
  useTokens,
  type HoverMatch,
} from './TokenContext'

/**
 * Pins and hover belong to a pair, not to the app (issue #163).
 *
 * Mounted tests can only see the state after React has flushed effects, so
 * they cannot tell "gated while rendering" from "cleared one commit later" --
 * and in a browser those differ by a painted frame showing the previous
 * witness's marks over the new witness's words. These record what each render
 * was given, which is the only place the difference is visible.
 */

const MATCH: HoverMatch = { candidateIdx: 3, score: 0.9, rank: 0 }

/** Every (witness, hovered, pins) triple the scope handed to a consumer. */
let renders: string[] = []
/** The same, read OUTSIDE the scope: the underlying store. */
let outerRenders: string[] = []

function Inside({ witness }: { witness: string }) {
  const { hoveredQueryTokenIdx, hoveredMatches, pinnedTokens, setHoveredQueryTokenIdx, setHoveredMatches, pinToken } =
    useTokens()
  renders.push(
    `${witness}:${hoveredQueryTokenIdx}:${hoveredMatches.length}:${pinnedTokens.size}`,
  )
  return (
    <>
      <button
        type="button"
        data-testid="hover"
        onClick={() => {
          setHoveredQueryTokenIdx(4)
          setHoveredMatches([MATCH])
        }}
      >
        hover
      </button>
      <button type="button" data-testid="pin" onClick={() => pinToken(4, [MATCH])}>
        pin
      </button>
    </>
  )
}

function Outside() {
  const { hoveredQueryTokenIdx, hoveredMatches, pinnedTokens } = useTokens()
  outerRenders.push(
    `${hoveredQueryTokenIdx}:${hoveredMatches.length}:${pinnedTokens.size}`,
  )
  return null
}

function Harness() {
  const [witness, setWitness] = useState('a')
  useEffect(() => {
    // Exposed so the test can change the witness without a control that would
    // itself live inside the scope.
    ;(globalThis as Record<string, unknown>).__setWitness = setWitness
  }, [])
  return (
    <TokenProvider>
      <Outside />
      <WitnessTokenScope witness={witness}>
        <Inside witness={witness} />
      </WitnessTokenScope>
    </TokenProvider>
  )
}

function switchWitness(next: string) {
  act(() => {
    ;(globalThis as { __setWitness?: (w: string) => void }).__setWitness?.(next)
  })
}

beforeEach(() => {
  renders = []
  outerRenders = []
})

describe('WitnessTokenScope (issue #163)', () => {
  it('never hands a new witness the previous one’s hover, not even for a render', () => {
    render(<Harness />)
    fireEvent.click(screen.getByTestId('hover'))
    expect(renders.some((r) => r.startsWith('a:4:1:'))).toBe(true)

    renders = []
    switchWitness('b')

    // Every render since the change, including the first one.
    expect(renders.length).toBeGreaterThan(0)
    for (const entry of renders) {
      expect(entry).toBe('b:null:0:0')
    }
  })

  it('never hands a new witness the previous one’s pins', () => {
    render(<Harness />)
    fireEvent.click(screen.getByTestId('pin'))
    expect(renders.some((r) => r.endsWith(':1'))).toBe(true)

    renders = []
    switchWitness('b')

    for (const entry of renders) {
      expect(entry).toBe('b:null:0:0')
    }
  })

  it('empties the store rather than merely hiding it', () => {
    render(<Harness />)
    fireEvent.click(screen.getByTestId('hover'))
    fireEvent.click(screen.getByTestId('pin'))

    outerRenders = []
    switchWitness('b')

    // The gate alone would leave the marks in the provider, ready to reappear
    // if the reviewer came back to the first witness.
    expect(outerRenders[outerRenders.length - 1]).toBe('null:0:0')
  })

  it('keeps marks made FOR the current witness', () => {
    render(<Harness />)
    switchWitness('b')
    fireEvent.click(screen.getByTestId('hover'))

    // The gate is about whose marks these are, not about how recently the
    // witness changed: a pin made after the change belongs to the new witness.
    expect(renders[renders.length - 1]).toBe('b:4:1:0')
  })

  it('leaves the app-wide preferences alone', () => {
    function Method() {
      const { selectedMethod, viewMode } = useTokens()
      return <span data-testid="prefs">{`${selectedMethod}/${viewMode}`}</span>
    }
    render(
      <TokenProvider>
        <WitnessTokenScope witness="a">
          <Method />
        </WitnessTokenScope>
      </TokenProvider>,
    )
    // Remounting the provider per witness would scope the marks correctly and
    // throw the reviewer's method and view choices away with them.
    expect(screen.getByTestId('prefs').textContent).toBe('ig/connections')
  })
})
