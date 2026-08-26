import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useTokenMap, type AttributionVariant } from './tokenMap'

const MODEL = 'bowphs_LaTa'
const DIR = 'CSAR.347.17'

/** Variants the artifact index has no row for, so the endpoint 404s. */
let missing = new Set<string>()
let delayMs = 0

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  missing = new Set()
  delayMs = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const params = new URL(String(input), 'http://localhost').searchParams
      const variant = params.get('variant') ?? ''
      if (delayMs > 0) await new Promise((r) => setTimeout(r, delayMs))
      if (missing.has(variant)) {
        return jsonResponse({ error: { message: 'Example not found' } }, 404)
      }
      return jsonResponse({
        example_id: 122,
        model: MODEL,
        similarity_matrix: [[1]],
        // A per-variant marker so a stale payload is recognisable.
        pair_matrices: { ig: { [variant]: [[1]] } },
      })
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderForVariant(variant: AttributionVariant) {
  return renderHook(
    ({ v }: { v: AttributionVariant }) =>
      useTokenMap(7, DIR, MODEL, 'ig', v),
    { initialProps: { v: variant } },
  )
}

describe('useTokenMap', () => {
  it('clears the previous key data instead of holding it under loading', async () => {
    delayMs = 30
    const { result, rerender } = renderForVariant('sif_abtt')
    await waitFor(() =>
      expect(result.current.data?.pair_matrices?.ig).toHaveProperty('sif_abtt'),
    )

    rerender({ v: 'baseline' })
    // While the new matrix is in flight there is no matrix, rather than the
    // previous variant's under the new variant's label (issue #73).
    expect(result.current.loading).toBe(true)
    expect(result.current.data).toBeNull()

    await waitFor(() =>
      expect(result.current.data?.pair_matrices?.ig).toHaveProperty('baseline'),
    )
  })

  it('renders nothing, not the last successful pair, when the fetch fails', async () => {
    missing.add('abtt')
    const { result, rerender } = renderForVariant('sif_abtt')
    await waitFor(() => expect(result.current.data).not.toBeNull())

    rerender({ v: 'abtt' })
    await waitFor(() => expect(result.current.error).not.toBeNull())
    expect(result.current.data).toBeNull()
  })
})
