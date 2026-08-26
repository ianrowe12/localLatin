import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useTokenMap } from './tokenMap'
import { toAttributionVariant, type PredictionVariant } from './variants'

// Moved here from VariantSelector.test.tsx, which went away with the variant
// picker in issue #94. The raw <-> baseline bridge did not: the prediction
// CSVs and the IG artifacts still name the uncorrected variant differently,
// and the app still asks the token-map endpoint for a named variant.

const MODEL = 'bowphs_LaTa'
const QUERY_ID = 7

let requested: string[] = []

function installFetch(): void {
  requested = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      requested.push(String(input))
      return new Response(
        JSON.stringify({ example_id: 1, model: MODEL, available_variants: [] }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    }),
  )
}

function TokenMapHarness({ variant }: { variant: PredictionVariant }) {
  useTokenMap(QUERY_ID, 'cand-dir', MODEL, 'ig', toAttributionVariant(variant))
  return null
}

async function tokenMapParams(variant: PredictionVariant): Promise<URLSearchParams> {
  render(<TokenMapHarness variant={variant} />)
  await waitFor(() => {
    expect(requested.some((url) => url.includes('/token_map'))).toBe(true)
  })
  return new URL(requested.find((url) => url.includes('/token_map'))!, 'http://localhost')
    .searchParams
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('token map variant mapping', () => {
  it("sends the artifact's 'baseline' name for the uncorrected variant", async () => {
    const params = await tokenMapParams('raw')
    expect(params.get('variant')).toBe('baseline')
    // The method filter keeps the response from shipping all 7x4 matrices.
    expect(params.get('method')).toBe('ig')
  })

  it('passes the other variant names through unchanged', async () => {
    const params = await tokenMapParams('sif_abtt')
    expect(params.get('variant')).toBe('sif_abtt')
  })
})
