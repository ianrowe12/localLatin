import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { useTokenMap } from '../../api/tokenMap'
import { toAttributionVariant } from '../../api/variants'
import PredictionList from './PredictionList'
import VariantSelector from './VariantSelector'

const MODEL = 'bowphs_LaTa'
const QUERY_ID = 7

const MODELS_PAYLOAD = [
  {
    slug: MODEL,
    display_name: 'LaTa (T5)',
    layer: 4,
    pooling: 'mean',
    prediction_count: 2238,
    available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
    default_variant: 'sif_abtt',
  },
]

function predictionsPayload(variant: string) {
  return {
    file_id: QUERY_ID,
    filename: 'query-7.txt',
    model: MODEL,
    variant,
    predictions: [
      {
        rank: 1,
        dir_name: `dir-for-${variant}`,
        score: 0.9,
        dir_files: ['a.txt'],
        preview_text: 'preview',
        candidate_files: [{ filename: 'a.txt', text: 'candidate text' }],
      },
    ],
  }
}

let requested: string[] = []

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  requested = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    requested.push(url)
    if (url.includes('/api/models')) return jsonResponse(MODELS_PAYLOAD)
    if (url.includes('/predictions')) {
      const variant = new URL(url, 'http://localhost').searchParams.get('variant') ?? ''
      return jsonResponse(predictionsPayload(variant))
    }
    if (url.includes('/token_map')) {
      return jsonResponse({ example_id: 1, model: MODEL, available_variants: [] })
    }
    return jsonResponse({})
  })
  vi.stubGlobal('fetch', fetchMock)
}

function predictionRequests(): URLSearchParams[] {
  return requested
    .filter((url) => url.includes('/predictions'))
    .map((url) => new URL(url, 'http://localhost').searchParams)
}

function lastPredictionRequest(): URLSearchParams | undefined {
  const all = predictionRequests()
  return all[all.length - 1]
}

function Harness() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return (
    <>
      <VariantSelector />
      <PredictionList />
    </>
  )
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('VariantSelector', () => {
  it('renders all four variants with the deployment default selected', async () => {
    render(
      <AppProvider>
        <VariantSelector />
      </AppProvider>,
    )

    const options = await screen.findAllByRole('radio')
    expect(options.map((o) => o.textContent)).toEqual([
      'Raw',
      'ABTT',
      'SIF',
      'SIF+ABTT',
    ])
    // Tooltips carry the one-line explanation for each variant.
    expect(screen.getByRole('radio', { name: /Raw/ })).toHaveProperty(
      'title',
      'Raw — no post-processing',
    )
    expect(screen.getByRole('radio', { name: /SIF\+ABTT/ }).getAttribute('aria-checked')).toBe(
      'true',
    )
  })

  it('is not gated on the reviewer role: no auth call is needed to render it', () => {
    render(
      <AppProvider>
        <VariantSelector />
      </AppProvider>,
    )
    expect(screen.getAllByRole('radio')).toHaveLength(4)
  })

  it('refetches predictions with the selected variant and persists the choice', async () => {
    const user = userEvent.setup()
    render(
      <AppProvider>
        <Harness />
      </AppProvider>,
    )

    await waitFor(() => {
      expect(predictionRequests().length).toBeGreaterThan(0)
    })
    expect(lastPredictionRequest()?.get('variant')).toBe('sif_abtt')

    await user.click(screen.getByRole('radio', { name: /^ABTT/ }))

    await waitFor(() => {
      expect(lastPredictionRequest()?.get('variant')).toBe('abtt')
    })
    expect(lastPredictionRequest()?.get('model')).toBe(MODEL)
    expect(localStorage.getItem('locallatin-variant')).toBe('abtt')

    // ...and the list actually re-renders with the new variant's ranking.
    expect(await screen.findByText('dir-for-abtt')).toBeTruthy()
  })

  it('restores a persisted choice over the deployment default', async () => {
    localStorage.setItem('locallatin-variant', 'sif')
    render(
      <AppProvider>
        <Harness />
      </AppProvider>,
    )

    await waitFor(() => {
      expect(predictionRequests().length).toBeGreaterThan(0)
    })
    // The server default (sif_abtt) must not clobber the reviewer's choice.
    expect(predictionRequests().every((p) => p.get('variant') === 'sif')).toBe(true)
  })
})

describe('token map variant mapping', () => {
  function TokenMapHarness({ variant }: { variant: 'raw' | 'sif_abtt' }) {
    useTokenMap(QUERY_ID, 'cand-dir', MODEL, 'ig', toAttributionVariant(variant))
    return null
  }

  it("sends the artifact's 'baseline' name when the reviewer picks Raw", async () => {
    render(<TokenMapHarness variant="raw" />)
    await waitFor(() => {
      expect(requested.some((url) => url.includes('/token_map'))).toBe(true)
    })
    const params = new URL(
      requested.find((url) => url.includes('/token_map'))!,
      'http://localhost',
    ).searchParams
    expect(params.get('variant')).toBe('baseline')
    // The method filter keeps the response from shipping all 7x4 matrices.
    expect(params.get('method')).toBe('ig')
  })

  it('passes the other variant names through unchanged', async () => {
    render(<TokenMapHarness variant="sif_abtt" />)
    await waitFor(() => {
      expect(requested.some((url) => url.includes('/token_map'))).toBe(true)
    })
    const params = new URL(
      requested.find((url) => url.includes('/token_map'))!,
      'http://localhost',
    ).searchParams
    expect(params.get('variant')).toBe('sif_abtt')
  })
})
