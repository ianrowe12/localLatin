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

/** What `GET /api/models` reports for the test currently running. */
function modelsPayload(
  availableVariants: string[] = ['raw', 'abtt', 'sif', 'sif_abtt'],
  defaultVariant = 'sif_abtt',
) {
  return [
    {
      slug: MODEL,
      display_name: 'LaTa (T5)',
      layer: 4,
      pooling: 'mean',
      prediction_count: 2238,
      available_variants: availableVariants,
      default_variant: defaultVariant,
    },
  ]
}

let models = modelsPayload()

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
    if (url.includes('/api/models')) return jsonResponse(models)
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
  models = modelsPayload()
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

  it('is not gated on the reviewer role: renders without consulting auth', async () => {
    render(
      <AppProvider>
        <VariantSelector />
      </AppProvider>,
    )
    expect(screen.getAllByRole('radio')).toHaveLength(4)
    await waitFor(() => {
      expect(requested.some((url) => url.includes('/api/models'))).toBe(true)
    })
    // No role lookup anywhere in this component's tree — the PI gate applies
    // to the attribution *method* picker, not to this control.
    expect(requested.some((url) => url.includes('/api/auth'))).toBe(false)
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

  // --- regression: default-resolution vs availability fallback -------------

  it("honours the deployment default when the compile-time default is unserved", async () => {
    // The reviewer's repro. Before the fix, two effects committed together:
    // applyDefaultVariant queued 'sif' while the fallback effect, reading the
    // pre-default activeVariant ('sif_abtt', unserved here), queued
    // available[0] = 'raw' and won the batch. The reviewer landed on the
    // uncorrected variant in a deployment that declared 'sif'.
    models = modelsPayload(['raw', 'abtt', 'sif'], 'sif')
    render(
      <AppProvider>
        <Harness />
      </AppProvider>,
    )

    await waitFor(() => {
      expect(predictionRequests().length).toBeGreaterThan(0)
    })
    await waitFor(() => {
      expect(
        screen.getByRole('radio', { name: /^SIF —/ }).getAttribute('aria-checked'),
      ).toBe('true')
    })
    // Settles on the deployment default. The first request necessarily uses
    // the compile-time default, since /api/models has not answered yet — but
    // 'raw' (available[0], what the buggy fallback picked) is never requested.
    expect(lastPredictionRequest()?.get('variant')).toBe('sif')
    expect(predictionRequests().some((p) => p.get('variant') === 'raw')).toBe(false)
    // Nothing here was a reviewer choice, so nothing may be recorded as one.
    expect(localStorage.getItem('locallatin-variant')).toBeNull()
  })

  it('keeps a stored choice on disk while its variant is unserved', async () => {
    // A machine fallback is display-level only: when the missing CSV lands,
    // the reviewer's real preference must come back.
    localStorage.setItem('locallatin-variant', 'sif_abtt')
    models = modelsPayload(['raw', 'abtt'], 'abtt')
    render(
      <AppProvider>
        <Harness />
      </AppProvider>,
    )

    await waitFor(() => {
      expect(predictionRequests().length).toBeGreaterThan(0)
    })
    // Displays the deployment default, since the stored choice is unserved...
    await waitFor(() => {
      expect(lastPredictionRequest()?.get('variant')).toBe('abtt')
    })
    // ...but the stored preference is untouched.
    expect(localStorage.getItem('locallatin-variant')).toBe('sif_abtt')
    // And the unserved variant is offered as disabled, not silently missing.
    expect(
      screen.getByRole('radio', { name: /^SIF\+ABTT/ }).hasAttribute('disabled'),
    ).toBe(true)
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
