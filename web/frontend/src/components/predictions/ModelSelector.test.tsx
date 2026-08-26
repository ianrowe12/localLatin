import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider } from '../../contexts/AppContext'
import ModelSelector from './ModelSelector'

/** /api/models is sorted by display name, so mT5-base is never first. */
function model(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'mean',
    prediction_count: 2238,
    available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
    default_variant: 'sif_abtt',
  }
}

let models = [
  model('KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5', 'KaLM-mini'),
  model('sentence-transformers_LaBSE', 'LaBSE'),
  model('bowphs_LaTa', 'LaTa'),
  model('google_mt5-base', 'mT5-base'),
]

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const body = url.includes('/api/models') ? models : {}
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
}

function renderSelector() {
  return render(
    <AppProvider>
      <ModelSelector />
    </AppProvider>,
  )
}

beforeEach(() => {
  models = [
    model('KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5', 'KaLM-mini'),
    model('sentence-transformers_LaBSE', 'LaBSE'),
    model('bowphs_LaTa', 'LaTa'),
    model('google_mt5-base', 'mT5-base'),
  ]
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ModelSelector default (issue #94)', () => {
  it('defaults to mT5-base even though it sorts last', async () => {
    renderSelector()
    const select = screen.getByLabelText('Select model') as HTMLSelectElement
    await waitFor(() => {
      expect(select.value).toBe('google_mt5-base')
    })
    // The reviewer reads a display name, so the slug must map to one.
    expect(select.selectedOptions[0].textContent).toBe('mT5-base')
  })

  it('falls back to the first served model where mT5-base is absent', async () => {
    models = [model('bowphs_LaTa', 'LaTa'), model('bowphs_PhilTa', 'PhilTa')]
    renderSelector()
    const select = screen.getByLabelText('Select model') as HTMLSelectElement
    await waitFor(() => {
      expect(select.value).toBe('bowphs_LaTa')
    })
  })
})
