import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LabelledKeysPanel from './LabelledKeysPanel'

const COMMA_KEY = 'Mansi 2, 359 B'

let requests: URL[] = []

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      requests.push(url)
      if (url.pathname === '/api/labelled_keys') {
        const search = url.searchParams.get('search') ?? ''
        const items =
          search === 'zzz'
            ? []
            : [
                { key: 'CNIC.325.1', unit_count: 5 },
                { key: COMMA_KEY, unit_count: 1 },
              ]
        return json({ items, total: items.length, page: 1, page_size: 50, has_more: false })
      }
      if (url.pathname.startsWith('/api/candidate_dir/')) {
        return json([{ filename: 'mansi_a.txt', text: 'Si quis episcopus' }])
      }
      return json({})
    }),
  )
}

beforeEach(() => {
  requests = []
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('LabelledKeysPanel', () => {
  it('lists keys with their unit counts', async () => {
    render(<LabelledKeysPanel />)
    expect(await screen.findByText('CNIC.325.1')).toBeTruthy()
    expect(screen.getByText('5 units')).toBeTruthy()
    expect(screen.getByText('1 unit')).toBeTruthy()
    expect(screen.getByTestId('labelled-key-count').textContent).toBe('2 keys')
  })

  it('searches keys and shows an empty state', async () => {
    render(<LabelledKeysPanel />)
    await screen.findByText('CNIC.325.1')

    await userEvent.type(screen.getByRole('textbox', { name: 'Search labelled keys' }), 'zzz')

    expect((await screen.findByText(/No keys match/)).textContent).toBe('No keys match “zzz”')
    expect(requests.slice(-1)[0].searchParams.get('search')).toBe('zzz')
  })

  it('opens a key to show its units, requesting the encoded name', async () => {
    render(<LabelledKeysPanel />)
    await userEvent.click(await screen.findByRole('button', { name: /Mansi 2, 359 B/ }))

    expect(await screen.findByText('mansi_a.txt')).toBeTruthy()
    expect(screen.getByText('Si quis episcopus')).toBeTruthy()
    await waitFor(() => {
      const units = requests.find((u) => u.pathname.startsWith('/api/candidate_dir/'))!
      expect(units.pathname).toBe(`/api/candidate_dir/${encodeURIComponent(COMMA_KEY)}/files`)
    })
  })

  it('reports a failed load', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ error: { message: 'boom' } }, 500)))
    render(<LabelledKeysPanel />)
    expect(await screen.findByText(/Failed to load keys: boom/)).toBeTruthy()
  })
})
