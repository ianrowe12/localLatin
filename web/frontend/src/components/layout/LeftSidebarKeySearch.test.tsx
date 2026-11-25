import { useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { FEEDBACK_UPDATED_EVENT } from '../../api/feedback'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import { TokenProvider } from '../../contexts/TokenContext'
import LeftSidebar from './LeftSidebar'

// jsdom has no layout, so Virtuoso would render no rows at all.
vi.mock('react-virtuoso', () => ({
  Virtuoso: ({
    totalCount,
    itemContent,
  }: {
    totalCount: number
    itemContent: (index: number) => React.ReactNode
  }) => (
    <div>{Array.from({ length: totalCount }, (_, i) => <div key={i}>{itemContent(i)}</div>)}</div>
  ),
}))

const MODEL = 'bowphs_LaTa'

function queryItem(fileId: number, keyMatches: unknown[] = []) {
  return {
    file_id: fileId,
    filename: `Hat42.${fileId}r.1.txt`,
    text_preview: 'Si quis episcopus',
    review_status: 'unreviewed',
    review_count: 0,
    key_matches: keyMatches,
  }
}

const SUGGESTIONS = [
  { key: 'DSIR.384.255 cap 7', unit_count: 5 },
  { key: 'DSIR.384.255 cap 8', unit_count: 1 },
]

let requests: URL[] = []
// Lets a test take over `/api/queries?key=...`; return null to use the default.
let keyListOverride: ((url: URL) => Promise<Response> | null) | null = null

const MODELS = [
  { slug: MODEL, display_name: 'LaTa' },
  { slug: 'bowphs_PhilTa', display_name: 'PhilTa' },
]

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
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
        return json({
          items: SUGGESTIONS,
          total: 2,
          page: 1,
          page_size: 8,
          has_more: false,
        })
      }
      if (url.pathname === '/api/models') return json(MODELS)
      if (url.pathname === '/api/queries') {
        const key = url.searchParams.get('key')
        const override = key ? keyListOverride?.(url) : null
        if (override) return override
        const items = key
          ? [
              queryItem(7, [
                { model: MODEL, rank: 3, dir_name: 'DSIR.384.255 cap 7' },
                { model: MODEL, rank: 9, dir_name: 'DSIR.384.255 cap 8' },
              ]),
              queryItem(8, [{ model: MODEL, rank: 1, dir_name: 'DSIR.384.255 cap 8' }]),
            ]
          : [queryItem(1)]
        return json({ items, total: items.length, page: 1, page_size: 50, has_more: false })
      }
      return json({})
    }),
  )
}

function queryListRequests(): URLSearchParams[] {
  return requests.filter((u) => u.pathname === '/api/queries').map((u) => u.searchParams)
}

function UseModel() {
  const { setActiveModel } = useApp()
  useEffect(() => setActiveModel(MODEL), [setActiveModel])
  return null
}

function renderSidebar() {
  return render(
    <ReviewerProvider>
      <AppProvider>
        <TokenProvider>
          <UseModel />
          <LeftSidebar isOpen onToggle={() => {}} />
        </TokenProvider>
      </AppProvider>
    </ReviewerProvider>,
  )
}

async function openKeyMode() {
  renderSidebar()
  await screen.findByText('Hat42.1r.1.txt')
  await userEvent.click(screen.getByRole('radio', { name: 'Suggested key' }))
  return screen.getByRole('textbox', { name: 'Search suggested keys' })
}

beforeEach(() => {
  requests = []
  keyListOverride = null
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('sidebar key search', () => {
  it('filters by key prefix using the selected model and variant', async () => {
    const box = await openKeyMode()
    expect(box.getAttribute('placeholder')).toBe('Search suggested keys, e.g. CNIC.325 or D')

    await userEvent.type(box, 'DSIR.384')

    await waitFor(() => {
      const last = queryListRequests().slice(-1)[0]
      expect(last.get('key')).toBe('DSIR.384')
    })
    const last = queryListRequests().slice(-1)[0]
    expect(last.get('key_match')).toBe('prefix')
    expect(last.get('key_model')).toBe(MODEL)
    expect(last.get('variant')).toBe('sif_abtt')
    expect(last.has('key_top1')).toBe(false)
    expect(last.has('search')).toBe(false)
  })

  it('shows the first match as a chip with +N and the number of texts', async () => {
    const box = await openKeyMode()
    await userEvent.type(box, 'DSIR')

    const chips = await screen.findAllByTestId('key-match-chip')
    expect(chips[0].textContent).toBe('DSIR.384.255 cap 7· #3+1')
    expect(chips[1].textContent).toBe('DSIR.384.255 cap 8· #1')
    expect(screen.getByTestId('key-match-count').textContent).toBe('2 texts')
  })

  it('offers up to 8 matching keys and switches to an exact match when one is picked', async () => {
    const box = await openKeyMode()
    await userEvent.type(box, 'DSIR')

    const list = await screen.findByRole('listbox', { name: 'Matching keys' })
    const option = within(list).getByRole('button', { name: /DSIR\.384\.255 cap 8/ })
    const suggestionRequest = requests.filter((u) => u.pathname === '/api/labelled_keys').slice(-1)[0]
    expect(suggestionRequest.searchParams.get('search')).toBe('DSIR')
    expect(suggestionRequest.searchParams.get('page_size')).toBe('8')

    await userEvent.click(option)

    await waitFor(() => {
      const last = queryListRequests().slice(-1)[0]
      expect(last.get('key')).toBe('DSIR.384.255 cap 8')
      expect(last.get('key_match')).toBe('exact')
    })
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('narrows to top suggestions and restores the normal list when cleared', async () => {
    const box = await openKeyMode()
    await userEvent.type(box, 'DSIR')
    await screen.findAllByTestId('key-match-chip')

    await userEvent.click(screen.getByRole('checkbox', { name: 'Top suggestion only' }))
    await waitFor(() => expect(queryListRequests().slice(-1)[0].get('key_top1')).toBe('true'))

    fireEvent.change(box, { target: { value: '' } })
    await screen.findByText('Hat42.1r.1.txt')
    expect(screen.queryAllByTestId('key-match-chip')).toHaveLength(0)
    expect(screen.queryByTestId('key-match-count')).toBeNull()
  })

  it('keeps file name search as the default mode', async () => {
    renderSidebar()
    await screen.findByText('Hat42.1r.1.txt')
    const box = screen.getByRole('textbox', { name: 'Search files' })
    await userEvent.type(box, 'Hat')
    await waitFor(() => expect(queryListRequests().slice(-1)[0].get('search')).toBe('Hat'))
    expect(queryListRequests().slice(-1)[0].has('key')).toBe(false)
  })

  describe('when the request fails', () => {
    const failing = () =>
      Promise.resolve(new Response('{"detail":"boom"}', { status: 500 }))

    it('shows an error with no count and none of the unfiltered list', async () => {
      keyListOverride = failing
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')

      const alert = await screen.findByTestId('key-search-error')
      expect(alert.textContent).toContain('Couldn\u2019t load texts for this key. Try again.')
      expect(screen.queryByTestId('key-match-count')).toBeNull()
      expect(screen.queryByText('Hat42.1r.1.txt')).toBeNull()
      expect(screen.queryAllByTestId('key-match-chip')).toHaveLength(0)
    })

    it('does not carry a count from the previous key into the failure', async () => {
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await screen.findByTestId('key-match-count')

      keyListOverride = failing
      await userEvent.type(box, '.384')

      await screen.findByTestId('key-search-error')
      expect(screen.queryByTestId('key-match-count')).toBeNull()
      expect(screen.queryAllByTestId('key-match-chip')).toHaveLength(0)
    })

    it('retries the same request and recovers', async () => {
      keyListOverride = failing
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await screen.findByTestId('key-search-error')

      keyListOverride = null
      await userEvent.click(screen.getByRole('button', { name: 'Retry' }))

      expect(await screen.findAllByTestId('key-match-chip')).toHaveLength(2)
      expect(screen.queryByTestId('key-search-error')).toBeNull()
      expect(screen.getByTestId('key-match-count').textContent).toBe('2 texts')
    })

    it('hides the count when a refresh of the same key fails', async () => {
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await screen.findByTestId('key-match-count')

      keyListOverride = failing
      act(() => {
        window.dispatchEvent(new Event(FEEDBACK_UPDATED_EVENT))
      })

      await screen.findByTestId('key-search-error')
      expect(screen.queryByTestId('key-match-count')).toBeNull()
      expect(screen.queryAllByTestId('key-match-chip')).toHaveLength(0)
    })

    it('clears the error when the key is cleared', async () => {
      keyListOverride = failing
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await screen.findByTestId('key-search-error')

      fireEvent.change(box, { target: { value: '' } })
      await screen.findByText('Hat42.1r.1.txt')
      expect(screen.queryByTestId('key-search-error')).toBeNull()
    })
  })

  it('ignores a slow response for an earlier key', async () => {
    let releaseSlow: (res: Response) => void = () => {}
    keyListOverride = (url) => {
      const key = url.searchParams.get('key')
      if (key === 'D') {
        return new Promise<Response>((resolve) => {
          releaseSlow = resolve
        })
      }
      if (key === 'DS') {
        return Promise.resolve(
          json({
            items: [queryItem(8, [{ model: MODEL, rank: 1, dir_name: 'DSIR.384.255 cap 8' }])],
            total: 1,
            page: 1,
            page_size: 50,
            has_more: false,
          }),
        )
      }
      return null
    }
    const box = await openKeyMode()
    fireEvent.change(box, { target: { value: 'D' } })
    await waitFor(() => expect(queryListRequests().slice(-1)[0].get('key')).toBe('D'))
    fireEvent.change(box, { target: { value: 'DS' } })
    await waitFor(() => expect(screen.getByTestId('key-match-count').textContent).toBe('1 text'))

    releaseSlow(
      json({
        items: [queryItem(7), queryItem(8), queryItem(9)],
        total: 3,
        page: 1,
        page_size: 50,
        has_more: false,
      }),
    )
    await new Promise((resolve) => setTimeout(resolve, 20))

    expect(screen.getByTestId('key-match-count').textContent).toBe('1 text')
    expect(screen.queryByText('Hat42.9r.1.txt')).toBeNull()
  })

  it('does not show the previous key\u2019s list while a new key loads', async () => {
    const box = await openKeyMode()
    await userEvent.type(box, 'DSIR')
    await screen.findByTestId('key-match-count')

    keyListOverride = () => new Promise<Response>(() => {})
    fireEvent.change(box, { target: { value: 'CNIC' } })

    await screen.findByText('Loading...')
    expect(screen.queryByTestId('key-match-count')).toBeNull()
    expect(screen.queryAllByTestId('key-match-chip')).toHaveLength(0)
  })

  describe('model scope', () => {
    const TWO_MODELS = [
      queryItem(7, [
        { model: MODEL, rank: 3, dir_name: 'DSIR.384.255 cap 7' },
        { model: 'bowphs_PhilTa', rank: 5, dir_name: 'DSIR.384.255 cap 8' },
      ]),
    ]

    it('defaults to this model and names no model on the chip', async () => {
      const box = await openKeyMode()
      expect(screen.getByRole('radio', { name: 'This model' }).getAttribute('aria-checked')).toBe('true')
      await userEvent.type(box, 'DSIR')

      await screen.findAllByTestId('key-match-chip')
      expect(queryListRequests().slice(-1)[0].get('key_model')).toBe(MODEL)
      expect(screen.queryAllByTestId('key-match-model')).toHaveLength(0)
    })

    it('omits key_model for all models and names the model on each chip', async () => {
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await screen.findAllByTestId('key-match-chip')

      keyListOverride = () =>
        Promise.resolve(
          json({ items: TWO_MODELS, total: 1, page: 1, page_size: 50, has_more: false }),
        )
      await userEvent.click(screen.getByRole('radio', { name: 'All models' }))

      await waitFor(() => {
        const last = queryListRequests().slice(-1)[0]
        expect(last.has('key_model')).toBe(false)
        expect(last.get('variant')).toBe('sif_abtt')
        expect(last.get('key')).toBe('DSIR')
      })
      const chip = await screen.findByTestId('key-match-chip')
      expect(chip.textContent).toBe('DSIR.384.255 cap 7· #3· LaTa+1')
      expect(chip.getAttribute('title')).toBe(
        'DSIR.384.255 cap 7 · rank 3 · LaTa\nDSIR.384.255 cap 8 · rank 5 · PhilTa',
      )
    })

    it('goes back to this model and drops the model name', async () => {
      const box = await openKeyMode()
      await userEvent.type(box, 'DSIR')
      await userEvent.click(screen.getByRole('radio', { name: 'All models' }))
      await waitFor(() => expect(queryListRequests().slice(-1)[0].has('key_model')).toBe(false))
      await screen.findAllByTestId('key-match-model')

      await userEvent.click(screen.getByRole('radio', { name: 'This model' }))
      await waitFor(() => expect(queryListRequests().slice(-1)[0].get('key_model')).toBe(MODEL))
      await waitFor(() => expect(screen.queryAllByTestId('key-match-model')).toHaveLength(0))
    })
  })
})
