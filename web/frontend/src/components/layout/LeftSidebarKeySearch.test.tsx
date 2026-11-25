import { useEffect } from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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
      if (url.pathname === '/api/queries') {
        const key = url.searchParams.get('key')
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
})
