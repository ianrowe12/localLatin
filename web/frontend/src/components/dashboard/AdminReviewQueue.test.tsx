import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AdminReviewQueue from './AdminReviewQueue'
import type { AdminReviewItem } from '../../api/adminReview'

const navigate = vi.fn()
let isAdmin = true
vi.mock('../../contexts/ReviewerContext', () => ({
  useReviewer: () => ({ isPiAdmin: isAdmin, user: { id: 1 } }),
}))
vi.mock('../../contexts/AppContext', () => ({
  useApp: () => ({ navigateToQuery: navigate }),
}))

const ITEM: AdminReviewItem = {
  xml_id: 'witness.1', query_id: 7, status: 'conflict', reviewer_status: 'conflict',
  reviewer_count: 2, deciding_answer: null,
  answers: [
    { reviewer: 'Example One', model_slug: 'model_one', variant: 'sif_abtt', keys: ['CTOU.567.16'], none_of_these: false, unresolved: false, notes: 'Check the ending', timestamp: '2026-11-16T12:00:00Z' },
    { reviewer: 'Example Two', model_slug: 'model_two', variant: null, keys: [], none_of_these: true, unresolved: false, notes: '', timestamp: '2026-11-16T12:01:00Z' },
  ],
}
const counts = { all: 21, conflict: 21, needs_check: 0, with_notes: 1, reviewed_once: 0, decided: 0 }
function reply(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

beforeEach(() => {
  isAdmin = true
  navigate.mockReset()
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    const page = Number(url.searchParams.get('page'))
    return reply({
      items: [{ ...ITEM, xml_id: page === 2 ? 'witness.2' : 'witness.1' }],
      counts, page, total: 21, page_size: 20, has_more: page === 1,
    })
  }))
})
afterEach(() => vi.unstubAllGlobals())

describe('admin review queue', () => {
  it('shows counted answers, counts and pagination, then opens the normal review view', async () => {
    const user = userEvent.setup()
    render(<AdminReviewQueue />)
    await screen.findByText('Example One: CTOU.567.16')
    expect(screen.getByText('Example Two: None of these')).toBeTruthy()
    expect(screen.getByText('Check the ending')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Disagreements (21)' })).toBeTruthy()
    await user.click(screen.getByRole('button', { name: 'Open witness.1' }))
    expect(navigate).toHaveBeenCalledWith(7)
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    await screen.findByRole('button', { name: 'Open witness.2' })
    expect(screen.queryByRole('button', { name: 'Open witness.1' })).toBeNull()
    await user.click(screen.getByRole('button', { name: 'With notes (1)' }))
    await waitFor(() => expect(fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('filter=with_notes&page=1'), expect.anything(),
    ))
  })

  it('renders nothing and makes no requests for reviewers', () => {
    isAdmin = false
    const { container } = render(<AdminReviewQueue />)
    expect(container.textContent).toBe('')
    expect(fetch).not.toHaveBeenCalled()
  })

  it('reports a load failure and retries explicitly', async () => {
    const user = userEvent.setup()
    vi.mocked(fetch).mockResolvedValueOnce(reply({ detail: 'Queue unavailable' }, 503))
    render(<AdminReviewQueue />)
    expect((await screen.findByRole('alert')).textContent).toContain('Queue unavailable')
    await user.click(screen.getByRole('button', { name: 'Refresh queue' }))
    await screen.findByRole('button', { name: 'Open witness.1' })
    expect(screen.queryByRole('alert')).toBeNull()
  })
})
