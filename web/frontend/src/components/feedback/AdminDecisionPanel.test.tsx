import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AdminDecisionPanel from './AdminDecisionPanel'
import type { AdminReviewItem, DecidingAnswer } from '../../api/adminReview'

let isAdmin = true
let queryId: number | null = 7
const posts: { url: string; body: unknown }[] = []
let postStatus = 201
vi.mock('../../contexts/ReviewerContext', () => ({
  useReviewer: () => ({ isPiAdmin: isAdmin, user: { id: 1 } }),
}))
vi.mock('../../contexts/AppContext', () => ({
  useApp: () => ({ activeQueryId: queryId }),
}))

function item(id: number): AdminReviewItem {
  return {
    xml_id: `witness.${id}`, query_id: id, status: 'conflict', reviewer_status: 'conflict',
    reviewer_count: 1, deciding_answer: null,
    answers: [{
      reviewer: 'Example Reviewer', model_slug: 'model_one', variant: 'sif_abtt',
      keys: ['CTOU.567.16', 'CARL.501.18'], none_of_these: false, unresolved: false,
      notes: `Please check witness ${id}`, timestamp: '2026-11-16T12:00:00Z',
    }],
  }
}
function reply(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}
beforeEach(() => {
  isAdmin = true
  queryId = 7
  posts.length = 0
  postStatus = 201
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (init?.method === 'POST') {
      const body = JSON.parse(String(init.body))
      posts.push({ url, body })
      if (postStatus !== 201) return reply({ detail: 'Could not save final answer' }, postStatus)
      return reply({
        ...body, id: posts.length, xml_id: `witness.${queryId}`, reviewer_account_id: 1,
        reviewer: 'Example Admin', timestamp: '2026-11-17T00:00:00Z',
      }, 201)
    }
    return reply(item(Number(url.split('/').pop())))
  }))
})
afterEach(() => vi.unstubAllGlobals())

describe('admin deciding answer', () => {
  it('shows other answers, records a key with a receipt, and appends an explicit none', async () => {
    const user = userEvent.setup()
    render(<AdminDecisionPanel />)
    await screen.findByLabelText('Final CCL key')
    await user.click(screen.getByText('Reviewer answers (1): Disagreement'))
    expect(screen.getByText('Example Reviewer: CTOU.567.16; CARL.501.18')).toBeTruthy()
    expect(screen.getByText('Please check witness 7')).toBeTruthy()
    await user.type(screen.getByLabelText('Final CCL key'), 'CTOU.567.16')
    await user.type(screen.getByLabelText('Decision note (optional)'), 'Checked against the reference')
    await user.click(screen.getByRole('button', { name: 'Record final answer' }))
    await screen.findByText('Final answer recorded.')
    expect(screen.getByText('Recorded final answer: CTOU.567.16')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Record final answer' })).toBeNull()
    expect(posts).toEqual([{ url: '/api/feedback/decisions/7', body: {
      ccl_key: 'CTOU.567.16', none_of_these: false, notes: 'Checked against the reference',
    } }])
    await user.click(screen.getByRole('button', { name: 'Change final answer' }))
    await user.click(screen.getByLabelText('None of these'))
    await user.click(screen.getByRole('button', { name: 'Record final answer' }))
    await screen.findByText('Recorded final answer: None of these')
    expect(posts).toHaveLength(2)
    expect(posts[1].body).toMatchObject({ ccl_key: null, none_of_these: true })
  })

  it('validates keys and notes with the shared safety rule', async () => {
    const user = userEvent.setup()
    render(<AdminDecisionPanel />)
    const key = await screen.findByLabelText('Final CCL key')
    await user.type(key, '=1+1')
    expect((await screen.findByRole('alert')).textContent).toContain('A CCL key cannot start')
    expect((screen.getByRole('button', { name: 'Record final answer' }) as HTMLButtonElement).disabled).toBe(true)
    await user.clear(key)
    await user.type(key, 'CTOU.567.16')
    await user.type(screen.getByLabelText('Decision note (optional)'), 'Note | other')
    expect((await screen.findByRole('alert')).textContent).toContain('A decision note cannot start')
    expect(posts).toHaveLength(0)
  })

  it('keeps an unsaved answer after failure and never claims it saved', async () => {
    const user = userEvent.setup()
    postStatus = 503
    render(<AdminDecisionPanel />)
    await user.type(await screen.findByLabelText('Final CCL key'), 'CTOU.567.16')
    await user.click(screen.getByRole('button', { name: 'Record final answer' }))
    expect((await screen.findByRole('alert')).textContent).toContain('Could not save final answer')
    expect((screen.getByLabelText('Final CCL key') as HTMLInputElement).value).toBe('CTOU.567.16')
    expect(screen.queryByText('Final answer recorded.')).toBeNull()
  })

  it('never renders or fetches for a reviewer or with no selected query', () => {
    isAdmin = false
    const { container, rerender } = render(<AdminDecisionPanel />)
    expect(container.textContent).toBe('')
    isAdmin = true
    queryId = null
    rerender(<AdminDecisionPanel />)
    expect(container.textContent).toBe('')
    expect(fetch).not.toHaveBeenCalled()
  })

  it('does not show an old answer or receipt when navigation races a save', async () => {
    const user = userEvent.setup()
    let finish!: (value: Response) => void
    const { rerender } = render(<AdminDecisionPanel />)
    await user.type(await screen.findByLabelText('Final CCL key'), 'CTOU.567.16')
    vi.mocked(fetch).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
    await user.click(screen.getByRole('button', { name: 'Record final answer' }))
    queryId = 8
    rerender(<AdminDecisionPanel />)
    await screen.findByLabelText('Final CCL key')
    finish(reply({
      id: 1, xml_id: 'witness.7', timestamp: '', reviewer_account_id: 1,
      reviewer: 'Example Admin', ccl_key: 'CTOU.567.16', notes: '',
    } satisfies DecidingAnswer, 201))
    await waitFor(() => expect((screen.getByLabelText('Final CCL key') as HTMLInputElement).value).toBe(''))
    expect(screen.queryByText('Final answer recorded.')).toBeNull()
    expect(screen.queryByText('Please check witness 7')).toBeNull()
  })
})
