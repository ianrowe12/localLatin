import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import FeedbackPanel from './FeedbackPanel'

/**
 * The read that moves the reviewer on after a save (issue #171 repair).
 *
 * A save schedules `GET /api/queries/next` on a 500ms timer that nothing
 * cancels, and the panel used to discard the resulting promise with `void`.
 * Two things followed. The timer kept reading on behalf of a panel that had
 * already been unmounted, which is how the suite ended with a request in
 * flight past its own fixture and a rejection with no catch block anywhere
 * ("UNHANDLED REJECTION: ApiError: Could not reach the server"). And in a
 * browser a failed advance was completely silent: the reviewer stayed on the
 * document they had just assessed with no word of why, the likeliest next act
 * being a second save into an append-only log.
 *
 * These tests drive the real panel and assert on what the fixture observes --
 * whether a request was issued at all, and what the reviewer is told -- rather
 * than on rejection timing, so a regression fails here deterministically
 * instead of surfacing as a straggler in whatever file runs next.
 */

const QUERY_ID = 7
const ELSEWHERE_ID = 9
const NEXT_ID = 11
const MODEL = 'bowphs_LaTa'

let nextQueryRequests = 0
let nextQueryFails: Error | null = null
let posted: Record<string, unknown>[] = []
let postFails: Error | null = null
/** Holds the save in flight so a test can act while it is still pending. */
let postGate: Promise<void> | null = null
let releasePost: () => void = () => {}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function holdPost(): void {
  postGate = new Promise<void>((resolve) => {
    releasePost = resolve
  })
}

const settle = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

/** Longer than the panel's 500ms advance timer, so it has certainly fired. */
const PAST_ADVANCE_TIMER = 900

function installFetch(): void {
  nextQueryRequests = 0
  nextQueryFails = null
  posted = []
  postFails = null
  postGate = null
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (init?.method === 'POST' && url.includes('/api/feedback')) {
        if (postGate) await postGate
        if (postFails) throw postFails
        posted.push(JSON.parse(String(init.body)))
        return jsonResponse({ success: true })
      }
      if (url.includes('/api/auth/me')) {
        return jsonResponse({
          id: 2,
          username: 'bob',
          display_name: 'Bob Bibliothecarius',
          role: 'reviewer',
          approval_status: 'approved',
        })
      }
      if (url.includes('/api/models')) {
        return jsonResponse([
          {
            slug: MODEL,
            display_name: MODEL,
            layer: 4,
            pooling: 'sif',
            prediction_count: 2238,
            available_variants: ['sif_abtt'],
            default_variant: 'sif_abtt',
            confidence_bands: { no_match: 0.2, verify: 0.4 },
            supports_reviewer_dirs: true,
          },
        ])
      }
      if (url.includes('/api/feedback/latest')) return jsonResponse(null)
      if (url.includes('/api/queries/next')) {
        nextQueryRequests += 1
        // A dropped connection, which is what `fetch` does when no response
        // arrives at all. `apiFetch` turns it into ApiError('network').
        if (nextQueryFails) throw nextQueryFails
        return jsonResponse({ file_id: NEXT_ID })
      }
      if (url.includes('/predictions')) {
        const params = new URL(url, 'http://localhost').searchParams
        return jsonResponse({
          file_id: Number(params.get('file_id') ?? QUERY_ID),
          filename: 'query.txt',
          model: MODEL,
          variant: 'sif_abtt',
          status: 'ok',
          predictions: [1, 2].map((rank) => ({
            rank,
            dir_name: `candidate-${rank}`,
            score: 1 - rank / 100,
            dir_files: [`${rank}.txt`],
            preview_text: 'preview',
            candidate_files: [{ filename: `${rank}.txt`, text: 'incipit textus' }],
            source: 'model',
          })),
          seeded_dirs: [],
        })
      }
      return jsonResponse(null)
    }),
  )
}

function SelectQuery() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return null
}

/** What the reviewer is looking at, and a way to look at something else. */
function QueryProbe() {
  const { activeQueryId, setActiveQueryId } = useApp()
  return (
    <>
      <span data-testid="active-query">{String(activeQueryId)}</span>
      <button
        type="button"
        data-testid="go-elsewhere"
        onClick={() => setActiveQueryId(ELSEWHERE_ID)}
      >
        open another document
      </button>
    </>
  )
}

function renderPanel() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <PredictionProvider>
          <FeedbackProvider>
            <SelectQuery />
            <QueryProbe />
            <FeedbackPanel />
          </FeedbackProvider>
        </PredictionProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

function activeQuery(): string {
  return screen.getByTestId('active-query').textContent ?? ''
}

function submitButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /Submit/ }) as HTMLButtonElement
}

function notesBox(): HTMLTextAreaElement {
  return screen.getByPlaceholderText(
    'Add notes for this query...',
  ) as HTMLTextAreaElement
}

/** Chooses rank 1 and saves it, leaving the advance timer running. */
async function assessAndSubmit(): Promise<void> {
  await userEvent.click(await screen.findByTestId('match-pill-1'))
  await userEvent.click(submitButton())
}

beforeEach(() => {
  installFetch()
})

describe('the move to the next document is a read like any other', () => {
  it('moves the reviewer on when it succeeds', async () => {
    renderPanel()
    await assessAndSubmit()

    await waitFor(() => expect(activeQuery()).toBe(String(NEXT_ID)), {
      timeout: 3000,
    })
    expect(nextQueryRequests).toBe(1)
    expect(screen.queryByTestId('assessment-advance-error')).toBeNull()
  })

  it('tells the reviewer when it fails, without claiming the save failed', async () => {
    nextQueryFails = new TypeError('Failed to fetch')
    renderPanel()
    await assessAndSubmit()

    const notice = await screen.findByTestId(
      'assessment-advance-error',
      {},
      { timeout: 3000 },
    )
    // The save itself went through, so this must not read as a failed save:
    // that would send the reviewer back to an append-only log to try again.
    expect(posted).toHaveLength(1)
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(notice.textContent).toContain('recorded')
    expect(notice.textContent).toContain('Could not reach the server.')
    // And they are still where they were, not silently somewhere else.
    expect(activeQuery()).toBe(String(QUERY_ID))
  })

  it('retries the move without saving anything again', async () => {
    nextQueryFails = new TypeError('Failed to fetch')
    renderPanel()
    await assessAndSubmit()
    await screen.findByTestId('assessment-advance-error', {}, { timeout: 3000 })

    nextQueryFails = null
    await userEvent.click(screen.getByTestId('assessment-advance-retry'))

    await waitFor(() => expect(activeQuery()).toBe(String(NEXT_ID)), {
      timeout: 3000,
    })
    expect(screen.queryByTestId('assessment-advance-error')).toBeNull()
    expect(posted).toHaveLength(1)
  })

  it('leaves the saved answer and the empty draft alone when it fails', async () => {
    nextQueryFails = new TypeError('Failed to fetch')
    renderPanel()
    await userEvent.click(await screen.findByTestId('match-pill-1'))
    await userEvent.type(notesBox(), 'hand B throughout')
    await userEvent.click(submitButton())

    await screen.findByTestId('assessment-advance-error', {}, { timeout: 3000 })
    // The save consumed the draft, exactly as it does when the advance works.
    // A failed advance neither resurrects it nor edits what was sent.
    expect(posted).toHaveLength(1)
    expect(posted[0]).toMatchObject({
      query_id: QUERY_ID,
      outcome: 'matched_rank',
      correct_rank: 1,
      notes: 'hand B throughout',
    })
    expect(notesBox().value).toBe('')
  })

  it('issues no read at all once the panel is gone', async () => {
    // The straggler that failed CI run 34546423664: the timer outlived the
    // panel, the fixture and the test file. Holding the save in flight until
    // after the unmount makes the ordering exact rather than a race.
    holdPost()
    const view = renderPanel()
    await assessAndSubmit()
    view.unmount()
    releasePost()

    await settle(PAST_ADVANCE_TIMER)
    expect(nextQueryRequests).toBe(0)
  })

  it('does not pull a reviewer off a document they have since opened', async () => {
    holdPost()
    renderPanel()
    await assessAndSubmit()
    await userEvent.click(screen.getByTestId('go-elsewhere'))
    releasePost()

    await settle(PAST_ADVANCE_TIMER)
    expect(activeQuery()).toBe(String(ELSEWHERE_ID))
    expect(nextQueryRequests).toBe(0)
  })

  it('never runs after a save that failed', async () => {
    postFails = new TypeError('Failed to fetch')
    renderPanel()
    await assessAndSubmit()

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    await settle(PAST_ADVANCE_TIMER)
    expect(nextQueryRequests).toBe(0)
    expect(screen.queryByTestId('assessment-advance-error')).toBeNull()
  })
})
