import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
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
 * Three things followed. The timer kept reading on behalf of a panel that had
 * already been unmounted, which is how the suite ended with a request in
 * flight past its own fixture and a rejection with no catch block anywhere
 * ("UNHANDLED REJECTION: ApiError: Could not reach the server"). In a browser
 * a failed advance was completely silent: the reviewer stayed on the document
 * they had just assessed with no word of why, the likeliest next act being a
 * second save into an append-only log. And a completion was applied wherever
 * the reviewer had got to by then.
 *
 * A read belongs to the ASSESSMENT VISIT that produced it: one document, under
 * one model, one pipeline and one account, entered once. Anything weaker lets
 * a stale read speak for work it has nothing to do with, and "your response
 * was recorded" over a document nobody answered is a false receipt of exactly
 * the kind a reviewer acts on.
 *
 * These tests drive the real panel under the real providers and assert on what
 * the fixture observes -- which requests were issued and with what, and what is
 * legible on screen -- rather than on rejection timing, so a regression fails
 * here deterministically instead of surfacing as a straggler in whatever file
 * runs next.
 */

const QUERY_ID = 7
const ELSEWHERE_ID = 9
const NEXT_ID = 11
const MODEL = 'bowphs_LaTa'
const OTHER_MODEL = 'google_mt5-base'

let nextQueryRequests: string[] = []
let nextQueryFails: Error | null = null
let posted: Record<string, unknown>[] = []
let postFails: Error | null = null
/** Holds the save in flight so a test can act while it is still pending. */
let postGate: Promise<void> | null = null
let releasePost: () => void = () => {}
/** Counted when the request arrives, before the gate and before any failure. */
let postAttempts = 0
/** Holds the next-document read the same way, and answers it by hand. */
let readGate: Promise<Response> | null = null
let answerRead: (response: Response) => void = () => {}
let breakRead: (reason: Error) => void = () => {}
let readStarted = false

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

function holdRead(): void {
  readGate = new Promise<Response>((resolve, reject) => {
    answerRead = resolve
    breakRead = reject
  })
  // A test may finish without answering it. Disposing it must not itself be an
  // unhandled rejection, which would mask the one under test.
  readGate.catch(() => {})
}

const settle = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms))

/** Longer than the panel's 500ms advance timer, so it has certainly fired. */
const PAST_ADVANCE_TIMER = 900

/** Answers the held read, then lets the resulting state settle. */
async function resolveRead(response: Response): Promise<void> {
  await act(async () => {
    answerRead(response)
    await settle(20)
  })
}

async function rejectRead(reason: Error): Promise<void> {
  await act(async () => {
    breakRead(reason)
    await settle(20)
  })
}

function installFetch(): void {
  nextQueryRequests = []
  nextQueryFails = null
  readGate = null
  readStarted = false
  posted = []
  postAttempts = 0
  postFails = null
  postGate = null
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost')

      if (url.pathname === '/api/feedback' && init?.method === 'POST') {
        postAttempts += 1
        if (postGate) await postGate
        if (postFails) throw postFails
        posted.push(JSON.parse(String(init.body)))
        return jsonResponse({ success: true })
      }
      if (url.pathname === '/api/auth/me') {
        return jsonResponse({
          id: 2,
          username: 'bob',
          display_name: 'Bob Bibliothecarius',
          role: 'reviewer',
          approval_status: 'approved',
        })
      }
      if (url.pathname === '/api/models') {
        return jsonResponse(
          [MODEL, OTHER_MODEL].map((slug) => ({
            slug,
            display_name: slug,
            layer: 4,
            pooling: 'sif',
            prediction_count: 2238,
            available_variants: ['sif_abtt'],
            default_variant: 'sif_abtt',
            confidence_bands: { no_match: 0.2, verify: 0.4 },
            supports_reviewer_dirs: true,
          })),
        )
      }
      if (url.pathname === '/api/feedback/latest') return jsonResponse(null)
      if (url.pathname === '/api/queries/next') {
        readStarted = true
        nextQueryRequests.push(url.search)
        if (readGate) return readGate
        // A dropped connection, which is what `fetch` does when no response
        // arrives at all. `apiFetch` turns it into ApiError('network').
        if (nextQueryFails) throw nextQueryFails
        return jsonResponse({ file_id: NEXT_ID })
      }
      const predictions = url.pathname.match(/^\/api\/query\/(\d+)\/predictions$/)
      if (predictions) {
        return jsonResponse({
          file_id: Number(predictions[1]),
          filename: 'query.txt',
          model: url.searchParams.get('model'),
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

/** What the reviewer is looking at, and the ways they can change it. */
function QueryProbe() {
  const { activeQueryId, activeModel, setActiveQueryId, setActiveModel } = useApp()
  return (
    <>
      <span data-testid="active-query">{String(activeQueryId)}</span>
      <span data-testid="active-model">{activeModel}</span>
      <button
        type="button"
        data-testid="go-elsewhere"
        onClick={() => setActiveQueryId(ELSEWHERE_ID)}
      >
        open another document
      </button>
      <button
        type="button"
        data-testid="switch-model"
        onClick={() => setActiveModel(OTHER_MODEL)}
      >
        switch model
      </button>
      <button
        type="button"
        data-testid="come-back"
        onClick={() => setActiveQueryId(QUERY_ID)}
      >
        back to the first document
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

function advanceNotice(): HTMLElement | null {
  return screen.queryByTestId('assessment-advance-error')
}

/** Chooses rank 1 and saves it, leaving the advance timer running. */
async function assessAndSubmit(): Promise<void> {
  await userEvent.click(await screen.findByTestId('match-pill-1'))
  await userEvent.click(submitButton())
  // The save has left the browser. Whether it lands, fails or is held open is
  // the individual test's business.
  await waitFor(() => expect(postAttempts).toBe(1))
}

async function untilReadStarts(): Promise<void> {
  await waitFor(() => expect(readStarted).toBe(true), { timeout: 2000 })
}

/** Leaves the assessment the save was made on, in each way the UI allows. */
async function leaveTheAssessment(kind: 'query' | 'model' | 'return'): Promise<void> {
  if (kind === 'query') await userEvent.click(screen.getByTestId('go-elsewhere'))
  if (kind === 'model') await userEvent.click(screen.getByTestId('switch-model'))
  if (kind === 'return') {
    // Same document, same model, same account -- and still not the same visit.
    await userEvent.click(screen.getByTestId('go-elsewhere'))
    await userEvent.click(screen.getByTestId('come-back'))
  }
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
    expect(nextQueryRequests).toHaveLength(1)
    expect(advanceNotice()).toBeNull()
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
    expect(advanceNotice()).toBeNull()
    expect(posted).toHaveLength(1)
    // Both reads continue from the document that was answered, which is the
    // retry's own business: it does not read "whatever is on screen now".
    expect(nextQueryRequests).toEqual([`?after=${QUERY_ID}`, `?after=${QUERY_ID}`])
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
    expect(nextQueryRequests).toHaveLength(0)
  })

  it('never runs after a save that failed', async () => {
    postFails = new TypeError('Failed to fetch')
    renderPanel()
    await assessAndSubmit()

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    await settle(PAST_ADVANCE_TIMER)
    expect(nextQueryRequests).toHaveLength(0)
    expect(advanceNotice()).toBeNull()
  })
})

describe('a read belongs to the assessment visit that produced it', () => {
  const departures = [
    ['another document', 'query' as const, String(ELSEWHERE_ID)],
    ['another model', 'model' as const, String(QUERY_ID)],
    ['the same document, entered again', 'return' as const, String(QUERY_ID)],
  ] as const

  it.each(departures)(
    'starts no read once the reviewer has moved to %s',
    async (_name, kind, expectedQuery) => {
      // The save is held until after the move, so the 500ms timer fires with
      // the reviewer already elsewhere. Exact ordering, not a race.
      holdPost()
      renderPanel()
      await assessAndSubmit()
      await leaveTheAssessment(kind)
      releasePost()

      await settle(PAST_ADVANCE_TIMER)
      expect(nextQueryRequests).toHaveLength(0)
      expect(activeQuery()).toBe(expectedQuery)
      expect(posted).toHaveLength(1)
    },
  )

  it.each(departures)(
    'applies no late success once the reviewer has moved to %s',
    async (_name, kind, expectedQuery) => {
      holdRead()
      renderPanel()
      await assessAndSubmit()
      await untilReadStarts()
      await leaveTheAssessment(kind)
      await resolveRead(jsonResponse({ file_id: NEXT_ID }))

      expect(activeQuery()).toBe(expectedQuery)
      expect(posted).toHaveLength(1)
    },
  )

  it.each(departures)(
    'says nothing about a saved response once the reviewer has moved to %s',
    async (_name, kind) => {
      // The false receipt: "Your response was recorded" over an assessment
      // nobody made. Worst on the same document re-entered, where the sentence
      // names the right file and still the wrong visit.
      holdRead()
      renderPanel()
      await assessAndSubmit()
      await untilReadStarts()
      await leaveTheAssessment(kind)
      expect(advanceNotice()).toBeNull()

      await rejectRead(new TypeError('Failed to fetch'))

      expect(advanceNotice()).toBeNull()
      expect(document.body.textContent).not.toContain('Your response was recorded')
      expect(posted).toHaveLength(1)
    },
  )

  it('keeps a failure reported on the visit it does belong to', async () => {
    // The other side of the guard: an owned failure must still be visible.
    holdRead()
    renderPanel()
    await assessAndSubmit()
    await untilReadStarts()
    await rejectRead(new TypeError('Failed to fetch'))

    const notice = await screen.findByTestId('assessment-advance-error')
    expect(notice.textContent).toContain('recorded')
    expect(activeQuery()).toBe(String(QUERY_ID))
  })
})

describe('a 200 is not by itself a next document', () => {
  it.each([
    ['a null body', null],
    ['a body with no file_id', { next: 11 }],
    ['a file_id that is not a query', { file_id: 'eleven' }],
  ])('reports %s as a failed move rather than rejecting', async (_name, body) => {
    // `apiFetch<NextQueryResponse>` is a TypeScript assertion, not a check, so
    // all of these resolve. Reading `file_id` off them used to throw OUTSIDE
    // the catch, on a promise nobody holds: an unhandled rejection again, and
    // silent on screen.
    holdRead()
    renderPanel()
    await assessAndSubmit()
    await untilReadStarts()
    await resolveRead(jsonResponse(body))

    const notice = await screen.findByTestId('assessment-advance-error')
    expect(notice.textContent).toContain('recorded')
    expect(activeQuery()).toBe(String(QUERY_ID))
    expect(posted).toHaveLength(1)
  })

  it('treats file_id null as no next document, not as a failure', async () => {
    // A real answer from the real route: everything assigned to this reviewer
    // is done. It is not an error and must not be reported as one.
    holdRead()
    renderPanel()
    await assessAndSubmit()
    await untilReadStarts()
    await resolveRead(jsonResponse({ file_id: null }))

    await waitFor(() => expect(activeQuery()).toBe('null'))
    expect(advanceNotice()).toBeNull()
    expect(posted).toHaveLength(1)
  })
})
