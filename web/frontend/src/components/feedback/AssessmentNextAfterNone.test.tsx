import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import { TokenProvider } from '../../contexts/TokenContext'
import FeedbackPanel from './FeedbackPanel'

/**
 * The way out of a recorded "None of the top N" answer (issue #221).
 *
 * Reported by the evaluators: the blue action writes the row itself, the panel
 * then clears the selection and the draft, and the bottom button -- withheld
 * while None is held so one decision cannot be written twice -- stayed disabled
 * with the receipt on screen. There was no forward move at all. The reviewer
 * had to hunt for the next fragment in the left panel, or press Skip, whose
 * hint asks for a note "so the PI can follow up": the wrong thing to say about
 * a fragment that has just been answered.
 *
 * The rule these tests pin down: once THIS reviewer has a recorded
 * none-of-top-N answer for the fragment on screen and no unsaved choice of
 * their own, the bottom button is a plain "Next" that moves them on and writes
 * NOTHING. It is not a second save; the answer is already in an append-only
 * log, and a button that wrote again would put it there twice.
 *
 * Three things are therefore asserted together throughout: the label, the move,
 * and the POST count. A "Next" that advanced by saving would pass the first two.
 */

const QUERY_ID = 7
const NEXT_ID = 11
/** A second fragment the reviewer can open by hand, mid-record. */
const ELSEWHERE_ID = 9
const MODEL = 'bowphs_LaTa'

interface SavedRow {
  outcome: string
  notes: string
  ccl_key: string | null
  ccl_key_action: string | null
  ccl_key_dir: string | null
  ccl_key_rank: number | null
}

let posted: Record<string, unknown>[] = []
/** What GET /api/feedback/latest answers: the merged prefill view (issue #96). */
let latest: SavedRow | null = null
/** The next-document read, and what it answers. */
let nextQueryRequests: string[] = []
let nextQueryFileId: number | null = NEXT_ID
let nextQueryFails: Error | null = null
/** Holds the record in flight, so a test can act while it is still pending. */
let postGate: Promise<void> | null = null
let releasePost: () => void = () => {}
let postAttempts = 0

function holdPost(): void {
  postGate = new Promise<void>((resolve) => {
    releasePost = resolve
  })
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function modelCard(rank: number) {
  return {
    rank,
    dir_name: `candidate-${rank}`,
    score: 1 - rank / 100,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'candidate text' }],
    source: 'model' as const,
  }
}

function installFetch(): void {
  posted = []
  nextQueryRequests = []
  nextQueryFileId = NEXT_ID
  nextQueryFails = null
  postGate = null
  postAttempts = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const raw = String(input)
      const url = new URL(raw, 'http://localhost')

      if (init?.method === 'POST' && url.pathname === '/api/feedback') {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>
        postAttempts += 1
        if (postGate) await postGate
        posted.push(body)
        if (body.outcome !== 'none_of_top_k') return jsonResponse({ success: true })
        const key = typeof body.ccl_key === 'string' ? body.ccl_key.trim() : ''
        const row = {
          id: posted.length,
          query_id: QUERY_ID,
          outcome: 'none_of_top_k',
          ccl_key: key || null,
          ccl_key_action: key ? 'created_reviewer_dir' : null,
          ccl_key_dir: key ? 'reviewer-dir-new' : null,
          ccl_key_rank: null,
        }
        latest = {
          outcome: 'none_of_top_k',
          notes: typeof body.notes === 'string' ? body.notes : '',
          ccl_key: row.ccl_key,
          ccl_key_action: row.ccl_key_action,
          ccl_key_dir: row.ccl_key_dir,
          ccl_key_rank: row.ccl_key_rank,
        }
        return jsonResponse(row, 201)
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
        return jsonResponse([
          {
            slug: MODEL,
            display_name: 'LaTa (T5)',
            layer: 4,
            pooling: 'sif',
            prediction_count: 2238,
            available_variants: ['sif_abtt'],
            default_variant: 'sif_abtt',
            confidence_bands: { no_match: 0.5, verify: 0.7 },
            supports_reviewer_dirs: true,
          },
        ])
      }
      if (url.pathname === '/api/feedback/latest') {
        // Per fragment, like the real route: an answer recorded on one is not
        // a prefill for another.
        const asked = Number(url.searchParams.get('query_id'))
        return jsonResponse(
          latest === null || asked !== QUERY_ID
            ? null
            : {
                id: 99,
                query_id: QUERY_ID,
                timestamp: '2026-09-18 10:00:00',
                model_slug: MODEL,
                variant: 'sif_abtt',
                correct_rank: 0,
                correct_dir: null,
                selected_ranks: null,
                reviewer: 'Bob Bibliothecarius',
                reviewer_account_id: 2,
                reviewer_username: 'bob',
                schema_version: 2,
                ...latest,
              },
        )
      }
      if (url.pathname === '/api/queries/next') {
        nextQueryRequests.push(url.search)
        if (nextQueryFails) throw nextQueryFails
        return jsonResponse({ file_id: nextQueryFileId })
      }
      if (url.pathname.endsWith('/predictions')) {
        const asked = Number(url.pathname.match(/\/api\/query\/(\d+)\//)?.[1])
        return jsonResponse({
          file_id: Number.isFinite(asked) ? asked : QUERY_ID,
          filename: `query-${asked}.txt`,
          model: MODEL,
          variant: 'sif_abtt',
          status: 'ok',
          predictions: [modelCard(1), modelCard(2)],
          seeded_dirs: [],
          reviewer_dir_candidates: [],
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
        open another fragment
      </button>
    </>
  )
}

function renderPanel() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <SavedDirectoryProvider accountKey="next-after-none">
            <PredictionProvider>
              <FeedbackProvider>
                <SelectQuery />
                <QueryProbe />
                <FeedbackPanel />
              </FeedbackProvider>
            </PredictionProvider>
          </SavedDirectoryProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

/** The bottom primary button, found by role rather than by its label. */
function primaryAction(): HTMLButtonElement {
  return screen.getByTestId('assessment-primary-action') as HTMLButtonElement
}

function activeQuery(): string {
  return screen.getByTestId('active-query').textContent ?? ''
}

async function pressNone(): Promise<void> {
  await userEvent.click(await screen.findByTestId('match-pill-none'))
}

/** Presses None and records the answer, with or without a key. */
async function recordNone(key?: string): Promise<void> {
  await pressNone()
  if (key) await userEvent.type(screen.getByTestId('ccl-key-input'), key)
  await userEvent.click(screen.getByTestId('none-of-top-k-submit'))
  await screen.findByTestId('none-of-top-k-recorded')
}

/** The stored answer a revisit finds, as the merged prefill serves it. */
const STORED_NONE: SavedRow = {
  outcome: 'none_of_top_k',
  notes: 'the source is elsewhere',
  ccl_key: 'CTOU.567.16',
  ccl_key_action: 'created_reviewer_dir',
  ccl_key_dir: 'reviewer-dir-abc',
  ccl_key_rank: null,
}

beforeEach(() => {
  latest = null
  localStorage.clear()
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('after recording a none-of-top-N answer', () => {
  it.each([
    ['with a key', 'CTOU.567.16'],
    ['without a key', undefined],
  ])('offers a Next that moves on and saves nothing (%s)', async (_name, key) => {
    renderPanel()
    await recordNone(key)

    const button = await waitFor(() => {
      const found = primaryAction()
      expect(found.disabled).toBe(false)
      return found
    })
    // The label has to say that nothing further is being written. "Submit &
    // Next" over an answer already in the log invites a second row.
    expect(button.textContent).toContain('Next')
    expect(button.textContent).not.toContain('Submit')

    await userEvent.click(button)

    await waitFor(() => expect(activeQuery()).toBe(String(NEXT_ID)))
    // The same read the save path performs, continuing from the fragment that
    // was answered.
    expect(nextQueryRequests).toEqual([`?after=${QUERY_ID}`])
    // One POST in the whole journey: the blue action's own.
    expect(posted).toHaveLength(1)
    expect(posted[0].outcome).toBe('none_of_top_k')
  })

  it('does not move on by itself, so the receipt can be read first', async () => {
    renderPanel()
    await recordNone('CTOU.567.16')

    expect(screen.getByTestId('none-of-top-k-receipt').textContent).toContain(
      'CTOU.567.16',
    )
    expect(activeQuery()).toBe(String(QUERY_ID))
    expect(nextQueryRequests).toHaveLength(0)
  })

  it('reports a failed move the way the save path does, and retries it', async () => {
    renderPanel()
    await recordNone()
    nextQueryFails = new TypeError('Failed to fetch')

    await userEvent.click(primaryAction())

    const notice = await screen.findByTestId('assessment-advance-error')
    expect(notice.textContent).toContain('recorded')
    expect(activeQuery()).toBe(String(QUERY_ID))

    nextQueryFails = null
    await userEvent.click(screen.getByTestId('assessment-advance-retry'))
    await waitFor(() => expect(activeQuery()).toBe(String(NEXT_ID)))
    expect(posted).toHaveLength(1)
  })

  it('ends the list exactly as the save path does', async () => {
    // `file_id: null` is a real answer from the real route: everything assigned
    // to this reviewer is done. Not an error, and not a document.
    nextQueryFileId = null
    renderPanel()
    await recordNone()

    await userEvent.click(primaryAction())
    await waitFor(() => expect(activeQuery()).toBe('null'))
    expect(screen.queryByTestId('assessment-advance-error')).toBeNull()
    expect(posted).toHaveLength(1)
  })
})

describe('coming back to a fragment this reviewer already answered', () => {
  it('offers the same Next, with no draft of their own on screen', async () => {
    latest = STORED_NONE
    renderPanel()

    await screen.findByTestId('none-of-top-k-recorded')
    await waitFor(() => expect(primaryAction().disabled).toBe(false))
    expect(primaryAction().textContent).toContain('Next')

    await userEvent.click(primaryAction())
    await waitFor(() => expect(activeQuery()).toBe(String(NEXT_ID)))
    expect(posted).toHaveLength(0)
  })

  it('turns back into a real save once a rank is chosen', async () => {
    latest = STORED_NONE
    renderPanel()
    await screen.findByTestId('none-of-top-k-recorded')

    await userEvent.click(await screen.findByTestId('match-pill-1'))
    const button = primaryAction()
    expect(button.textContent).toContain('Submit')
    expect(button.disabled).toBe(false)

    await userEvent.click(button)
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      query_id: QUERY_ID,
      outcome: 'matched_rank',
      correct_rank: 1,
    })
  })

  it('stays disabled on a colleague\'s answer, which is not this reviewer\'s', async () => {
    // The merged prefill carries the team's newest NOTE with the CALLER's own
    // decision. A row that is not this reviewer's none-of-top-N answer must not
    // enable a button that reads as "you have answered this".
    latest = {
      outcome: 'legacy_unresolved',
      notes: 'Abigail thinks this is CTOU.567.16',
      ccl_key: null,
      ccl_key_action: null,
      ccl_key_dir: null,
      ccl_key_rank: null,
    }
    renderPanel()

    await screen.findByTestId('match-pill-none')
    await waitFor(() =>
      expect(
        (screen.getByPlaceholderText(/note/i) as HTMLTextAreaElement).value,
      ).toContain('Abigail'),
    )
    expect(screen.queryByTestId('none-of-top-k-recorded')).toBeNull()
    expect(primaryAction().disabled).toBe(true)
    expect(primaryAction().textContent).toContain('Submit')
  })
})

describe('the Skip hint', () => {
  it('is not asked for over an answer that is already recorded', async () => {
    renderPanel()
    expect(screen.queryByTestId('skip-needs-note')).toBeTruthy()

    await recordNone('CTOU.567.16')
    await waitFor(() => expect(primaryAction().disabled).toBe(false))
    // "Add a note ... so the PI can follow up" is what a deferral needs. Beside
    // a finished answer and a live Next it reads as a demand for work already
    // done.
    expect(screen.queryByTestId('skip-needs-note')).toBeNull()
  })

  it('comes back if Skip is actually pressed there', async () => {
    renderPanel()
    await recordNone()
    await waitFor(() => expect(primaryAction().disabled).toBe(false))

    await userEvent.click(screen.getByRole('button', { name: 'Skip' }))
    expect(screen.getByTestId('skip-needs-note')).toBeTruthy()
    // Refusing the skip is not a save, and not a move.
    expect(posted).toHaveLength(1)
    expect(nextQueryRequests).toHaveLength(0)
  })
})

describe('a record that answers after the reviewer has moved on', () => {
  it('touches neither the new fragment\'s draft nor its button', async () => {
    holdPost()
    renderPanel()
    await pressNone()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))
    await waitFor(() => expect(postAttempts).toBe(1))

    // The reviewer opens another fragment and starts typing there.
    await userEvent.click(screen.getByTestId('go-elsewhere'))
    await waitFor(() => expect(activeQuery()).toBe(String(ELSEWHERE_ID)))
    const notes = screen.getByPlaceholderText(/note/i) as HTMLTextAreaElement
    await userEvent.type(notes, 'hand B throughout')

    releasePost()
    await waitFor(() => expect(posted).toHaveLength(1))
    await new Promise((resolve) => setTimeout(resolve, 50))

    // The answer belongs to the fragment it was made for. Here it must not
    // clear the unsaved draft, and must not offer a Next for an answer nobody
    // gave on this fragment.
    expect((screen.getByPlaceholderText(/note/i) as HTMLTextAreaElement).value).toBe(
      'hand B throughout',
    )
    expect(primaryAction().disabled).toBe(true)
    expect(primaryAction().textContent).toContain('Submit')
    expect(nextQueryRequests).toHaveLength(0)
  })
})

describe('while the blue form is open', () => {
  it('keeps the button disabled and explains itself in one line', async () => {
    renderPanel()
    await pressNone()

    expect(primaryAction().disabled).toBe(true)
    // A disabled button with no explanation reads as a broken app.
    const hint = screen.getByTestId('assessment-primary-blocked')
    expect(hint.textContent).toContain('Record this answer')
    // And the explanation reaches the control it is about, rather than only
    // sitting near it on screen.
    expect(hint.getAttribute('role')).toBe('status')
    expect(primaryAction().getAttribute('aria-describedby')).toBe(hint.id)
    expect(hint.id).toBeTruthy()
    expect(nextQueryRequests).toHaveLength(0)
  })

  it('keeps it disabled when Change reopens the form over a stored answer', async () => {
    latest = STORED_NONE
    renderPanel()
    await userEvent.click(await screen.findByTestId('none-of-top-k-change'))

    expect(primaryAction().disabled).toBe(true)
    expect(screen.getByTestId('assessment-primary-blocked')).toBeTruthy()
    expect(posted).toHaveLength(0)
  })

  it('shows no explanation when nothing is blocking the button', async () => {
    latest = STORED_NONE
    renderPanel()
    await screen.findByTestId('none-of-top-k-recorded')

    await waitFor(() => expect(primaryAction().disabled).toBe(false))
    expect(screen.queryByTestId('assessment-primary-blocked')).toBeNull()
    expect(primaryAction().getAttribute('aria-describedby')).toBeNull()
  })
})
