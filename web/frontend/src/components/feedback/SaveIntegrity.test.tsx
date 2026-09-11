import { useEffect, useState } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import {
  FeedbackProvider,
  useFeedback,
  type FeedbackContextValue,
  type SaveOutcome,
} from '../../contexts/FeedbackContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider, useReviewer } from '../../contexts/ReviewerContext'
import { TokenProvider } from '../../contexts/TokenContext'
import FeedbackPanel from './FeedbackPanel'
import RightSidebar from '../layout/RightSidebar'
import type { FeedbackEntry } from '../../api/feedback'

/**
 * Save and Skip answer to the server, not to the click (issue #158).
 *
 * Both buttons used to fire and forget: `submitFeedback()` was called without
 * `await`, the panel scheduled a move to the next document on a timer, and the
 * draft was deleted before any answer arrived. A slow POST could therefore be
 * clicked twice into an append-only log, a reviewer who kept typing could be
 * carried off the document with their newer text destroyed, and a save that
 * never landed still looked like one.
 *
 * Every POST and every next-document lookup in this file is held open by the
 * test, so each of those windows is a place the test can stand rather than a
 * race it has to hope for. The components under test are the real ones, driven
 * through App.tsx's provider tree.
 */

const QUERY_A = 7
const QUERY_B = 8
const QUERY_C = 9
const MODEL = 'bowphs_LaTa'
const DRAFT_STORAGE_KEY = 'locallatin-feedback-drafts'
/** The panel waits this long after a save before it looks for the next query. */
const ADVANCE_DELAY_MS = 500

const draftKeyFor = (queryId: number, accountId = 2) =>
  `acct${accountId}:${queryId}-${MODEL}-sif_abtt`

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function modelCard(rank: number, dir?: string) {
  return {
    rank,
    dir_name: dir ?? `candidate-${rank}`,
    score: 1 - rank / 100,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'candidate text' }],
    source: 'model',
  }
}

let account = { id: 2, username: 'bob', display_name: 'Bob Bibliothecarius' }
let postPayloads: Record<string, unknown>[] = []
let postGates: Deferred<Response>[] = []
let nextGates: Deferred<Response>[] = []
let latestGates: Deferred<Response>[] = []
/** When false, `/api/feedback/latest` answers immediately with `latestEntry`. */
let holdLatest = false
let latestEntry: FeedbackEntry | null = null

/**
 * The row a real, accepted POST answers with.
 *
 * Built from the payload the way the route does: the canonical rank is the
 * first selection, its directory is resolved from the identities the reviewer
 * saw, and a skip or a none-of-these keeps neither.
 */
function receiptFor(payload: Record<string, unknown>, overrides = {}): FeedbackEntry {
  const ranks = Array.isArray(payload.selected_ranks)
    ? (payload.selected_ranks as number[])
    : null
  const rank = payload.outcome === 'skipped' ? null : (payload.correct_rank as number)
  const dirs = (payload.expected_candidate_dirs ?? {}) as Record<string, string>
  return {
    id: 900 + postPayloads.length,
    query_id: Number(payload.query_id),
    timestamp: '2026-09-10 12:00:00',
    model_slug: String(payload.model_slug),
    variant: (payload.variant ?? 'sif_abtt') as FeedbackEntry['variant'],
    outcome: payload.outcome as FeedbackEntry['outcome'],
    correct_rank: rank,
    correct_dir: payload.outcome === 'matched_rank' ? (dirs[String(rank)] ?? null) : null,
    selected_ranks: ranks !== null && ranks.length > 0 ? ranks : null,
    notes: String(payload.notes ?? ''),
    reviewer: account.display_name,
    reviewer_account_id: account.id,
    reviewer_username: account.username,
    schema_version: 2,
    ...overrides,
  }
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const path = new URL(url, 'http://localhost').pathname
      const params = new URL(url, 'http://localhost').searchParams

      // Every save is held open, so the test decides when -- and whether --
      // the server answers.
      if (init?.method === 'POST' && path === '/api/feedback') {
        postPayloads.push(JSON.parse(String(init.body)) as Record<string, unknown>)
        const gate = deferred<Response>()
        postGates.push(gate)
        return gate.promise
      }
      if (path === '/api/auth/me') {
        return jsonResponse({ ...account, role: 'reviewer', approval_status: 'approved' })
      }
      if (path === '/api/models') {
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
      if (path === '/api/feedback/latest') {
        if (!holdLatest) return jsonResponse(latestEntry)
        const gate = deferred<Response>()
        latestGates.push(gate)
        return gate.promise
      }
      if (path === '/api/queries/next') {
        const gate = deferred<Response>()
        nextGates.push(gate)
        return gate.promise
      }
      const predictions = /^\/api\/query\/(\d+)\/predictions$/.exec(path)
      if (predictions) {
        return jsonResponse({
          file_id: Number(predictions[1]),
          filename: `query-${predictions[1]}.txt`,
          model: params.get('model') ?? MODEL,
          variant: 'sif_abtt',
          status: 'ok',
          predictions: [modelCard(1), modelCard(2)],
          seeded_dirs: [],
        })
      }
      const query = /^\/api\/query\/(\d+)$/.exec(path)
      if (query) {
        return jsonResponse({
          file_id: Number(query[1]),
          filename: `query-${query[1]}.txt`,
          text: 'query text under review',
          dir_name: null,
        })
      }
      return jsonResponse(null)
    }),
  )
}

/** Answer a held save. Defaults to the receipt the real router would send. */
async function settlePost(
  index = 0,
  options: { status?: number; body?: unknown; error?: Error } = {},
): Promise<void> {
  const gate = postGates[index]
  if (gate === undefined) throw new Error(`no POST #${index} to settle`)
  await act(async () => {
    if (options.error !== undefined) gate.reject(options.error)
    else {
      const body =
        'body' in options ? options.body : receiptFor(postPayloads[index] ?? {})
      gate.resolve(jsonResponse(body, options.status ?? 201))
    }
    await Promise.resolve()
  })
}

/** Answer a held next-document lookup. */
async function settleNext(
  index = 0,
  options: { fileId?: number; status?: number; error?: Error } = {},
): Promise<void> {
  const gate = nextGates[index]
  if (gate === undefined) throw new Error(`no next-query lookup #${index} to settle`)
  await act(async () => {
    if (options.error !== undefined) gate.reject(options.error)
    else if (options.status !== undefined && options.status >= 400) {
      gate.resolve(
        jsonResponse(
          { error: { code: 'internal_error', message: 'The queue is unavailable.' } },
          options.status,
        ),
      )
    } else gate.resolve(jsonResponse({ file_id: options.fileId ?? QUERY_B }))
    await Promise.resolve()
  })
}

/** Let the post-save timer fire (or prove that nothing follows it). */
async function letTheTimerFire(): Promise<void> {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ADVANCE_DELAY_MS + 150))
  })
}

let feedbackApi: FeedbackContextValue | null = null
let appApi: ReturnType<typeof useApp> | null = null
let reviewerApi: ReturnType<typeof useReviewer> | null = null

function Harness({ queryId = QUERY_A }: { queryId?: number }) {
  const app = useApp()
  feedbackApi = useFeedback()
  appApi = app
  reviewerApi = useReviewer()
  const { setActiveQueryId, setActiveModel } = app
  useEffect(() => {
    setActiveQueryId(queryId)
    setActiveModel(MODEL)
  }, [queryId, setActiveModel, setActiveQueryId])
  return <div data-testid="active-query">{String(app.activeQueryId)}</div>
}

/**
 * The sidebar as the app mounts it: collapsing it really does unmount the
 * panel and its controls, which is the point of the remount tests.
 */
function SidebarHost() {
  const [isOpen, setIsOpen] = useState(true)
  return <RightSidebar isOpen={isOpen} onToggle={() => setIsOpen((open) => !open)} />
}

function renderPanel(options: { sidebar?: boolean } = {}) {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <PredictionProvider>
            <FeedbackProvider>
              <Harness />
              {options.sidebar === true ? <SidebarHost /> : <FeedbackPanel />}
            </FeedbackProvider>
          </PredictionProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

function notesBox(): HTMLTextAreaElement {
  return screen.getByPlaceholderText('Add notes for this query...') as HTMLTextAreaElement
}

function submitButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /Submit/ }) as HTMLButtonElement
}

function skipButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /Skip/ }) as HTMLButtonElement
}

function storedDrafts(): Record<string, unknown> {
  const raw = localStorage.getItem(DRAFT_STORAGE_KEY)
  return Object.fromEntries(JSON.parse(raw ?? '[]') as [string, unknown][])
}

function activeQueryId(): string {
  return screen.getByTestId('active-query').textContent ?? ''
}

/** Choose the first candidate, which is all a submit needs to be savable. */
async function chooseFirstCandidate(): Promise<void> {
  await userEvent.click(await screen.findByTestId('match-pill-1'))
}

beforeEach(() => {
  localStorage.clear()
  account = { id: 2, username: 'bob', display_name: 'Bob Bibliothecarius' }
  postPayloads = []
  postGates = []
  nextGates = []
  latestGates = []
  holdLatest = false
  latestEntry = null
  feedbackApi = null
  appApi = null
  reviewerApi = null
  installFetch()
})

// A held request can outlive its test; the fixture stays installed so a
// straggler hits the mock rather than jsdom's real fetch.
afterAll(() => {
  vi.unstubAllGlobals()
})

describe('a save is not finished until the server says so', () => {
  it('keeps the draft, and the reviewer, in place while the POST is out', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'hand A throughout')
    await userEvent.click(submitButton())

    // In flight: both controls are locked, the answer is still on screen, and
    // nothing has been cleared on the strength of a click.
    await waitFor(() => expect(postGates).toHaveLength(1))
    expect(await screen.findByTestId('submit-spinner')).toBeTruthy()
    expect(submitButton().disabled).toBe(true)
    expect(skipButton().disabled).toBe(true)
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })

    await settlePost()

    // Acknowledged: the draft goes only now, and only because a receipt came.
    await waitFor(() => expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined())
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(postPayloads).toHaveLength(1)
  })

  it('moves to the next document only after the receipt and the lookup', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    // No receipt yet, so no lookup: the panel cannot owe a move it has not
    // earned.
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))

    await settlePost()
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })
    await settleNext(0, { fileId: QUERY_B })
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_B)))
  })

  it('keeps the draft and refuses to claim a save when the POST is rejected', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'rank 11 now points elsewhere')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      status: 409,
      body: {
        error: {
          code: 'CANDIDATE_IDENTITY_CHANGED',
          message: 'Rank 1 now resolves to a different directory.',
        },
      },
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('rejected')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
    expect(screen.queryByTestId('submit-check')).toBeNull()
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))
  })

  it('treats an unreadable receipt as an unknown outcome, not a success', async () => {
    // The old client took HTTP 200 as proof. A proxy's cached page or a
    // truncated body would then delete a draft nobody can show was saved.
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, { status: 200, body: { success: true } })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(error.textContent).toContain('cannot tell whether your assessment was recorded')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
  })

  it('refuses a receipt that belongs to another assessment', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      body: receiptFor(postPayloads[0], { query_id: QUERY_C }),
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
  })

  it('lets the same assessment be sent again after a failure', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost(0, { error: new TypeError('Failed to fetch') })
    await screen.findByTestId('assessment-save-error')

    // The lock is released by the failure: a retry is the reviewer's to make.
    await waitFor(() => expect(submitButton().disabled).toBe(false))
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(2))
    await settlePost(1)
    await waitFor(() => expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined())
    expect(postPayloads).toHaveLength(2)
  })
})

describe('one assessment, one request', () => {
  it('posts once when a save is asked for twice in the same tick', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await waitFor(() => expect(feedbackApi?.draftKey).toBe(draftKeyFor(QUERY_A)))

    // Straight at the context, which is where the lock has to live: the button
    // has its own guard, but a second caller must not slip past it.
    let first!: Promise<SaveOutcome>
    let second!: Promise<SaveOutcome>
    act(() => {
      first = feedbackApi!.submitFeedback()
      second = feedbackApi!.submitFeedback()
    })
    await waitFor(() => expect(postGates).toHaveLength(1))
    expect(await second).toEqual({ status: 'already_pending' })

    await settlePost()
    expect((await first).status).toBe('saved')
    expect(postPayloads).toHaveLength(1)
  })

  it('posts once when Submit and Skip are asked for together', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'waiting on the PI')
    await waitFor(() => expect(feedbackApi?.draft.notes).toBe('waiting on the PI'))

    let submit!: Promise<SaveOutcome>
    let skip!: Promise<SaveOutcome>
    act(() => {
      submit = feedbackApi!.submitFeedback()
      skip = feedbackApi!.skipFeedback()
    })
    await waitFor(() => expect(postGates).toHaveLength(1))
    expect(await skip).toEqual({ status: 'already_pending' })

    await settlePost()
    expect((await submit).status).toBe('saved')
    expect(postPayloads).toHaveLength(1)
    // The one request that went is the one that was asked for first, recorded
    // as what it is rather than inferred from the draft.
    expect(postPayloads[0]).toMatchObject({ outcome: 'matched_rank', correct_rank: 1 })
  })

  it('locks per assessment, so a pending save elsewhere blocks nothing here', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    // The reviewer moves on while the first save is still out.
    act(() => appApi!.setActiveQueryId(QUERY_B))
    await waitFor(() => expect(feedbackApi?.draftKey).toBe(draftKeyFor(QUERY_B)))
    // The lock belongs to the assessment that is saving, not to the buttons.
    expect(feedbackApi?.pendingSave).toBeNull()

    await chooseFirstCandidate()
    await waitFor(() => expect(submitButton().disabled).toBe(false))
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(2))
    expect(postPayloads[1]).toMatchObject({ query_id: QUERY_B })
  })
})

describe('newer work is never overwritten or hidden', () => {
  it('keeps text typed while the POST was out, and stays on the document', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'first thought')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    // The box is still live during a save, and what is typed into it has not
    // been sent.
    await userEvent.type(notesBox(), ' and a second')
    await settlePost()

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-superseded')).toBe('true')
    expect(notice.textContent).toContain('still unsent')
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))
    expect(notesBox().value).toBe('first thought and a second')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({
      notes: 'first thought and a second',
    })
    // What was sent is what was on screen when Submit was pressed.
    expect(postPayloads[0]).toMatchObject({ notes: 'first thought' })
  })

  it('holds the move when the reviewer types after the receipt', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()

    // Inside the pause before the move: a keystroke here is unsent work.
    act(() => feedbackApi!.setNotes('actually, hand B'))
    await letTheTimerFire()

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-navigation')).toBe('held')
    expect(notice.textContent).toContain('your newer draft is unsent')
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))
    expect(notesBox().value).toBe('actually, hand B')
  })

  it('counts an edit that ends where it started as newer work', async () => {
    // Comparing values would call this draft equal to the one that was sent
    // and navigate away from it; a revision counter cannot be talked round.
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'as sent')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()

    act(() => feedbackApi!.setNotes('as sent!'))
    act(() => feedbackApi!.setNotes('as sent'))
    await letTheTimerFire()

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-navigation')).toBe('held')
    expect(nextGates).toHaveLength(0)
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ notes: 'as sent' })
  })

  it('clears only the revision that was sent', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    act(() => feedbackApi!.setNotes('written while saving'))
    await settlePost()

    // The save succeeded and is not repeated, but the newer draft survives it.
    await waitFor(() =>
      expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({
        notes: 'written while saving',
      }),
    )
    expect(postPayloads).toHaveLength(1)
  })
})

describe('a late answer cannot steer a screen that has moved on', () => {
  it('does not navigate when the reviewer left and came back', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    act(() => appApi!.setActiveQueryId(QUERY_B))
    await waitFor(() => expect(feedbackApi?.draftKey).toBe(draftKeyFor(QUERY_B)))
    act(() => appApi!.setActiveQueryId(QUERY_A))
    await waitFor(() => expect(feedbackApi?.draftKey).toBe(draftKeyFor(QUERY_A)))

    await settlePost()
    await letTheTimerFire()

    // The key matches again, but this is a deliberate second visit: a save
    // made before the reviewer left must not throw them forward out of it.
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))
  })

  it('does not navigate when the lookup returns after the reviewer has moved', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })

    act(() => appApi!.setActiveQueryId(QUERY_C))
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_C)))
    await settleNext(0, { fileId: QUERY_B })

    await letTheTimerFire()
    expect(activeQueryId()).toBe(String(QUERY_C))
  })

  it('does not navigate or throw when the panel is gone', async () => {
    const view = renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    view.unmount()
    await settlePost()
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
  })

  it('does not navigate when the reviewer signs out mid-save', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await act(async () => {
      await reviewerApi!.clearReviewer()
    })
    await settlePost()
    await letTheTimerFire()

    // No signed-in reviewer means no assessment on screen, so there is nothing
    // for the finished save to acknowledge or advance past.
    expect(nextGates).toHaveLength(0)
    expect(screen.queryByTestId('assessment-saved-notice')).toBeNull()
  })

  it('does not put a saved draft back when /feedback/latest answers late', async () => {
    holdLatest = true
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'sent and done')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined())

    // The seeding request was issued before the save. Its answer is older than
    // the cleared draft, so applying it would present just-saved text as unsent.
    await waitFor(() => expect(latestGates.length).toBeGreaterThan(0))
    await act(async () => {
      latestGates[0].resolve(
        jsonResponse(
          receiptFor({ query_id: QUERY_A, model_slug: MODEL, outcome: 'correct' }, {
            notes: 'sent and done',
            correct_rank: 1,
          }),
        ),
      )
      await Promise.resolve()
    })

    await letTheTimerFire()
    expect(notesBox().value).toBe('')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined()
  })
})

describe('a failed move is not a failed save', () => {
  it('says the assessment is saved and retries only the lookup', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })
    await settleNext(0, { error: new TypeError('Failed to fetch') })

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-navigation')).toBe('failed')
    expect(notice.textContent).toContain('nothing here needs saving again')
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()

    await userEvent.click(screen.getByTestId('assessment-retry-advance'))
    await waitFor(() => expect(nextGates).toHaveLength(2))
    // The retry repeats the lookup and never the write: the log is append-only.
    expect(postPayloads).toHaveLength(1)

    await settleNext(1, { fileId: QUERY_B })
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_B)))
  })

  it('reports a failed lookup by its own status, not as a lost assessment', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })
    await settleNext(0, { status: 500 })

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.textContent).toContain('The queue is unavailable.')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined()
  })
})

describe('the acknowledgement toast promises only what it can do', () => {
  it('offers the draft back without claiming the record was withdrawn', async () => {
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'same incipit as 12')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()

    const toast = await screen.findByText('Assessment saved to the review log')
    expect(toast).toBeTruthy()
    expect(document.body.textContent).toContain(
      'The saved record stays in the log; this only puts your draft back.',
    )
    expect(screen.queryByRole('button', { name: 'Undo' })).toBeNull()

    await userEvent.click(screen.getByRole('button', { name: 'Put my draft back' }))
    await waitFor(() => expect(notesBox().value).toBe('same incipit as 12'))
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({
      correctRank: 1,
      notes: 'same incipit as 12',
    })
    // Putting the draft back is local: nothing was sent to withdraw the row.
    expect(postPayloads).toHaveLength(1)
  })

  it('does not offer to overwrite newer unsent work', async () => {
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'first thought')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    act(() => feedbackApi!.setNotes('a better reading'))
    await settlePost()

    await screen.findByText('Assessment saved to the review log')
    expect(document.body.textContent).toContain(
      'Your newer edits to this document are still unsent.',
    )
    expect(screen.queryByRole('button', { name: 'Put my draft back' })).toBeNull()
    expect(notesBox().value).toBe('a better reading')
  })

  it('names a skip as a skip', async () => {
    renderPanel({ sidebar: true })
    await screen.findByTestId('match-pill-1')
    await userEvent.type(notesBox(), 'illegible, ask the PI')
    await userEvent.click(skipButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    expect(postPayloads[0]).toMatchObject({ outcome: 'skipped' })
    await settlePost()

    await screen.findByText('Skip saved to the review log')
  })
})

/**
 * The repairs asked for by the independent review of this branch.
 *
 * Three of them are about WHOSE answer it is. A save, a next-document lookup
 * and a prefill all finish some time after they were asked for, by which point
 * the reviewer may be on another document, in another view, or signed in as
 * someone else. Each one therefore carries the assessment and the visit it
 * started on, and may only change the screen it belongs to.
 *
 * The other three are about WHAT an answer proves: a receipt is proof only if
 * it is the row that was asked for, a request in flight belongs to the
 * assessment rather than to a set of buttons that unmount with the sidebar,
 * and a notice about unsent work has to be read from the box as it is now.
 */
describe('an answer changes only the assessment that asked for it', () => {
  it('refuses a receipt whose recorded directory is not the one on screen', async () => {
    // The directory is the assignment. A row against another directory is a
    // different decision from the one the reviewer made, however well formed.
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      body: receiptFor(postPayloads[0], { correct_dir: 'somewhere-else' }),
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
  })

  it('refuses a receipt that drops a choice the reviewer made', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      body: receiptFor(postPayloads[0], { selected_ranks: null }),
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
  })

  it('refuses a receipt recorded against another reviewer', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      body: receiptFor(postPayloads[0], { reviewer_account_id: 404 }),
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
  })

  it('refuses a receipt whose note is not the note that was sent', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'the hand changes here')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await settlePost(0, {
      body: receiptFor(postPayloads[0], { notes: 'the hand changes' }),
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(notesBox().value).toBe('the hand changes here')
  })

  it('keeps the returned-to draft when the save lands after a round trip', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'read again on return')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    act(() => appApi!.setActiveQueryId(QUERY_B))
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_B)))
    act(() => appApi!.setActiveQueryId(QUERY_A))
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_A)))
    await settlePost()

    // This is a second visit to the same assessment. The old save may record
    // its row, but it may not empty a box the reviewer has come back to.
    expect(notesBox().value).toBe('read again on return')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({
      notes: 'read again on return',
    })
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
  })

  it('does not let a finished save on another document steer this one', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    act(() => appApi!.setActiveQueryId(QUERY_B))
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_B)))
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(2))

    // B is acknowledged and counting down to its move; A then finishes late.
    await settlePost(1)
    await settlePost(0)
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })

    // One lookup, and it is B's: A's completion neither cancelled it nor added
    // a second one.
    await settleNext(0, { fileId: QUERY_C })
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_C)))
  })

  it('does not let a finished save on another document replace this recovery', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    act(() => appApi!.setActiveQueryId(QUERY_B))
    await waitFor(() => expect(activeQueryId()).toBe(String(QUERY_B)))
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(2))
    await settlePost(1)
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })
    await settleNext(0, { status: 500 })
    await screen.findByTestId('assessment-saved-notice')

    // A's late success must not take away the retry that belongs to B.
    await settlePost(0)
    await letTheTimerFire()
    const notice = screen.getByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-navigation')).toBe('failed')
    await userEvent.click(screen.getByTestId('assessment-retry-advance'))
    await waitFor(() => expect(nextGates).toHaveLength(2))
    expect(postPayloads).toHaveLength(2)
  })

  it('keeps a save in flight across a remount of the controls', async () => {
    // Collapsing the sidebar unmounts the buttons. Their own state cannot be
    // the record of a request that is still out, or the reviewer is invited to
    // send the same assessment a second time.
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await userEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }))
    await userEvent.click(screen.getByRole('button', { name: 'Expand sidebar' }))

    await waitFor(() => expect(submitButton().disabled).toBe(true))
    expect(document.body.textContent).toContain('Saving your assessment')
    await userEvent.click(submitButton())
    expect(postPayloads).toHaveLength(1)
  })

  it('still offers a colleague note on the next document after a save', async () => {
    // The seeding guard used to be one counter for the whole panel, so any
    // save silenced the prefill everywhere. It is now per assessment.
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined())

    latestEntry = receiptFor(
      {
        query_id: QUERY_B,
        model_slug: MODEL,
        outcome: 'matched_rank',
        notes: 'a colleague read this',
      },
      { reviewer: 'Alice Archivista', reviewer_account_id: 3, reviewer_username: 'alice' },
    )
    act(() => appApi!.setActiveQueryId(QUERY_B))

    await waitFor(() => expect(notesBox().value).toBe('a colleague read this'))
  })

  it('does not prefill over an edit that was typed and undone', async () => {
    holdLatest = true
    renderPanel()
    await screen.findByTestId('match-pill-1')
    await waitFor(() => expect(latestGates.length).toBeGreaterThan(0))
    await userEvent.type(notesBox(), 'x')
    await userEvent.clear(notesBox())

    await act(async () => {
      latestGates[0].resolve(
        jsonResponse(
          receiptFor(
            {
              query_id: QUERY_A,
              model_slug: MODEL,
              outcome: 'matched_rank',
              notes: 'older reading',
            },
            {
              reviewer: 'Alice Archivista',
              reviewer_account_id: 3,
              reviewer_username: 'alice',
            },
          ),
        ),
      )
      await Promise.resolve()
    })

    // An empty box the reviewer emptied is a decision, not an absence: the
    // attribution still appears, but the older note is not poured back in.
    expect(notesBox().value).toBe('')
    expect((await screen.findByTestId('note-attribution')).textContent).toContain('alice')
  })

  it('says newer work is unsent when the lookup fails after more typing', async () => {
    renderPanel()
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))
    await settlePost()
    await waitFor(() => expect(nextGates).toHaveLength(1), { timeout: 2000 })
    act(() => feedbackApi!.setNotes('written while the lookup was out'))
    await settleNext(0, { status: 503 })

    const notice = await screen.findByTestId('assessment-saved-notice')
    expect(notice.getAttribute('data-navigation')).toBe('failed')
    expect(notice.textContent).not.toContain('nothing here needs saving again')
    expect(notice.textContent).toContain('unsent')
    expect(notesBox().value).toBe('written while the lookup was out')
  })
})

/**
 * Collapsing the sidebar unmounts the panel and the buttons, so a save can
 * settle with nothing on screen that asked for it. The answer still belongs to
 * the assessment, and the reviewer still has to be told what it was.
 */
describe('an outcome outlives the controls that asked for it', () => {
  async function remountControls(): Promise<void> {
    await userEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }))
    await userEvent.click(screen.getByRole('button', { name: 'Expand sidebar' }))
  }

  /** Put the panel away, and leave it away. */
  async function collapseSidebar(): Promise<void> {
    await userEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }))
  }

  async function expandSidebar(): Promise<void> {
    await userEvent.click(screen.getByRole('button', { name: 'Expand sidebar' }))
  }

  it('shows a failure that settled while the sidebar was still collapsed', async () => {
    // The other ordering: the answer arrives with nothing on screen at all, so
    // there is no component whose own state could have carried it.
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'hand A throughout')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await collapseSidebar()
    await settlePost(0, { error: new TypeError('Failed to fetch') })
    await expandSidebar()

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    const status = await screen.findByTestId('submit-button-status')
    expect(status.parentElement?.getAttribute('data-op-state')).toBe('failed')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
    expect(notesBox().value).toBe('hand A throughout')
  })

  it('reads nothing, and acknowledges, for a success that settled with the sidebar shut', async () => {
    // The reviewer put the panel away and never came back before the answer
    // arrived, so no second visit exists for this to steer or clobber: the
    // submitted revision is consumed exactly as it is with the sidebar open.
    // What must NOT happen is the move -- a read belongs to the assessment
    // visit that asked for it, and no panel did (issue #171).
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await collapseSidebar()
    await settlePost(0)
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
    await expandSidebar()

    expect(activeQueryId()).toBe(String(QUERY_A))
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toBeUndefined()
    // Not silent about it either: the acknowledgement is on screen, with the
    // draft still recoverable by hand.
    expect(document.body.textContent).toContain('Assessment saved to the review log')
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: 'Put my draft back' }))
    expect(feedbackApi?.draft.correctRank).toBe(1)
  })

  it('shows a network failure that landed while the controls were gone', async () => {
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.type(notesBox(), 'hand A throughout')
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await remountControls()
    await settlePost(0, { error: new TypeError('Failed to fetch') })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    const status = await screen.findByTestId('submit-button-status')
    expect(status.parentElement?.getAttribute('data-op-state')).toBe('failed')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
    expect(notesBox().value).toBe('hand A throughout')

    // The retry is a real second request, made knowingly.
    await waitFor(() => expect(submitButton().disabled).toBe(false))
    await userEvent.click(submitButton())
    await waitFor(() => expect(postPayloads).toHaveLength(2))
  })

  it('shows a rejected save that landed while the controls were gone', async () => {
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await remountControls()
    await settlePost(0, {
      status: 409,
      body: {
        error: {
          code: 'CANDIDATE_IDENTITY_CHANGED',
          message: 'Rank 1 now resolves to a different directory.',
        },
      },
    })

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('rejected')
    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
    await letTheTimerFire()
    expect(nextGates).toHaveLength(0)
  })

  it('does not clear or carry away the screen the reviewer came back to', async () => {
    // The receipt is real and the row is committed, but the panel on screen is
    // a new visit: it cannot lose its draft or be steered off the document by
    // an answer to a question it did not ask.
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await remountControls()
    await settlePost(0)
    await letTheTimerFire()

    expect(storedDrafts()[draftKeyFor(QUERY_A)]).toMatchObject({ correctRank: 1 })
    expect(nextGates).toHaveLength(0)
    expect(activeQueryId()).toBe(String(QUERY_A))
    expect(screen.queryByTestId('assessment-saved-notice')).toBeNull()
    expect(document.body.textContent).not.toContain('Assessment saved to the review log')

    // What did happen is still reported, by the panel that is now on screen.
    const notice = await screen.findByTestId('assessment-prior-save')
    expect(notice.getAttribute('data-superseded')).toBe('false')
    expect(notice.textContent).toContain('recorded after you left this screen')
  })

  it('calls work typed after the reviewer came back unsent', async () => {
    renderPanel({ sidebar: true })
    await chooseFirstCandidate()
    await userEvent.click(submitButton())
    await waitFor(() => expect(postGates).toHaveLength(1))

    await remountControls()
    await settlePost(0)
    await userEvent.type(notesBox(), 'read again on return')

    const notice = await screen.findByTestId('assessment-prior-save')
    expect(notice.getAttribute('data-superseded')).toBe('true')
    expect(screen.getByText('Your newer edits to this document are still unsent.')).toBeTruthy()
    expect(notesBox().value).toBe('read again on return')
  })
})
