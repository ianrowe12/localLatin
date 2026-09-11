import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from './AppContext'
import { FeedbackProvider } from './FeedbackContext'
import { PredictionProvider } from './PredictionContext'
import { ReviewerProvider } from './ReviewerContext'
import { SavedDirectoryProvider } from './SavedDirectoryContext'
import { TokenProvider } from './TokenContext'
import FeedbackPanel from '../components/feedback/FeedbackPanel'
import CenterArea from '../components/layout/CenterArea'
import ModelSelector from '../components/predictions/ModelSelector'
import PredictionList from '../components/predictions/PredictionList'

/**
 * One authoritative prediction state (issue #156).
 *
 * These tests drive the real components under the real provider tree from
 * App.tsx. The bug they pin down was never visible in a single component: the
 * ranked list, the evidence view and the assessment panel each ran their own
 * copy of `usePredictions`, so they could hold three different answers to "what
 * is the ranking for this document" at the same time -- and a failed request
 * rendered as the same grey sentence as an empty one.
 */

const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'google_mt5-base'
const QUERY_ID = 7

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const sleep = (ms: number) =>
  ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve()

function modelEntry(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'sif',
    prediction_count: 2238,
    available_variants: ['sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.5, verify: 0.7 },
    supports_reviewer_dirs: true,
  }
}

function modelCard(rank: number, score: number, dir?: string) {
  return {
    rank,
    dir_name: dir ?? `candidate-${rank}`,
    score,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'candidate text' }],
    source: 'model',
  }
}

function reviewerCard(rank: number, score: number) {
  return {
    rank,
    dir_name: 'reviewer-dir-1',
    score,
    dir_files: ['query-3.txt'],
    preview_text: 'seed text',
    candidate_files: [{ filename: 'query-3.txt', text: 'seed text' }],
    source: 'reviewer',
    label: 'Unattested homily',
    created_by: 'Abigail',
    seed_query_id: 3,
  }
}

/** What the /predictions route should answer, per model, for the next request. */
type Answer = {
  status?: string | null
  predictions?: unknown[]
  seeded_dirs?: unknown[]
  httpStatus?: number
  delayMs?: number
  /**
   * Hold the response open until the test releases it. A millisecond delay
   * races `userEvent`, which awaits its own microtasks and can outlast it on a
   * loaded machine; a gate makes "the ranking is not available yet" a fact the
   * test controls rather than a window it hopes to hit.
   */
  hold?: boolean
  /** Overrides the identity fields, to forge a response for another request. */
  identity?: Record<string, unknown>
}

let answers: Record<string, Answer> = {}
/** Answers consumed in order for repeat requests; falls back to `answers`. */
let queued: Record<string, Answer[]> = {}
let predictionRequests: string[] = []
let reviewerDirPosts: unknown[] = []
let reviewerDirStatus = 201
/** Resolves the currently held predictions response, if there is one. */
let releaseHeld: (() => void) | null = null

function answerFor(model: string): Answer {
  const next = queued[model]?.shift()
  return next ?? answers[model] ?? { predictions: [modelCard(1, 0.9)] }
}

/** The one row this suite's backend stores for QUERY_ID's seed. */
function createdDirFixture() {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    status: 'awaiting_match',
    seed_query_id: QUERY_ID,
    member_query_ids: [QUERY_ID],
    created_at: '2026-08-26 00:00:00',
    created_by: 'Abigail',
    model_slug: MODEL_A,
    variant: 'sif_abtt',
    best_match_score: 0.31,
    has_potential_match: false,
  }
}

function installFetch(): void {
  predictionRequests = []
  reviewerDirPosts = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const params = new URL(url, 'http://localhost').searchParams

      if (init?.method === 'POST' && url.includes('/api/reviewer_dirs')) {
        reviewerDirPosts.push(JSON.parse(String(init.body)))
        if (reviewerDirStatus !== 201) {
          return jsonResponse({ detail: 'Authentication required' }, reviewerDirStatus)
        }
        return jsonResponse(createdDirFixture(), 201)
      }

      // The seed-filtered lookup behind the durable acknowledgement (issue
      // #161). Before a successful create the seed has no directory; after one
      // the same row the POST returned is what the database holds.
      if (url.includes('/api/reviewer_dirs')) {
        return jsonResponse(
          reviewerDirPosts.length > 0 && reviewerDirStatus === 201
            ? [createdDirFixture()]
            : [],
        )
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
          modelEntry(MODEL_A, 'LaTa (T5)'),
          modelEntry(MODEL_B, 'mT5-base'),
        ])
      }
      if (url.includes('/api/feedback/latest')) return jsonResponse(null)
      if (url.includes('/token_map')) {
        return jsonResponse({ error: { message: 'no artifact' } }, 404)
      }
      if (url.includes('/predictions')) {
        const model = params.get('model') ?? ''
        predictionRequests.push(url)
        const answer = answerFor(model)
        if (answer.hold === true) {
          await new Promise<void>((resolve) => {
            releaseHeld = resolve
          })
        }
        await sleep(answer.delayMs ?? 0)
        if (answer.httpStatus && answer.httpStatus !== 200) {
          return jsonResponse(
            { error: { code: 'internal_error', message: 'Ranking store unavailable' } },
            answer.httpStatus,
          )
        }
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          model,
          variant: 'sif_abtt',
          status: answer.status === undefined ? 'ok' : answer.status,
          predictions: answer.predictions ?? [],
          seeded_dirs: answer.seeded_dirs ?? [],
          ...(answer.identity ?? {}),
        })
      }
      if (/\/api\/query\/\d+$/.test(new URL(url, 'http://localhost').pathname)) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          text: 'uerba domini',
          tokens: [
            { text: 'uerba', index: 0, category: 'content' },
            { text: 'domini', index: 1, category: 'content' },
          ],
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
    setActiveModel(MODEL_A)
  }, [setActiveQueryId, setActiveModel])
  return null
}

/**
 * The three consumers, in App.tsx's own provider order. Mounting all three is
 * the point: the regression this guards against is them disagreeing.
 */
function renderReview() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <SavedDirectoryProvider accountKey="test-account">
            <PredictionProvider>
              <FeedbackProvider>
                <SelectQuery />
                <PredictionList />
                <CenterArea />
                <FeedbackPanel />
              </FeedbackProvider>
            </PredictionProvider>
          </SavedDirectoryProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

/** The list on its own, for the cases the other two views play no part in. */
function renderList() {
  return render(
    <AppProvider>
      <SavedDirectoryProvider accountKey="test-account">
        <PredictionProvider>
          <SelectQuery />
          <ModelSelector />
          <PredictionList />
        </PredictionProvider>
      </SavedDirectoryProvider>
    </AppProvider>,
  )
}

async function selectModel(slug: string) {
  await userEvent.selectOptions(screen.getByLabelText('Select model'), slug)
}

beforeEach(() => {
  answers = {}
  queued = {}
  reviewerDirStatus = 201
  releaseHeld = null
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('one shared ranking', () => {
  it('answers the list, the evidence view and the assessment panel from one request', async () => {
    // Three consumers, one request. Three copies of the hook meant three
    // requests and three independent answers, which is how the list could show
    // a failure while the panel offered ranks from a different response.
    answers[MODEL_A] = {
      predictions: [modelCard(1, 0.9), modelCard(2, 0.85), modelCard(3, 0.8)],
    }
    renderReview()

    await screen.findByTestId('band-chip-1')
    // The panel's pill count comes from the same candidates the list drew.
    expect(await screen.findByText('None of the 3 model candidates')).toBeTruthy()
    expect(predictionRequests).toHaveLength(1)
  })

  it('never hands a consumer the previous model’s ranking under the new model’s name', async () => {
    // `activeModel` changes during render; an effect-only guard clears the old
    // data one commit later, and that commit is a real paint.
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9, 'LATA.DIR')] }
    // Held rather than delayed: the assertion below is about the state between
    // the selection and the new answer, and a timed response can settle inside
    // `userEvent`'s own awaits, leaving the test to assert the spinner against
    // a ranking that has already arrived.
    answers[MODEL_B] = { predictions: [modelCard(1, 0.9, 'MT5.DIR')], hold: true }
    renderList()

    expect(await screen.findByTitle('LATA.DIR')).toBeTruthy()
    await selectModel(MODEL_B)

    // The instant the model changes there is no current ranking, so the old
    // directory is gone rather than relabelled. mT5 cannot have answered: its
    // response is still held.
    expect(screen.queryByTitle('LATA.DIR')).toBeNull()
    expect(screen.getByTestId('predictions-loading')).toBeTruthy()

    await act(async () => {
      releaseHeld?.()
      await Promise.resolve()
    })
    expect(await screen.findByTitle('MT5.DIR')).toBeTruthy()
    expect(screen.queryByTitle('LATA.DIR')).toBeNull()
  })

  it('drops a superseded response instead of letting it overwrite the current one', async () => {
    // The reviewer moves on while the first request is still in flight. Its
    // answer is about a selection nobody is looking at any more.
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9, 'LATA.DIR')], delayMs: 60 }
    answers[MODEL_B] = { predictions: [modelCard(1, 0.9, 'MT5.DIR')] }
    renderList()

    await screen.findByTestId('predictions-loading')
    await selectModel(MODEL_B)
    expect(await screen.findByTitle('MT5.DIR')).toBeTruthy()

    // Long enough for the superseded LaTa response to land.
    await new Promise((resolve) => setTimeout(resolve, 120))
    expect(screen.queryByTitle('LATA.DIR')).toBeNull()
    expect(screen.getByTitle('MT5.DIR')).toBeTruthy()
  })
})

describe('failure is reported as failure', () => {
  it('says the ranking did not load, and offers to try again', async () => {
    // The regression: a 500 rendered as "No predictions available", which reads
    // as "the model found nothing" -- a verdict on the document.
    answers[MODEL_A] = { httpStatus: 500 }
    renderList()

    const alert = await screen.findByTestId('predictions-error')
    expect(alert.getAttribute('role')).toBe('alert')
    expect(alert.textContent).toContain('Ranking store unavailable')
    expect(screen.queryByTestId('predictions-empty')).toBeNull()
    expect(screen.getByTestId('predictions-retry')).toBeTruthy()
  })

  it('refetches on retry and clears the failure when the server recovers', async () => {
    queued[MODEL_A] = [{ httpStatus: 500 }]
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9, 'RECOVERED.DIR')] }
    renderList()

    await screen.findByTestId('predictions-error')
    await userEvent.click(screen.getByTestId('predictions-retry'))

    expect(await screen.findByTitle('RECOVERED.DIR')).toBeTruthy()
    expect(screen.queryByTestId('predictions-error')).toBeNull()
    expect(predictionRequests).toHaveLength(2)
  })

  it('tells the evidence view too, rather than leaving a blank pane', async () => {
    // A blank candidate panel beside a document is itself a claim. It should
    // not be made when the truth is that the request failed.
    answers[MODEL_A] = { httpStatus: 500 }
    renderReview()

    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('did not load')
  })

  it('survives a malformed seeded directory instead of blanking the app', async () => {
    // AwaitingMatchBadge reads .status on every seeded directory during render,
    // so an unchecked entry threw inside React and took the whole review view
    // down -- no list, no evidence, no explanation. A rejected payload is a
    // reported failure instead.
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [null] }
    renderReview()

    const alert = await screen.findByTestId('predictions-error')
    expect(alert.textContent).toContain('not with a ranking this app can read')
    // The document panel is still mounted; only the ranking is missing.
    expect(screen.getByTestId('candidate-evidence-note').textContent).toContain(
      'did not load',
    )
  })

  it('treats a 200 for the wrong document as a failure, not as a ranking', async () => {    // A validated identity mismatch is the difference between "no candidates"
    // and "candidates for somebody else's manuscript".
    answers[MODEL_A] = {
      predictions: [modelCard(1, 0.9)],
      identity: { file_id: 999 },
    }
    renderList()

    const alert = await screen.findByTestId('predictions-error')
    expect(alert.textContent).toContain('not with a ranking this app can read')
    expect(screen.queryByTestId('band-chip-1')).toBeNull()
  })
})

describe('excluded, empty and unknown are three different things', () => {
  it('explains a blank source document', async () => {
    answers[MODEL_A] = { status: 'excluded_blank_source', predictions: [] }
    renderList()

    const notice = await screen.findByTestId('predictions-excluded')
    expect(notice.textContent).toContain('source text was blank')
    expect(notice.textContent).toContain('nothing here to assess')
    expect(screen.queryByTestId('predictions-empty')).toBeNull()
  })

  it('explains a zero-norm embedding differently', async () => {
    // Not the same failure and not the same remedy: the text is there, the
    // model just produced nothing usable from it.
    answers[MODEL_A] = { status: 'excluded_zero_norm', predictions: [] }
    renderList()

    const notice = await screen.findByTestId('predictions-excluded')
    expect(notice.textContent).toContain('zero norm')
    expect(notice.textContent).not.toContain('source text was blank')
  })

  it('keeps an unrecognised exclusion an exclusion, and names it', async () => {
    answers[MODEL_A] = { status: 'excluded_future_reason', predictions: [] }
    renderList()

    const notice = await screen.findByTestId('predictions-excluded')
    expect(notice.textContent).toContain('excluded_future_reason')
  })

  it('admits it does not know why an ok row came back with nothing', async () => {
    answers[MODEL_A] = { status: 'ok', predictions: [] }
    renderList()

    const notice = await screen.findByTestId('predictions-empty')
    expect(notice.textContent).toContain('no reason recorded')
    expect(screen.queryByTestId('predictions-excluded')).toBeNull()
  })

  it('does not guess "excluded" from a deployment that sends no status', async () => {
    // Legacy CSVs have no status column at all. Unknown is unknown.
    answers[MODEL_A] = { status: null, predictions: [] }
    renderList()

    expect(await screen.findByTestId('predictions-empty')).toBeTruthy()
    expect(screen.queryByTestId('predictions-excluded')).toBeNull()
  })

  it('does not let a reviewer directory stand in for a model ranking', async () => {
    // A colleague's new directory scoring 0.6 is not the model saying anything
    // about this document. The card is still offered; the model's answer is
    // still reported as empty.
    answers[MODEL_A] = { status: 'ok', predictions: [reviewerCard(11, 0.6)] }
    renderList()

    expect(await screen.findByTestId('predictions-empty')).toBeTruthy()
    expect(screen.getByText('Unattested homily')).toBeTruthy()
    expect(screen.queryByTestId('band-chip-1')).toBeNull()
  })
})

describe('a created directory survives the refetch it caused', () => {
  it('keeps the acknowledgement even when the refetch fails', async () => {
    // Creating a directory is permanent and unrepeatable, and it broadcasts a
    // refresh. If the acknowledgement dies with the old ranking, the reviewer
    // has no evidence the creation happened and may do it again.
    answers[MODEL_A] = { predictions: [modelCard(1, 0.31)] }
    queued[MODEL_A] = [{ predictions: [modelCard(1, 0.31)] }, { httpStatus: 503 }]
    renderList()

    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    await userEvent.click(screen.getByTestId('new-directory-submit'))

    await waitFor(() => {
      expect(reviewerDirPosts).toHaveLength(1)
    })
    // The refetch its own creation triggered failed...
    expect(await screen.findByTestId('predictions-error')).toBeTruthy()
    // ...and the confirmation is still on screen.
    const created = await screen.findByTestId('new-directory-created')
    expect(created.textContent).toContain('Unattested homily')
  })

  it('does not offer the new-directory CTA before the ranking has settled', async () => {
    // "Start a new directory" is a judgement about the ranking. Offering it
    // while the ranking is still loading invites it on no evidence at all.
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9)], delayMs: 30 }
    renderList()

    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
  })
})
