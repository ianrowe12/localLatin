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

/** An unranked reviewer directory, as the API serves one since issue #196. */
function reviewerCard(score: number) {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    ccl_key: '',
    score,
    dir_files: ['query-3.txt'],
    preview_text: 'seed text',
    candidate_files: [{ filename: 'query-3.txt', text: 'seed text' }],
    created_by: 'Abigail',
    seed_query_id: 3,
    member_query_ids: [3],
  }
}

/** What the /predictions route should answer, per model, for the next request. */
type Answer = {
  status?: string | null
  predictions?: unknown[]
  seeded_dirs?: unknown[]
  reviewer_dir_candidates?: unknown[]
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
/** Directories the database already holds for the document under test. */
let seedLookup: unknown[] = []
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
      // #161). `seedLookup` is what the database already holds for this
      // document; a successful create adds the row the POST returned.
      if (url.includes('/api/reviewer_dirs')) {
        return jsonResponse(
          reviewerDirPosts.length > 0 && reviewerDirStatus === 201
            ? [createdDirFixture()]
            : seedLookup,
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
          reviewer_dir_candidates: answer.reviewer_dir_candidates ?? [],
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

/**
 * The list on its own, for the cases the other two views play no part in.
 *
 * `showReviewerDirectories` is off on the reviewer-facing page since issue
 * #221; a case that is about those cards asks for them explicitly.
 */
function renderList(options: { showReviewerDirectories?: boolean } = {}) {
  return render(
    <AppProvider>
      <SavedDirectoryProvider accountKey="test-account">
        <PredictionProvider>
          <SelectQuery />
          <ModelSelector />
          <PredictionList
            showReviewerDirectories={options.showReviewerDirectories}
          />
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
  seedLookup = []
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
    expect(await screen.findByText('None of the top 3')).toBeTruthy()
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
    answers[MODEL_A] = {
      status: 'ok',
      predictions: [],
      reviewer_dir_candidates: [reviewerCard(0.6)],
    }
    // Drawn on purpose here: the question is whether the shared state keeps the
    // two apart, so the block that would blur them has to be on screen.
    renderList({ showReviewerDirectories: true })

    expect(await screen.findByTestId('predictions-empty')).toBeTruthy()
    // The card is offered, under its own heading and with the key that files a
    // document into it, and the model's answer is still reported as empty.
    expect(screen.getByTestId('reviewer-dir-card-reviewer-dir-1')).toBeTruthy()
    expect(screen.getByTestId('reviewer-dirs-heading')).toBeTruthy()
    expect(screen.queryByTestId('band-chip-1')).toBeNull()
  })
})

describe('what is already recorded survives the refetch (issue #161, #196)', () => {
  it('keeps a seeded directory on screen when the refetch fails', async () => {
    // A directory this document seeds is permanent, and nothing in this panel
    // can create or remove one any more. It must not vanish because a request
    // failed: the reviewer would be left unable to tell that it exists.
    const seeded = {
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
    seedLookup = [seeded]
    answers[MODEL_A] = { predictions: [modelCard(1, 0.31)], seeded_dirs: [seeded] }
    answers[MODEL_B] = { httpStatus: 503 }
    renderList()

    const notice = await screen.findByTestId('new-directory-saved')
    expect(notice.textContent).toContain('Unattested homily')

    // A request that fails replaces the whole ranked subtree. The record of a
    // permanent grouping is not part of that subtree's evidence.
    await selectModel(MODEL_B)

    expect(await screen.findByTestId('predictions-error')).toBeTruthy()
    expect(screen.getByTestId('new-directory-saved').textContent).toContain(
      'Unattested homily',
    )
  })

  it('offers no way to create a directory while the ranking loads', async () => {
    answers[MODEL_A] = { predictions: [modelCard(1, 0.9)], delayMs: 30 }
    renderList()

    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(await screen.findByTestId('band-chip-1')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(reviewerDirPosts).toHaveLength(0)
  })
})
