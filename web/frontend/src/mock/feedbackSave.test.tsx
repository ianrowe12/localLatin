import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../contexts/AppContext'
import { FeedbackProvider, useFeedback, type FeedbackContextValue } from '../contexts/FeedbackContext'
import { PredictionProvider } from '../contexts/PredictionContext'
import { ReviewerProvider, useReviewer } from '../contexts/ReviewerContext'
import { TokenProvider } from '../contexts/TokenContext'
import FeedbackPanel from '../components/feedback/FeedbackPanel'
import { installMockHandler } from './handler'

/**
 * `npm run dev:mock` has to be able to save (issue #158).
 *
 * The client now refuses anything that is not a full feedback row, because a
 * bare `{ success: true }` is not proof that a write happened. The mock used
 * to answer exactly that, so every Save and Skip in mock mode failed. This
 * drives the real panel against the real mock handler, which is the only way
 * to catch the two drifting apart again.
 */

const MODEL = 'bowphs_LaTa'
const QUERY = 5
const MATCH_RANK = 1
/** The mock signs `reviewer` in as account 2; `pi` is 1. */
const REVIEWER_ACCOUNT = 2

let feedbackApi: FeedbackContextValue | null = null
let saveRequests = 0
let settleOrder: string[] = []
const realFetch = window.fetch

function api(): FeedbackContextValue {
  if (feedbackApi === null) throw new Error('the panel never mounted')
  return feedbackApi
}

function requestPath(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input
  return input instanceof URL ? input.href : input.url
}

/**
 * Pin the mock's own 200-400 ms jitter to its two ends: every ordinary request
 * at 200 ms, authentication at 399.8 ms.
 *
 * The handler reads `Math.random()` synchronously, before its first `await`,
 * so a spy around the call fixes that one request's delay and nothing else.
 * Both endpoints are inside the range the mock already produces, so this makes
 * one of its real schedules reproducible rather than inventing a new one.
 *
 * The schedule chosen is the adverse one: the ranking, and therefore the match
 * pills, arrive roughly 200 ms before the account does. That is what the setup
 * below has to survive, and it is why these tests wait on the account instead
 * of on the pill.
 */
function authSettlesAfterPredictions(): void {
  const mockFetch = window.fetch
  window.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const path = requestPath(input)
    if (path.includes('/api/feedback') && init?.method === 'POST') saveRequests += 1
    const jitter = vi
      .spyOn(Math, 'random')
      .mockReturnValue(path.includes('/api/auth/') ? 0.999 : 0)
    try {
      const response = mockFetch(input, init)
      void response.then(() => settleOrder.push(path)).catch(() => settleOrder.push(path))
      return response
    } finally {
      jitter.mockRestore()
    }
  }
}

function firstSettleIndex(fragment: string): number {
  return settleOrder.findIndex((path) => path.includes(fragment))
}

/**
 * Prove the schedule above is the adverse one, from the order requests
 * actually finished rather than from a clock reading: the ranking lands first
 * and sign-in lands 199.8 ms later. A wait on the rendered pill therefore
 * cannot be a wait on the account.
 */
function expectPredictionsSettledBeforeSignIn(): void {
  const predictions = firstSettleIndex('/predictions')
  const signIn = firstSettleIndex('/api/auth/signin')
  expect(predictions).toBeGreaterThanOrEqual(0)
  expect(signIn).toBeGreaterThanOrEqual(0)
  expect(predictions).toBeLessThan(signIn)
}

function Harness() {
  const app = useApp()
  const reviewer = useReviewer()
  feedbackApi = useFeedback()
  const { setActiveQueryId, setActiveModel } = app
  const { signIn } = reviewer
  useEffect(() => {
    void signIn({ username: 'reviewer', password: 'mock-only' })
    setActiveQueryId(QUERY)
    setActiveModel(MODEL)
  }, [setActiveModel, setActiveQueryId, signIn])
  return null
}

function renderPanel() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <PredictionProvider>
            <FeedbackProvider>
              <Harness />
              <FeedbackPanel />
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

function enabledButton(name: RegExp): HTMLButtonElement {
  const button = screen.getByRole('button', { name }) as HTMLButtonElement
  expect(button.disabled).toBe(false)
  return button
}

/**
 * Wait for an assessment this account owns and can actually answer.
 *
 * `match-pill-1` exists as soon as the ranking lands, which under the schedule
 * above is well before sign-in resolves. At that moment `accountId` is null, so
 * there is no draft key to write into and `canEvaluate` is false: the pill and
 * Skip are disabled, the context refuses the choice, the typed note goes
 * nowhere and Submit sends nothing. A later `pendingSave === null` would then
 * be true only because no request was ever made. So the readiness these tests
 * wait for is the account, its draft key, and current evidence that supports
 * the answer they are about to give.
 */
async function awaitOwnedAssessment(): Promise<void> {
  await screen.findByTestId(`match-pill-${MATCH_RANK}`, {}, { timeout: 5000 })
  await waitFor(
    () => {
      expect(api().accountId).toBe(REVIEWER_ACCOUNT)
      expect(api().draftKey).toMatch(new RegExp(`^acct${REVIEWER_ACCOUNT}:`))
      expect(api().evidence.phase).toBe('ready')
      expect(api().evidence.canEvaluate).toBe(true)
      expect(api().evidence.candidateAt(MATCH_RANK)?.usable).toBe(true)
    },
    { timeout: 5000 },
  )
  expectPredictionsSettledBeforeSignIn()
}

beforeEach(async () => {
  localStorage.clear()
  feedbackApi = null
  saveRequests = 0
  settleOrder = []
  installMockHandler()
  // The mock's signed-in user is module state that outlives a test, so clear
  // it: every case here starts unauthenticated and has to wait for its own
  // sign-in, exactly as `npm run dev:mock` does on a cold load.
  await window.fetch('/api/auth/signout', { method: 'POST' })
  authSettlesAfterPredictions()
})

afterEach(() => {
  window.fetch = realFetch
  vi.restoreAllMocks()
})

describe('the dev mock can still save', () => {
  it('acknowledges a match saved through the mock handler', async () => {
    renderPanel()
    await awaitOwnedAssessment()

    const pill = screen.getByTestId(`match-pill-${MATCH_RANK}`) as HTMLButtonElement
    expect(pill.disabled).toBe(false)
    await userEvent.click(pill)
    await userEvent.type(notesBox(), 'saved under dev:mock')
    // The answer has to be in this account's own draft before Submit means
    // anything; a refused click leaves both of these unset.
    expect(api().draft.correctRank).toBe(MATCH_RANK)
    expect(api().draft.notes).toBe('saved under dev:mock')

    await userEvent.click(enabledButton(/Submit/))

    // Waiting for a null `pendingSave` proves nothing: it is also null before
    // any POST. The receipt is the thing that says the mock answered.
    await waitFor(() => expect(api().lastSubmission?.kind).toBe('submit'), { timeout: 5000 })
    expect(saveRequests).toBe(1)
    expect(api().pendingSave).toBeNull()
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(api().lastSubmission?.entry.outcome).toBe('matched_rank')
    expect(api().lastSubmission?.entry.notes).toBe('saved under dev:mock')
  })

  it('acknowledges a note-required Skip through the mock handler', async () => {
    renderPanel()
    await awaitOwnedAssessment()

    await userEvent.type(notesBox(), 'illegible opening')
    expect(api().draft.notes).toBe('illegible opening')

    await userEvent.click(enabledButton(/Skip/))

    await waitFor(() => expect(api().lastSubmission?.kind).toBe('skip'), { timeout: 5000 })
    expect(saveRequests).toBe(1)
    expect(api().pendingSave).toBeNull()
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(api().lastSubmission?.entry.outcome).toBe('skipped')
    expect(api().draft.notes).toBe('')
  })
})
