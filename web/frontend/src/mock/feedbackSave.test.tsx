import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
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

let feedbackApi: FeedbackContextValue | null = null
const realFetch = window.fetch

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

beforeEach(() => {
  localStorage.clear()
  feedbackApi = null
  installMockHandler()
})

afterEach(() => {
  window.fetch = realFetch
})

describe('the dev mock can still save', () => {
  it('acknowledges a match saved through the mock handler', async () => {
    renderPanel()
    await userEvent.click(await screen.findByTestId('match-pill-1', {}, { timeout: 5000 }))
    await userEvent.type(notesBox(), 'saved under dev:mock')
    await userEvent.click(screen.getByRole('button', { name: /Submit/ }))

    await waitFor(() => expect(feedbackApi?.pendingSave).toBeNull(), { timeout: 5000 })
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(feedbackApi?.lastSubmission?.kind).toBe('submit')
    expect(feedbackApi?.lastSubmission?.entry.outcome).toBe('matched_rank')
    expect(feedbackApi?.lastSubmission?.entry.notes).toBe('saved under dev:mock')
  })

  it('acknowledges a note-required Skip through the mock handler', async () => {
    renderPanel()
    await screen.findByTestId('match-pill-1', {}, { timeout: 5000 })
    await userEvent.type(notesBox(), 'illegible opening')
    await userEvent.click(screen.getByRole('button', { name: /Skip/ }))

    await waitFor(() => expect(feedbackApi?.pendingSave).toBeNull(), { timeout: 5000 })
    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(feedbackApi?.lastSubmission?.kind).toBe('skip')
    expect(feedbackApi?.lastSubmission?.entry.outcome).toBe('skipped')
    expect(feedbackApi?.draft.notes).toBe('')
  })
})
