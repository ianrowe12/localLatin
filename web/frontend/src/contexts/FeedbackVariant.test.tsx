import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FeedbackPanel from '../components/feedback/FeedbackPanel'
import { AppProvider, useApp } from './AppContext'
import { FeedbackProvider } from './FeedbackContext'
import { PredictionProvider } from './PredictionContext'
import { ReviewerProvider } from './ReviewerContext'
import { DEFAULT_VARIANT } from '../api/variants'

const QUERY_ID = 3
const MODEL = 'bowphs_LaTa'
const ACCOUNT_ID = 2
const DRAFT_STORAGE_KEY = 'locallatin-feedback-drafts'

let posted: Record<string, unknown>[] = []

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  posted = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'POST' && init.body && url.includes('/api/feedback')) {
        posted.push(JSON.parse(String(init.body)))
        return jsonResponse({ success: true })
      }
      if (url.includes('/api/auth/me')) {
        return jsonResponse({
          id: ACCOUNT_ID,
          username: 'bob',
          display_name: 'Bob Bibliothecarius',
          role: 'reviewer',
          approval_status: 'approved',
        })
      }
      if (url.includes('/api/feedback/latest')) return jsonResponse(null)
      if (url.includes('/api/queries/next')) return jsonResponse({ file_id: QUERY_ID })
      if (url.includes('/predictions')) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-3.txt',
          model: MODEL,
          variant: DEFAULT_VARIANT,
          predictions: [
            {
              rank: 1,
              dir_name: 'Can.apost.42',
              score: 0.55,
              dir_files: ['a.txt'],
              preview_text: 'preview',
              candidate_files: [{ filename: 'a.txt', text: 'canon text' }],
            },
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
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return null
}

function renderPanel() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <PredictionProvider>
          <FeedbackProvider>
            <SelectQuery />
            <FeedbackPanel />
          </FeedbackProvider>
        </PredictionProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('feedback carries the active variant', () => {
  it('records the served variant on the submitted row', async () => {
    renderPanel()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Match prediction #1' }),
    )
    await userEvent.click(screen.getByRole('button', { name: /Submit/ }))

    await waitFor(() => expect(posted.length).toBeGreaterThan(0))
    expect(posted[0]).toMatchObject({
      query_id: QUERY_ID,
      model_slug: MODEL,
      variant: DEFAULT_VARIANT,
      outcome: 'matched_rank',
      correct_rank: 1,
      // The rank's directory as the reviewer saw it, as a save precondition.
      // `correct_dir` is null on purpose: the server resolves the assignment
      // from its own snapshot and this field was never its authority.
      expected_candidate_dirs: { '1': 'Can.apost.42' },
      correct_dir: null,
    })
  })

  it('keys drafts per variant so answers never bleed across them', async () => {
    // Asserts the behaviour rather than the key format, so a key-format
    // refactor cannot quietly break isolation while the test still passes.
    // The same query, model and account under a different variant ranks
    // differently, so its draft must not surface here (issue #48).
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [
          `acct${ACCOUNT_ID}:${QUERY_ID}-${MODEL}-raw`,
          {
            correctRank: 1,
            selectedRanks: [1],
            selections: [{ rank: 1, dirName: 'Something.else', source: 'model' }],
            notes: 'raw answer',
          },
        ],
      ]),
    )
    renderPanel()

    const pill = await screen.findByRole('button', { name: 'Match prediction #1' })
    expect(pill.getAttribute('aria-pressed')).toBe('false')
    expect(
      (screen.getByPlaceholderText('Add notes for this query...') as HTMLTextAreaElement)
        .value,
    ).toBe('')
    // And the other variant's draft is still there, untouched.
    expect(localStorage.getItem(DRAFT_STORAGE_KEY)).toContain('raw answer')
  })
})
