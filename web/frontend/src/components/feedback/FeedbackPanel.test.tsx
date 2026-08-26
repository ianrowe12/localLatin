import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FeedbackPanel from './FeedbackPanel'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import type { FeedbackEntry } from '../../api/feedback'

const QUERY_ID = 7
const MODEL = 'bowphs_LaTa'
const DRAFT_STORAGE_KEY = 'locallatin-feedback-drafts'

/** A note saved by reviewer A, as GET /api/feedback/latest returns it. */
const NOTE_FROM_ALICE: FeedbackEntry = {
  id: 41,
  query_id: QUERY_ID,
  timestamp: '2026-08-24 09:15:00',
  model_slug: MODEL,
  variant: 'sif_abtt',
  outcome: 'matched_rank',
  correct_rank: 2,
  correct_dir: 'candidate-2',
  selected_ranks: null,
  notes: 'Second candidate shares the incipit.',
  reviewer: 'Alice Antiqua',
  reviewer_account_id: 1,
  reviewer_username: 'alice',
  schema_version: 2,
}

let latestEntry: FeedbackEntry | null = NOTE_FROM_ALICE

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** Signed in as reviewer B, who has never touched this query. */
function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/auth/me')) {
        return jsonResponse({
          id: 2,
          username: 'bob',
          display_name: 'Bob Bibliothecarius',
          role: 'reviewer',
          approval_status: 'approved',
        })
      }
      if (url.includes('/api/feedback/latest')) {
        return jsonResponse(latestEntry)
      }
      if (url.includes('/predictions')) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          model: MODEL,
          variant: 'sif_abtt',
          predictions: [1, 2, 3].map((rank) => ({
            rank,
            dir_name: `candidate-${rank}`,
            score: 1 - rank / 100,
            dir_files: [`${rank}.txt`],
            preview_text: 'preview',
            candidate_files: null,
          })),
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
        <FeedbackProvider>
          <SelectQuery />
          <FeedbackPanel />
        </FeedbackProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

function notesBox(): HTMLTextAreaElement {
  return screen.getByPlaceholderText('Add notes for this query...') as HTMLTextAreaElement
}

beforeEach(() => {
  latestEntry = NOTE_FROM_ALICE
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('shared notes across reviewers', () => {
  it('prefills another reviewer’s note and decision, attributed to them', async () => {
    renderPanel()

    await waitFor(() =>
      expect(notesBox().value).toBe('Second candidate shares the incipit.'),
    )
    // The decision travels with the note: rank 2 comes back pressed.
    expect(
      screen
        .getByRole('button', { name: 'Match prediction #2' })
        .getAttribute('aria-pressed'),
    ).toBe('true')

    const attribution = screen.getByTestId('note-attribution')
    expect(attribution.textContent).toContain('Last note by alice')
    expect(attribution.textContent).toContain('2026-08-24')
    expect(attribution.textContent).not.toContain('unsaved draft')
  })

  it('falls back to the display name when the author has no account row', async () => {
    latestEntry = {
      ...NOTE_FROM_ALICE,
      reviewer_account_id: null,
      reviewer_username: null,
    }
    renderPanel()

    await waitFor(() =>
      expect(screen.getByTestId('note-attribution').textContent).toContain(
        'Last note by Alice Antiqua',
      ),
    )
  })

  it('shows no attribution when nobody has reviewed the query yet', async () => {
    latestEntry = null
    renderPanel()

    await waitFor(() => expect(notesBox()).toBeTruthy())
    expect(screen.queryByTestId('note-attribution')).toBeNull()
  })

  it('keeps an in-progress local draft instead of the shared note', async () => {
    // Reviewer B was mid-sentence when the shared prefill arrived.
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [
          `${QUERY_ID}-${MODEL}-sif_abtt`,
          { correctRank: 3, notes: 'still typing this' },
        ],
      ]),
    )
    renderPanel()

    await waitFor(() => expect(notesBox().value).toBe('still typing this'))
    // Attribution still reports the shared note, and says the box is not it.
    const attribution = await screen.findByTestId('note-attribution')
    expect(attribution.textContent).toContain('Last note by alice')
    expect(attribution.textContent).toContain('your unsaved draft is shown below')

    // Give the prefill every chance to clobber the draft before asserting.
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(notesBox().value).toBe('still typing this')
    expect(
      screen
        .getByRole('button', { name: 'Match prediction #3' })
        .getAttribute('aria-pressed'),
    ).toBe('true')
  })
})
