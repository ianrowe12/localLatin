import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FeedbackPanel from './FeedbackPanel'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import type { FeedbackEntry } from '../../api/feedback'

const QUERY_ID = 7
const MODEL = 'bowphs_LaTa'
const DRAFT_STORAGE_KEY = 'locallatin-feedback-drafts'

/**
 * What GET /api/feedback/latest returns to reviewer B when reviewer A has left
 * a note and B has answered nothing: A's note and attribution, and no decision.
 * The server does that merge (issue #96) -- notes are the team's, answers are
 * each reviewer's own -- so the panel only ever sees one already-merged entry.
 */
const NOTE_FROM_ALICE: FeedbackEntry = {
  id: 41,
  query_id: QUERY_ID,
  timestamp: '2026-08-24 09:15:00',
  model_slug: MODEL,
  variant: 'sif_abtt',
  outcome: 'legacy_unresolved',
  correct_rank: null,
  correct_dir: null,
  selected_ranks: null,
  notes: 'Second candidate shares the incipit.',
  reviewer: 'Alice Antiqua',
  reviewer_account_id: 1,
  reviewer_username: 'alice',
  schema_version: 2,
}

/** The same note, merged with reviewer B's own earlier answer of rank 4. */
const ALICE_NOTE_WITH_MY_RANK: FeedbackEntry = {
  ...NOTE_FROM_ALICE,
  outcome: 'matched_rank',
  correct_rank: 4,
  correct_dir: 'candidate-4',
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
      if (url.includes('/api/queries/next')) {
        // The adversarial case for the post-submit render: the panel stays on
        // this query, so it is still mounted with the draft already deleted.
        return jsonResponse({ file_id: QUERY_ID })
      }
      if (url.includes('/predictions')) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          model: MODEL,
          variant: 'sif_abtt',
          predictions: [1, 2, 3, 4, 5].map((rank) => ({
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
  it('prefills another reviewer’s note, attributed, and presses no rank', async () => {
    renderPanel()

    await waitFor(() =>
      expect(notesBox().value).toBe('Second candidate shares the incipit.'),
    )
    // Alice's prose arrives; Alice's answer does not. Nothing is pressed, so
    // reviewer B cannot submit somebody else's decision by reflex.
    for (const rank of [1, 2, 3]) {
      expect(
        screen
          .getByRole('button', { name: `Match prediction #${rank}` })
          .getAttribute('aria-pressed'),
      ).toBe('false')
    }

    const attribution = screen.getByTestId('note-attribution')
    expect(attribution.textContent).toContain('Last note by alice')
    expect(attribution.textContent).toContain('2026-08-24')
    expect(attribution.textContent).not.toContain('unsaved draft')
  })

  it('restores the caller’s own rank alongside the shared note', async () => {
    latestEntry = ALICE_NOTE_WITH_MY_RANK
    renderPanel()

    await waitFor(() =>
      expect(
        screen
          .getByRole('button', { name: 'Match prediction #4' })
          .getAttribute('aria-pressed'),
      ).toBe('true'),
    )
    expect(notesBox().value).toBe('Second candidate shares the incipit.')
    expect(screen.getByTestId('note-attribution').textContent).toContain(
      'Last note by alice',
    )
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

  it('does not claim an unsaved draft after the draft has been saved', async () => {
    // submitFeedback deletes the draft, and the panel stays mounted for the
    // advance timeout. An "unsaved draft is shown below" line over an empty
    // box, under somebody else's name, is exactly the wrong thing to flash at
    // a reviewer who just saved.
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [`${QUERY_ID}-${MODEL}-sif_abtt`, { correctRank: 1, notes: 'my answer' }],
      ]),
    )
    renderPanel()

    await waitFor(() => expect(notesBox().value).toBe('my answer'))
    await userEvent.click(screen.getByRole('button', { name: /Submit/ }))

    await waitFor(() => expect(notesBox().value).toBe(''))
    const attribution = screen.getByTestId('note-attribution')
    expect(attribution.textContent).toContain('Last note by alice')
    expect(attribution.textContent).not.toContain('unsaved draft')
  })
})
