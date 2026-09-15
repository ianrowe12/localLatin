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
 * The blue action AS THE PANEL WIRES IT (issue #196, review finding 1).
 *
 * `NoneOfTopTenAction.test.tsx` renders the component through a harness that
 * mirrors the panel; this file renders the real `FeedbackPanel` inside the real
 * provider tree, because the defect it exists to catch lived exactly in the
 * seam between them: the component held a receipt and the panel unmounted it in
 * the same breath by closing the form.
 *
 * Two journeys, both end to end:
 *
 *   record   -> press None, type a key, press Record, and read the sentence
 *               naming the branch the server took;
 *   revisit  -> come back to the document later and find that answer shown
 *               back, with Change rather than a live Record over an answer
 *               already in an append-only log.
 */

const QUERY_ID = 7
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
/** What GET /api/feedback/latest answers: the reviewer's own stored decision. */
let latest: SavedRow | null = null
/** Whether the next POST is accepted, and with what the server would answer. */
let postStatus = 201

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
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (init?.method === 'POST' && url.includes('/api/feedback')) {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>
        posted.push(body)
        if (postStatus >= 400) {
          return jsonResponse(
            { error: { code: 'RANKING_NOT_EVALUABLE', message: 'Refused.' } },
            postStatus,
          )
        }
        const key = typeof body.ccl_key === 'string' ? body.ccl_key.trim() : ''
        // The server's own resolution, in miniature: this fixture's corpus has
        // `candidate-1` labelled at rank 1 and nothing else.
        const labelled = key === 'candidate-1'
        const row = {
          id: posted.length,
          query_id: QUERY_ID,
          outcome: 'none_of_top_k',
          ccl_key: key || null,
          ccl_key_action: !key
            ? null
            : labelled
              ? 'matched_labelled_dir'
              : 'created_reviewer_dir',
          ccl_key_dir: !key
            ? null
            : labelled
              ? 'candidate-1'
              : 'reviewer-dir-new',
          ccl_key_rank: labelled ? 1 : null,
        }
        // What a revisit would then find.
        latest = {
          outcome: 'none_of_top_k',
          notes: typeof body.notes === 'string' ? body.notes : '',
          ccl_key: row.ccl_key,
          ccl_key_action: row.ccl_key_action,
          ccl_key_dir: row.ccl_key_dir,
          ccl_key_rank: row.ccl_key_rank,
        }
        return jsonResponse(row, postStatus)
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
      if (url.includes('/api/feedback/latest')) {
        return jsonResponse(
          latest === null
            ? null
            : {
                id: 99,
                query_id: QUERY_ID,
                timestamp: '2026-09-15 10:00:00',
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
      if (url.includes('/api/queries/next')) return jsonResponse({ file_id: QUERY_ID })
      if (url.includes('/predictions')) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
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

function renderPanel() {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <SavedDirectoryProvider accountKey="none-of-top-ten-panel">
            <PredictionProvider>
              <FeedbackProvider>
                <SelectQuery />
                <FeedbackPanel />
              </FeedbackProvider>
            </PredictionProvider>
          </SavedDirectoryProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>,
  )
}

beforeEach(() => {
  latest = null
  postStatus = 201
  localStorage.clear()
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function pressNone(): Promise<void> {
  await userEvent.click(await screen.findByTestId('match-pill-none'))
}

describe('recording an answer through the panel', () => {
  it('leaves the receipt on screen after the panel closes the form', async () => {
    renderPanel()
    await pressNone()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    // The form closes -- the panel clears the None selection -- and the
    // sentence naming the branch survives it. This is the seam the component
    // tests could not see.
    await waitFor(() =>
      expect(screen.getByTestId('none-of-top-k-recorded')).toBeTruthy(),
    )
    expect(screen.queryByTestId('none-of-top-k-form')).toBeNull()
    expect(screen.getByTestId('none-of-top-k-receipt').textContent).toBe(
      'Recorded, and a group under CTOU.567.16 now holds this document.',
    )
    expect(
      (screen.getByTestId('match-pill-none') as HTMLButtonElement).getAttribute(
        'aria-pressed',
      ),
    ).toBe('false')
    expect(posted).toHaveLength(1)
  })

  it('reports a key that named a shortlisted directory as shortlisted', async () => {
    renderPanel()
    await pressNone()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'candidate-1')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toBe(
      'Recorded: matches directory candidate-1, which is in the shortlist at #1. Nothing new was created.',
    )
  })

  it('records a non-match with no key and says so', async () => {
    renderPanel()
    await pressNone()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toBe('Recorded: none of the ranked candidates match.')
    expect(posted[0].ccl_key).toBeUndefined()
  })

  it('keeps the form and the draft when the server refuses', async () => {
    postStatus = 422
    renderPanel()
    await pressNone()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    expect((await screen.findByTestId('none-of-top-k-error')).textContent).toContain(
      'Nothing was recorded',
    )
    expect(screen.getByTestId('none-of-top-k-form')).toBeTruthy()
    expect((screen.getByTestId('ccl-key-input') as HTMLInputElement).value).toBe(
      'CTOU.567.16',
    )
    expect(screen.queryByTestId('none-of-top-k-recorded')).toBeNull()
  })

  it('withholds the panel Save while None is held, so one answer is one row', async () => {
    renderPanel()
    await pressNone()
    const submit = screen.getByRole('button', { name: /submit|save/i })
    expect((submit as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('coming back to a document already answered', () => {
  it('shows the stored answer rather than an empty form', async () => {
    latest = {
      outcome: 'none_of_top_k',
      notes: 'the source is elsewhere',
      ccl_key: 'CTOU.567.16',
      ccl_key_action: 'joined_reviewer_dir',
      ccl_key_dir: 'reviewer-dir-abc',
      ccl_key_rank: null,
    }
    renderPanel()

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toBe(
      'Recorded, and this document joined the group under CTOU.567.16.',
    )
    // No live Record button over an answer already in an append-only log, and
    // the None control is not pressed on the reviewer's behalf.
    expect(screen.queryByTestId('none-of-top-k-submit')).toBeNull()
    expect(
      screen.getByTestId('match-pill-none').getAttribute('aria-pressed'),
    ).toBe('false')
    // The prose still prefills, as it does for any restored row.
    await waitFor(() =>
      expect(
        (screen.getByPlaceholderText(/note/i) as HTMLTextAreaElement).value,
      ).toBe('the source is elsewhere'),
    )
  })

  it('survives the round trip: record, leave, come back', async () => {
    const view = renderPanel()
    await pressNone()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))
    await screen.findByTestId('none-of-top-k-recorded')

    // The reviewer leaves and comes back: a fresh mount with nothing in local
    // state, reading the same row the server now holds.
    view.unmount()
    renderPanel()

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toContain('CTOU.567.16')
    expect(screen.queryByTestId('none-of-top-k-submit')).toBeNull()
  })

  it('offers Change, which reopens an empty field over the stored answer', async () => {
    latest = {
      outcome: 'none_of_top_k',
      notes: '',
      ccl_key: 'CTOU.567.16',
      ccl_key_action: 'created_reviewer_dir',
      ccl_key_dir: 'reviewer-dir-abc',
      ccl_key_rank: null,
    }
    renderPanel()

    await userEvent.click(await screen.findByTestId('none-of-top-k-change'))
    expect((screen.getByTestId('ccl-key-input') as HTMLInputElement).value).toBe('')
    expect(screen.getByTestId('none-of-top-k-previous').textContent).toContain(
      'CTOU.567.16',
    )
    // Change is not a write: nothing has been posted.
    expect(posted).toHaveLength(0)
  })

  it('does not show one reviewer\'s key as another reviewer\'s answer', async () => {
    // The merged prefill carries a colleague's NOTE with the caller's own
    // (absent) decision, which is what `/api/feedback/latest` returns for a
    // document this reviewer has not answered. Nothing about a key may survive
    // that merge; the server drops it, and the panel must not resurrect it.
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
    expect(screen.queryByTestId('none-of-top-k-recorded')).toBeNull()
    expect(screen.queryByTestId('none-of-top-k-receipt')).toBeNull()
  })
})
