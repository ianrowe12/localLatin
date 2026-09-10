import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import { TokenProvider } from '../../contexts/TokenContext'
import FeedbackPanel from './FeedbackPanel'
import CenterArea from '../layout/CenterArea'
import PredictionList from '../predictions/PredictionList'
import type { FeedbackEntry } from '../../api/feedback'

/**
 * Assessment controls answer to the evidence on screen (issue #157).
 *
 * The panel used to draw ten pills from `Array.from({ length: maxRank })` with
 * a fallback of ten, so a failed request offered ten pressable ranks and let
 * Submit post "none of top 10" against a ranking nobody had ever seen. Drafts
 * remembered a bare rank number, so an answer could be re-pointed at whatever
 * directory later occupied it, and they were stored per browser rather than per
 * reviewer, so one reviewer's unsent choice greeted the next one as their own.
 *
 * These tests drive the real components under App.tsx's provider tree.
 */

const QUERY_ID = 7
const MODEL = 'bowphs_LaTa'
const OTHER_MODEL = 'google_mt5-base'
const DRAFT_STORAGE_KEY = 'locallatin-feedback-drafts'
const draftKeyFor = (accountId: number, model = MODEL, variant = 'sif_abtt') =>
  `acct${accountId}:${QUERY_ID}-${model}-${variant}`

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function modelCard(rank: number, dir?: string, text = 'candidate text') {
  return {
    rank,
    dir_name: dir ?? `candidate-${rank}`,
    score: 1 - rank / 100,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text }],
    source: 'model',
  }
}

function reviewerCard(rank: number, dir: string, label = 'Unattested homily') {
  return {
    ...modelCard(rank, dir, 'seed text'),
    source: 'reviewer',
    label,
    created_by: 'Abigail',
    seed_query_id: 3,
  }
}

type Answer = {
  status?: string | null
  predictions?: unknown[]
  httpStatus?: number
}

let answers: Record<string, Answer> = {}
let queued: Record<string, Answer[]> = {}
let predictionRequests = 0
let posted: Record<string, unknown>[] = []
let postResult: { status: number; body: unknown } = {
  status: 200,
  body: { success: true },
}
/** Set to make the POST reject the way a dropped connection does. */
let postFails: Error | null = null
let postAttempts = 0
let account = { id: 2, username: 'bob', display_name: 'Bob Bibliothecarius' }
let latestEntry: FeedbackEntry | null = null

function answerFor(model: string): Answer {
  return queued[model]?.shift() ?? answers[model] ?? { predictions: [modelCard(1)] }
}

function installFetch(): void {
  predictionRequests = 0
  posted = []
  postAttempts = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const params = new URL(url, 'http://localhost').searchParams

      if (init?.method === 'POST' && url.includes('/api/feedback')) {
        postAttempts += 1
        // A network failure is not an HTTP status: fetch rejects, and the body
        // may well have reached the server first.
        if (postFails) throw postFails
        posted.push(JSON.parse(String(init.body)))
        return jsonResponse(postResult.body, postResult.status)
      }
      if (url.includes('/api/auth/me')) {
        return jsonResponse({
          ...account,
          role: 'reviewer',
          approval_status: 'approved',
        })
      }
      if (url.includes('/api/models')) {
        return jsonResponse(
          [MODEL, OTHER_MODEL].map((slug) => ({
            slug,
            display_name: slug,
            layer: 4,
            pooling: 'sif',
            prediction_count: 2238,
            available_variants: ['sif_abtt'],
            default_variant: 'sif_abtt',
            confidence_bands: { no_match: 0.2, verify: 0.4 },
            supports_reviewer_dirs: true,
          })),
        )
      }
      if (url.includes('/api/feedback/latest')) return jsonResponse(latestEntry)
      if (url.includes('/api/queries/next')) return jsonResponse({ file_id: QUERY_ID })
      // Only reached by the mounted-evidence tests; CenterArea reads the query
      // text from here and the candidate text out of the prediction payload.
      if (url.includes('/token_map')) {
        return jsonResponse({ error: { message: 'no artifact' } }, 404)
      }
      if (/\/api\/query\/\d+$/.test(new URL(url, 'http://localhost').pathname)) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          text: 'query text under review',
          dir_name: null,
        })
      }
      if (url.includes('/predictions')) {
        predictionRequests += 1
        const model = params.get('model') ?? ''
        const answer = answerFor(model)
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
          seeded_dirs: [],
        })
      }
      return jsonResponse(null)
    }),
  )
}

function SelectQuery({ model = MODEL }: { model?: string }) {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(model)
  }, [model, setActiveQueryId, setActiveModel])
  return null
}

function renderPanel(
  options: { withList?: boolean; withCenter?: boolean; model?: string } = {},
) {
  return render(
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <PredictionProvider>
            <FeedbackProvider>
              <SelectQuery model={options.model} />
              {options.withList === true && <PredictionList />}
              {/* The evidence pane itself, so a test can ask what is actually
                  legible on screen rather than trusting the payload. */}
              {options.withCenter === true && <CenterArea />}
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

function submitButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /Submit/ }) as HTMLButtonElement
}

function skipButton(): HTMLButtonElement {
  return screen.getByRole('button', { name: /Skip/ }) as HTMLButtonElement
}

function pill(rank: number): HTMLButtonElement {
  return screen.getByTestId(`match-pill-${rank}`) as HTMLButtonElement
}

function nonePill(): HTMLButtonElement {
  return screen.getByTestId('match-pill-none') as HTMLButtonElement
}

function storedDrafts(): Record<string, unknown> {
  const raw = localStorage.getItem(DRAFT_STORAGE_KEY)
  return Object.fromEntries(JSON.parse(raw ?? '[]') as [string, unknown][])
}

beforeEach(() => {
  answers = {}
  queued = {}
  latestEntry = null
  account = { id: 2, username: 'bob', display_name: 'Bob Bibliothecarius' }
  postResult = { status: 200, body: { success: true } }
  postFails = null
  installFetch()
})

// The panel advances to the next query on a 500ms timer after a save, so a
// test can finish with that request still pending. Keeping the mock installed
// until the file is done means such a straggler hits the fixture rather than
// jsdom's real fetch, which would reject into nobody's catch block.
afterAll(() => {
  vi.unstubAllGlobals()
})

describe('no evidence, no assessment', () => {
  it.each([
    ['a failed request', { httpStatus: 500 }, 'The ranking did not load'],
    [
      'an excluded query',
      { status: 'excluded_blank_source', predictions: [] },
      'Not assessable for this model',
    ],
    ['an unexplained empty ranking', { predictions: [] }, 'No ranking came back'],
  ])('offers no rank and no Submit for %s', async (_name, answer, title) => {
    answers[MODEL] = answer as Answer
    renderPanel()

    const notice = await screen.findByTestId('assessment-unavailable')
    expect(notice.textContent).toContain(title)
    expect(screen.queryByTestId('match-pill-1')).toBeNull()
    expect(screen.queryByTestId('match-pill-none')).toBeNull()
    expect(submitButton().disabled).toBe(true)
  })

  it('keeps notes as a draft through a failure and offers them back when it loads', async () => {
    answers[MODEL] = { httpStatus: 500 }
    const view = renderPanel()

    await screen.findByTestId('assessment-unavailable')
    await userEvent.type(notesBox(), 'ranking never arrived')
    await waitFor(() =>
      expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({
        notes: 'ranking never arrived',
      }),
    )

    // A failed request is not evidence about the document, so the reviewer's
    // words survive it: they come back when the ranking does.
    view.unmount()
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2)] }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    expect(notesBox().value).toBe('ranking never arrived')
    expect(pill(1).getAttribute('aria-pressed')).toBe('false')
    expect(submitButton().disabled).toBe(true)
  })

  it('still allows a deliberate skip with a note when nothing loaded', async () => {
    answers[MODEL] = { httpStatus: 500 }
    renderPanel()

    await screen.findByTestId('assessment-unavailable')
    expect(skipButton().disabled).toBe(false)
    await userEvent.click(skipButton())
    // A skip is a decision, so it needs its reason. Nothing is posted without.
    expect(posted).toHaveLength(0)

    await userEvent.type(notesBox(), 'ranking will not load for this one')
    await userEvent.click(skipButton())
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      outcome: 'skipped',
      correct_rank: null,
      notes: 'ranking will not load for this one',
    })
    expect(posted[0].expected_candidate_dirs).toBeUndefined()
  })
})

describe('the pills are the ranking', () => {
  it('draws actual sparse ranks, including anchored reviewer directories', async () => {
    answers[MODEL] = {
      predictions: [
        modelCard(1),
        reviewerCard(11, 'reviewer-dir-a'),
        reviewerCard(12, 'reviewer-dir-b', 'Second reviewer directory'),
      ],
    }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    // No phantom #2..#10, and the reviewer directories keep the ranks the API
    // anchored them at rather than being renumbered 2 and 3.
    expect(screen.queryByTestId('match-pill-2')).toBeNull()
    expect(pill(11).textContent).toBe('#11')
    expect(pill(12).textContent).toBe('#12')
    expect(
      screen.getByRole('button', { name: 'Match reviewer directory #11' }),
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Match prediction #1' })).toBeTruthy()
    // None is a judgement about the model's candidates only.
    expect(nonePill().textContent).toBe('None of the 1 model candidate')
  })

  it('lets a reviewer directory at rank 11 be chosen and saved as rank 11', async () => {
    answers[MODEL] = { predictions: [modelCard(1), reviewerCard(11, 'reviewer-dir-a')] }
    renderPanel()

    await userEvent.click(await screen.findByTestId('match-pill-11'))
    await userEvent.click(submitButton())

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      outcome: 'matched_rank',
      correct_rank: 11,
      selected_ranks: [11],
      expected_candidate_dirs: { '11': 'reviewer-dir-a' },
      correct_dir: null,
    })
  })

  it('disables only the unreadable candidate, and blocks None while it is offered', async () => {
    answers[MODEL] = {
      predictions: [
        modelCard(1),
        { ...modelCard(2), candidate_files: [{ filename: '2.txt', text: '   ' }] },
      ],
    }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    expect(pill(2).disabled).toBe(true)
    // Rejecting a candidate nobody can read would claim a reading that is not
    // here, and the server refuses it (RANKING_NOT_EVALUABLE).
    expect(nonePill().disabled).toBe(true)
    expect(screen.getByTestId('assessment-notice').textContent).toContain(
      'cannot be read on this screen',
    )

    // The readable candidate is still a real answer.
    expect(pill(1).disabled).toBe(false)
    await userEvent.click(pill(1))
    await userEvent.click(submitButton())
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({ correct_rank: 1 })
  })

  it('refuses the whole ranking when no model candidate can be read', async () => {
    // A deployment missing the labelled texts serves model candidates with no
    // text beside reviewer directories that read fine. There is nothing to
    // judge the document against, so no pill may be pressed -- and the reason
    // has to be on screen, or a disabled Submit reads as a broken app.
    answers[MODEL] = {
      predictions: [
        { ...modelCard(1), candidate_files: [] },
        reviewerCard(11, 'reviewer-dir-a'),
      ],
    }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    expect(pill(1).disabled).toBe(true)
    expect(pill(11).disabled).toBe(true)
    expect(nonePill().disabled).toBe(true)
    expect(screen.getByTestId('assessment-unavailable').textContent).toContain(
      'No readable candidate evidence',
    )
    expect(submitButton().disabled).toBe(true)

    // Clicking is refused visibly rather than silently: nothing is drafted.
    await userEvent.click(pill(11))
    expect(pill(11).getAttribute('aria-pressed')).toBe('false')
    expect(localStorage.getItem(DRAFT_STORAGE_KEY)).toBe('[]')

    // Skip with a note is still the way out.
    await userEvent.type(notesBox(), 'candidate texts are missing here')
    await userEvent.click(skipButton())
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({ outcome: 'skipped' })
  })

  it('refuses a candidate whose only readable witness is off screen', async () => {
    // Mounted with the real evidence pane, because the claim under test is
    // about what a reviewer can read, not about what the payload contains.
    // The server would accept this directory (it has one readable file), but
    // this build shows candidate_files[0] and has no witness selector, so the
    // readable file is text nobody is ever shown. Enabling the pill would
    // solicit a judgement of evidence that never reached the screen; enabling
    // None would let it be rejected the same way. Both stay shut until a
    // witness selector exists (issue #163).
    answers[MODEL] = {
      predictions: [
        {
          ...modelCard(1),
          candidate_files: [
            { filename: 'blank.txt', text: '' },
            { filename: 'hidden-witness.txt', text: 'incipit sermo lupi' },
          ],
        },
      ],
    }
    renderPanel({ withCenter: true, withList: true })

    await screen.findByTestId('match-pill-1')

    // The hidden witness is nowhere on screen: not its text, not its name.
    expect(document.body.textContent).not.toContain('incipit sermo lupi')
    expect(document.body.textContent).not.toContain('hidden-witness.txt')

    // So neither answer about it is offered.
    expect(pill(1).disabled).toBe(true)
    expect(nonePill().disabled).toBe(true)
    expect(submitButton().disabled).toBe(true)
    await userEvent.click(pill(1))
    expect(pill(1).getAttribute('aria-pressed')).toBe('false')
    expect(localStorage.getItem(DRAFT_STORAGE_KEY)).toBe('[]')

    // The #156 note still explains the blank pane honestly, including that
    // other files in the directory do carry text.
    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('blank.txt')
    expect(note.textContent).toContain('no readable text')

    // Inspection is untouched: the rank is still selectable for reading, and
    // it still shows the blank first witness rather than silently swapping in
    // the readable one.
    await userEvent.click(screen.getByRole('button', { name: /Prediction rank 1:/ }))
    expect((await screen.findByTestId('candidate-evidence-note')).textContent).toContain(
      'blank.txt',
    )
    expect(document.body.textContent).not.toContain('incipit sermo lupi')

    // Skip with a note remains the way to report it.
    await userEvent.type(notesBox(), 'only the second file has text')
    await userEvent.click(skipButton())
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({ outcome: 'skipped' })
  })

  it('records an explicit None without any candidate identity', async () => {
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2)] }
    renderPanel()

    await userEvent.click(await screen.findByTestId('match-pill-none'))
    await userEvent.click(submitButton())

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      outcome: 'none_of_top_k',
      correct_rank: 0,
      correct_dir: null,
    })
    expect(posted[0].expected_candidate_dirs).toBeUndefined()
    expect(posted[0].selected_ranks).toBeUndefined()
  })
})

describe('a draft remembers what was chosen, not where it sat', () => {
  it('sends every selected rank with the directory the reviewer saw, in click order', async () => {
    answers[MODEL] = {
      predictions: [modelCard(1), modelCard(2), reviewerCard(11, 'reviewer-dir-a')],
    }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    await userEvent.click(screen.getByLabelText('Select multiple'))
    await userEvent.click(pill(11))
    await userEvent.click(pill(1))
    // Two choices are recorded, but only one directory receives the document,
    // and the reviewer is told which.
    expect(screen.getByTestId('canonical-choice-note').textContent).toContain('#11')
    await userEvent.click(submitButton())

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      outcome: 'matched_rank',
      // The first click is the canonical answer, not the lowest rank.
      correct_rank: 11,
      selected_ranks: [11, 1],
      expected_candidate_dirs: { '11': 'reviewer-dir-a', '1': 'candidate-1' },
    })
  })

  it('drops a choice whose rank now holds a different directory, and says so', async () => {
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [
          draftKeyFor(2),
          {
            correctRank: 11,
            selectedRanks: [11],
            selections: [{ rank: 11, dirName: 'reviewer-dir-a', source: 'reviewer' }],
            notes: 'matches the homily',
          },
        ],
      ]),
    )
    answers[MODEL] = {
      predictions: [modelCard(1), reviewerCard(11, 'reviewer-dir-b', 'Another one')],
    }
    renderPanel()

    await screen.findByTestId('match-pill-11')
    expect(pill(11).getAttribute('aria-pressed')).toBe('false')
    const notice = await screen.findByTestId('assessment-notice')
    expect(notice.textContent).toContain('reviewer-dir-b')
    expect(notice.textContent).toContain('reviewer-dir-a')
    expect(submitButton().disabled).toBe(true)
    // The note survives; only the choice that lost its meaning is gone.
    expect(notesBox().value).toBe('matches the homily')
    await waitFor(() =>
      expect(storedDrafts()[draftKeyFor(2)]).toEqual({
        correctRank: null,
        notes: 'matches the homily',
      }),
    )
  })

  it('drops a choice whose rank is no longer offered at all', async () => {
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [
          draftKeyFor(2),
          {
            correctRank: 3,
            selectedRanks: [3],
            selections: [{ rank: 3, dirName: 'candidate-3', source: 'model' }],
            notes: '',
          },
        ],
      ]),
    )
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2)] }
    renderPanel()

    const notice = await screen.findByTestId('assessment-notice')
    expect(notice.textContent).toContain('no longer offered')
    expect(submitButton().disabled).toBe(true)
  })

  it('keeps an identity-less restored choice pending until it is confirmed', async () => {
    // The server records the directory for the canonical rank only, so a
    // restored second choice comes back as a bare rank. Pressing it as an
    // answer would assign whatever now sits there.
    latestEntry = {
      id: 5,
      query_id: QUERY_ID,
      timestamp: '2026-09-01 10:00:00',
      model_slug: MODEL,
      variant: 'sif_abtt',
      outcome: 'matched_rank',
      correct_rank: 11,
      correct_dir: 'reviewer-dir-a',
      selected_ranks: [11, 1],
      notes: 'two readings',
      reviewer: 'Bob Bibliothecarius',
      reviewer_account_id: 2,
      reviewer_username: 'bob',
      schema_version: 2,
    }
    answers[MODEL] = { predictions: [modelCard(1), reviewerCard(11, 'reviewer-dir-a')] }
    renderPanel()

    await waitFor(() => expect(pill(11).getAttribute('aria-pressed')).toBe('true'))
    expect(pill(1).getAttribute('aria-pressed')).toBe('false')
    expect(pill(1).getAttribute('data-unconfirmed')).toBe('true')
    // A half-confirmed answer is not saved as its confirmed half.
    expect(submitButton().disabled).toBe(true)
    expect(screen.getByTestId('assessment-notice').textContent).toContain(
      'Select it again',
    )

    // A restored multi-answer stays in multi-select mode, so confirming the
    // second choice does not replace the first.
    expect((screen.getByLabelText('Select multiple') as HTMLInputElement).checked).toBe(
      true,
    )
    await userEvent.click(pill(1))
    expect(submitButton().disabled).toBe(false)
    await userEvent.click(submitButton())
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({
      correct_rank: 11,
      selected_ranks: [11, 1],
      expected_candidate_dirs: { '11': 'reviewer-dir-a', '1': 'candidate-1' },
    })
  })

  it('survives a model round-trip without pruning against the other ranking', async () => {
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2), modelCard(3)] }
    answers[OTHER_MODEL] = { predictions: [modelCard(1)] }
    const view = renderPanel()

    await screen.findByTestId('match-pill-3')
    await userEvent.click(pill(3))
    await waitFor(() =>
      expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({ correctRank: 3 }),
    )

    // The other model offers one rank; the first model's draft is not its
    // business, and must not be reconciled against it.
    view.unmount()
    renderPanel({ model: OTHER_MODEL })
    await screen.findByTestId('match-pill-1')
    expect(screen.queryByTestId('match-pill-3')).toBeNull()
    expect(pill(1).getAttribute('aria-pressed')).toBe('false')
    // The first model's draft is untouched, ready for the reviewer's return.
    expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({
      correctRank: 3,
      selections: [{ rank: 3, dirName: 'candidate-3', source: 'model' }],
    })
  })
})

describe('drafts belong to a reviewer', () => {
  it('does not hand one reviewer’s unsent choice to the next', async () => {
    localStorage.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify([
        [
          draftKeyFor(2),
          {
            correctRank: 1,
            selectedRanks: [1],
            selections: [{ rank: 1, dirName: 'candidate-1', source: 'model' }],
            notes: 'bob was here',
          },
        ],
      ]),
    )
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2)] }
    account = { id: 3, username: 'carol', display_name: 'Carol Codex' }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    expect(pill(1).getAttribute('aria-pressed')).toBe('false')
    expect(notesBox().value).toBe('')
    // Bob's own draft is still his, waiting for him.
    expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({ notes: 'bob was here' })
  })

  it('quarantines an unowned pre-#157 draft instead of disclosing it', async () => {
    // A draft written before drafts were keyed by reviewer carries no account.
    // Showing its words to whoever signs in next would hand one person's
    // unsent sentence to another, and there is no evidence that says whose it
    // was. So the panel states that something is held and stops there: no
    // content, no copy button, and the stored bytes untouched, because
    // deleting it would destroy the only copy of a real reviewer's thinking.
    // Notes deliberately shared through a submitted review are a different
    // thing and still arrive via /api/feedback/latest.
    const LEGACY_KEY = `${QUERY_ID}-${MODEL}-sif_abtt`
    const legacyEntry = {
      correctRank: 2,
      selectedRanks: [2],
      notes: 'unsent thought from an older build',
    }
    const storedRaw = JSON.stringify([[LEGACY_KEY, legacyEntry]])
    localStorage.setItem(DRAFT_STORAGE_KEY, storedRaw)
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2)] }

    const expectQuarantined = async () => {
      const notice = await screen.findByTestId('legacy-draft-notice')
      expect(notice.textContent).not.toContain('unsent thought from an older build')
      expect(notice.textContent).toContain('not shown here')
      expect(screen.queryByTestId('adopt-legacy-note')).toBeNull()
      expect(document.body.textContent).not.toContain(
        'unsent thought from an older build',
      )
      expect(notesBox().value).toBe('')
      expect(pill(2).getAttribute('aria-pressed')).toBe('false')
    }

    // Reviewer A sees only the notice.
    const view = renderPanel()
    await screen.findByTestId('match-pill-1')
    await expectQuarantined()

    // A signs out through the real control...
    await userEvent.click(screen.getByRole('button', { name: 'sign out' }))
    await waitFor(() => expect(screen.queryByText('Bob Bibliothecarius')).toBeNull())

    // ...and B signs in. The unowned words are still nobody's.
    view.unmount()
    account = { id: 3, username: 'carol', display_name: 'Carol Codex' }
    renderPanel()
    await screen.findByTestId('match-pill-1')
    await expectQuarantined()

    // Neither session consumed, rewrote or dropped the stored entry.
    expect(storedDrafts()[LEGACY_KEY]).toEqual(legacyEntry)
  })
})

describe('save failures', () => {
  it('promises an empty log only when the server refused before writing', async () => {
    answers[MODEL] = { predictions: [modelCard(1), reviewerCard(11, 'reviewer-dir-a')] }
    postResult = {
      status: 409,
      body: {
        error: {
          code: 'CANDIDATE_IDENTITY_CHANGED',
          message: 'Rank 11 now resolves to a different directory.',
        },
      },
    }
    renderPanel()

    await userEvent.click(await screen.findByTestId('match-pill-11'))
    await userEvent.type(notesBox(), 'same incipit')
    await userEvent.click(submitButton())

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.textContent).toContain('different directory')
    // A named 4xx is raised by the router before db.insert, so this is the
    // one case where "nothing was saved" is a fact and not a hope.
    expect(error.getAttribute('data-outcome')).toBe('rejected')
    expect(error.textContent).toContain('Nothing was saved')
    // The answer is still on screen, still theirs to resubmit or revise.
    expect(pill(11).getAttribute('aria-pressed')).toBe('true')
    expect(notesBox().value).toBe('same incipit')
    expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({ correctRank: 11 })

    const before = predictionRequests
    await userEvent.click(screen.getByTestId('assessment-refresh-ranking'))
    await waitFor(() => expect(predictionRequests).toBeGreaterThan(before))
    // The reload returns the same directory at the same rank, which is the
    // same candidate however many requests have been made since: a refresh
    // must not cost the reviewer their answer.
    await waitFor(() => expect(pill(11).getAttribute('aria-pressed')).toBe('true'))
    expect(screen.queryByTestId('assessment-notice')).toBeNull()

    postResult = { status: 200, body: { success: true } }
    await userEvent.click(submitButton())
    await waitFor(() => expect(posted).toHaveLength(2))
    expect(posted[1]).toMatchObject({
      correct_rank: 11,
      expected_candidate_dirs: { '11': 'reviewer-dir-a' },
    })
  })

  it('refuses to guess after a 500, because the write may already have landed', async () => {
    // feedback.py commits the feedback row and only then writes reviewer
    // directory membership, so a 500 can arrive after the assessment is in an
    // append-only table. Telling the reviewer nothing was saved would send
    // them back to save a duplicate.
    answers[MODEL] = { predictions: [modelCard(1)] }
    postResult = {
      status: 500,
      body: { error: { code: 'internal_error', message: 'Database is locked.' } },
    }
    renderPanel()

    await userEvent.click(await screen.findByTestId('match-pill-1'))
    await userEvent.click(submitButton())

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(error.textContent).toContain('Database is locked.')
    expect(error.textContent).toContain('cannot tell whether your assessment was recorded')
    expect(error.textContent).not.toContain('Nothing was saved')
    // No reload button, and nothing retries on its own: a silent retry into an
    // append-only table is exactly the duplicate this copy warns against.
    expect(screen.queryByTestId('assessment-refresh-ranking')).toBeNull()
    expect(posted).toHaveLength(1)
    expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({ correctRank: 1 })
  })

  it('refuses to guess when the request never came back', async () => {
    // A dropped connection says nothing about what the server did with the
    // body it may already have received.
    answers[MODEL] = { predictions: [modelCard(1)] }
    renderPanel()

    await userEvent.click(await screen.findByTestId('match-pill-1'))
    await userEvent.type(notesBox(), 'ends mid-word')
    postFails = new TypeError('Failed to fetch')
    await userEvent.click(submitButton())

    const error = await screen.findByTestId('assessment-save-error')
    expect(error.getAttribute('data-outcome')).toBe('uncertain')
    expect(error.textContent).not.toContain('Nothing was saved')
    expect(screen.queryByTestId('assessment-refresh-ranking')).toBeNull()
    expect(notesBox().value).toBe('ends mid-word')
    expect(storedDrafts()[draftKeyFor(2)]).toMatchObject({ correctRank: 1 })

    // Still no automatic second attempt after the failure settles.
    await new Promise((r) => setTimeout(r, 50))
    expect(postAttempts).toBe(1)
  })

  it('refuses to send a skip with no note, and says so without a save claim', async () => {
    // A local refusal: nothing left the browser, so the panel does not have to
    // guess about the log at all. It asks for the note instead.
    answers[MODEL] = { predictions: [modelCard(1)] }
    renderPanel()

    await screen.findByTestId('match-pill-1')
    await userEvent.click(skipButton())

    expect(screen.queryByTestId('assessment-save-error')).toBeNull()
    expect(document.body.textContent).toContain('Add a note')
    expect(postAttempts).toBe(0)
  })
})

describe('looking is not answering', () => {
  it('leaves the assessment untouched when a candidate is inspected', async () => {
    answers[MODEL] = { predictions: [modelCard(1), modelCard(2), modelCard(3)] }
    renderPanel({ withList: true })

    await screen.findByTestId('match-pill-1')
    await userEvent.click(screen.getByText('candidate-2').closest('button')!)
    await userEvent.keyboard('3')

    for (const rank of [1, 2, 3]) {
      expect(pill(rank).getAttribute('aria-pressed')).toBe('false')
    }
    expect(nonePill().getAttribute('aria-pressed')).toBe('false')
    expect(submitButton().disabled).toBe(true)
    expect(localStorage.getItem(DRAFT_STORAGE_KEY)).toBe('[]')
  })
})
