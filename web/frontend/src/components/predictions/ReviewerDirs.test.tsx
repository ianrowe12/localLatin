import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import AwaitingMatchBadge from './AwaitingMatchBadge'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import SavedDirectoryObservations from '../../contexts/SavedDirectoryObservations'
import PredictionList from './PredictionList'

const MODEL = 'bowphs_LaTa'
const QUERY_ID = 7

function modelsPayload() {
  return [
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
  ]
}

function modelCard(rank: number, score: number) {
  return {
    rank,
    dir_name: `candidate-${rank}`,
    score,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'candidate text' }],
    source: 'model' as const,
  }
}

/** An unranked reviewer directory, as the API serves one since issue #196. */
function reviewerCard(score: number, dirId = 'reviewer-dir-1') {
  return {
    dir_id: dirId,
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

/** Mutable per test, so a test can decide what the list currently contains. */
let predictions: {
  predictions: unknown[]
  seeded_dirs: unknown[]
  reviewer_dir_candidates?: unknown[]
} = { predictions: [modelCard(1, 0.91)], seeded_dirs: [] }

let posted: { url: string; body: unknown }[] = []

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(createResponse?: () => Response): void {
  posted = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (init?.method === 'POST') {
      posted.push({ url, body: JSON.parse(String(init.body)) })
      if (url.includes('/api/reviewer_dirs')) {
        return (
          createResponse?.() ??
          jsonResponse(
            {
              dir_id: 'reviewer-dir-1',
              label: 'Unattested homily',
              status: 'awaiting_match',
              seed_query_id: QUERY_ID,
              member_query_ids: [QUERY_ID],
              created_at: '2026-08-26 00:00:00',
              created_by: 'Abigail',
              model_slug: MODEL,
              variant: 'sif_abtt',
              best_match_score: 0.31,
              has_potential_match: false,
            },
            201,
          )
        )
      }
      return jsonResponse({})
    }
      if (url.includes('/api/reviewer_dirs')) {
        // The seed lookup (issue #161). It is a real endpoint, so the fixture
        // answers it like one: the directories this query seeds, and an empty
        // list when it seeds none. A body this stub cannot produce would be a
        // failure, not an empty answer.
        const seed = Number(
          new URL(url, 'http://test.local').searchParams.get('seed_query_id'),
        )
        return jsonResponse(
          (predictions.seeded_dirs as { seed_query_id: number }[]).filter(
            (dir) => dir.seed_query_id === seed,
          ),
        )
      }
    if (url.includes('/api/models')) return jsonResponse(modelsPayload())
    if (url.includes('/predictions')) {
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
        model: MODEL,
        variant: 'sif_abtt',
        ...predictions,
      })
    }
    return jsonResponse({})
  })
  vi.stubGlobal('fetch', fetchMock)
}

function Harness() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return <PredictionList />
}

function renderList() {
  return render(
    <AppProvider>
      <SavedDirectoryProvider accountKey="test-account">
        <PredictionProvider>
          {/* App's own composition: the list renders the durable directory
              record and this is the one thing that feeds it (issue #161). */}
          <SavedDirectoryObservations />
          <Harness />
        </PredictionProvider>
      </SavedDirectoryProvider>
    </AppProvider>,
  )
}

beforeEach(() => {
  predictions = { predictions: [modelCard(1, 0.91)], seeded_dirs: [] }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

// --- distinct candidate card ----------------------------------------------

describe('reviewer directory candidates', () => {
  it('renders a distinct card, labelled, attributed and unranked', async () => {
    predictions = {
      predictions: [modelCard(1, 0.44)],
      seeded_dirs: [],
      reviewer_dir_candidates: [reviewerCard(0.62)],
    }
    renderList()

    const card = await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')
    // Distinct from a model card: its own testid, its own accessible name, and
    // the reviewer's label rather than the opaque directory id. The accessible
    // name says outright that it is not ranked (issue #196).
    expect(card.getAttribute('aria-label')).toBe(
      'Reviewer directory Unattested homily, not ranked',
    )
    expect(card.textContent).toContain('Reviewer directory')
    expect(card.textContent).toContain('Unattested homily')
    expect(card.textContent).toContain('Created by Abigail')
    expect(card.textContent).toContain('0.620')
    expect(card.textContent).not.toContain('11')

    // Under its own heading, not in the ranked list.
    expect(screen.getByTestId('reviewer-dirs-heading').textContent).toBe(
      'Directories created by reviewers',
    )

    // The model's own card is still an ordinary prediction card.
    expect(
      screen.getByRole('button', { name: /^Prediction rank 1: candidate-1\./ }),
    ).toBeTruthy()
    expect(screen.queryByTestId('reviewer-dir-card-candidate-1')).toBeNull()
  })

  it('shows no heading and no block when none is offered', async () => {
    predictions = { predictions: [modelCard(1, 0.91)], seeded_dirs: [] }
    renderList()
    await screen.findByRole('button', { name: /^Prediction rank 1: candidate-1\./ })
    expect(screen.queryByTestId('reviewer-dirs-heading')).toBeNull()
  })

  it('keeps every model rank intact beside them', async () => {
    predictions = {
      predictions: Array.from({ length: 10 }, (_, i) => modelCard(i + 1, 0.9 - i / 100)),
      seeded_dirs: [],
      reviewer_dir_candidates: [reviewerCard(0.61)],
    }
    renderList()

    await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')
    // All ten model cards survive, and the reviewer directory displaces none.
    expect(
      screen.getByRole('button', { name: /^Prediction rank 10: candidate-10\./ }),
    ).toBeTruthy()
    expect(
      screen.getByTestId('reviewer-dir-card-reviewer-dir-1').getAttribute('aria-label'),
    ).toContain('not ranked')
  })

  it('arrow keys walk the model ranks only and never reach a reviewer card', async () => {
    predictions = {
      predictions: [modelCard(1, 0.44), modelCard(2, 0.2)],
      seeded_dirs: [],
      reviewer_dir_candidates: [reviewerCard(0.62)],
    }
    renderList()
    await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')

    await userEvent.keyboard('{ArrowDown}{ArrowDown}{ArrowDown}')
    await waitFor(() => {
      expect(
        screen
          .getByRole('button', { name: /^Prediction rank 2: candidate-2\./ })
          .getAttribute('aria-pressed'),
      ).toBe('true')
    })
    expect(
      screen
        .getByTestId('reviewer-dir-card-reviewer-dir-1')
        .getAttribute('aria-pressed'),
    ).toBe('false')
  })

  it('opens a reviewer directory for reading without selecting a rank', async () => {
    predictions = {
      predictions: [modelCard(1, 0.44)],
      seeded_dirs: [],
      reviewer_dir_candidates: [reviewerCard(0.62)],
    }
    renderList()

    const card = await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')
    expect(card.getAttribute('aria-pressed')).toBe('false')
    await userEvent.click(card)
    await waitFor(() => {
      expect(
        screen
          .getByTestId('reviewer-dir-card-reviewer-dir-1')
          .getAttribute('aria-pressed'),
      ).toBe('true')
    })
    // Reading it is not answering with it: the model's rank 1 stays selected,
    // and a document is filed here by naming the directory's CCL key instead.
    expect(
      screen
        .getByRole('button', { name: /^Prediction rank 1: candidate-1\./ })
        .getAttribute('aria-pressed'),
    ).toBe('false')
  })
})

// --- badge lifecycle -------------------------------------------------------

describe('AwaitingMatchBadge', () => {
  const base = {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    seed_query_id: 3,
    member_query_ids: [3],
    created_at: '2026-08-26 00:00:00',
    created_by: 'Abigail',
    model_slug: MODEL,
    variant: 'sif_abtt' as const,
    has_potential_match: false,
  }

  it('renders nothing for a document that seeded no directory', () => {
    const { container } = render(<AwaitingMatchBadge seededDirs={[]} />)
    expect(container.textContent).toBe('')
  })

  it('shows "Awaiting future match" while nothing has matched', () => {
    render(
      <AwaitingMatchBadge
        seededDirs={[{ ...base, status: 'awaiting_match', best_match_score: 0.31 }]}
      />,
    )
    expect(screen.getByTestId('awaiting-match-badge').textContent).toBe(
      'Awaiting future match',
    )
    expect(screen.queryByTestId('matched-dir-badge')).toBeNull()
  })

  it('flips to the matched badge once a query crosses the band', () => {
    render(
      <AwaitingMatchBadge
        seededDirs={[{ ...base, status: 'matched', best_match_score: 0.8 }]}
      />,
    )
    expect(screen.getByTestId('matched-dir-badge').textContent).toBe(
      'New directory matched',
    )
    expect(screen.queryByTestId('awaiting-match-badge')).toBeNull()
  })

  it('marks an unconfirmed above-band neighbour as a lead, not a match', () => {
    // The old behaviour turned this green. On the real corpus that fired for
    // 57-70% of directories at creation, before any human confirmed anything.
    render(
      <AwaitingMatchBadge
        seededDirs={[
          {
            ...base,
            status: 'awaiting_match',
            best_match_score: 0.82,
            has_potential_match: true,
          },
        ]}
      />,
    )
    const badge = screen.getByTestId('awaiting-match-badge')
    expect(badge.textContent).toBe('Awaiting future match · lead')
    expect(badge.getAttribute('title')).toContain('0.82')
    expect(badge.getAttribute('title')).toContain('unconfirmed')
    expect(screen.queryByTestId('matched-dir-badge')).toBeNull()
  })

  it('counts multiple directories in the same state', () => {
    render(
      <AwaitingMatchBadge
        seededDirs={[
          { ...base, status: 'awaiting_match', best_match_score: 0.31 },
          {
            ...base,
            dir_id: 'reviewer-dir-2',
            status: 'awaiting_match',
            best_match_score: 0.2,
          },
        ]}
      />,
    )
    expect(screen.getByTestId('awaiting-match-badge').textContent).toBe(
      'Awaiting future match (2)',
    )
  })

  it('appears in the prediction list for the seed document', async () => {
    predictions = {
      predictions: [modelCard(1, 0.41)],
      // Seeded by the document on screen. The list reads the durable record
      // (issue #161), which keeps directories under the query that actually
      // seeded them, so a row belonging to another document is not badged here.
      seeded_dirs: [
        {
          ...base,
          seed_query_id: QUERY_ID,
          member_query_ids: [QUERY_ID],
          status: 'awaiting_match',
          best_match_score: 0.31,
        },
      ],
    }
    renderList()
    expect((await screen.findByTestId('awaiting-match-badge')).textContent).toBe(
      'Awaiting future match',
    )
  })
})

// --- the retired creation flow (issue #196) --------------------------------

describe('the retired new-directory creation flow', () => {
  it('offers no create control below the no-match band', async () => {
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()
    const callout = await screen.findByTestId('no-match-callout')
    expect(callout.getAttribute('role')).toBe('alert')
    expect(callout.textContent).toContain('Potentially no match')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(callout.querySelector('button')).toBeNull()
  })

  it('offers no create control above it either', async () => {
    predictions = { predictions: [modelCard(1, 0.72)], seeded_dirs: [] }
    renderList()
    await screen.findByRole('button', { name: /^Prediction rank 1: candidate-1\./ })
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(screen.queryByTestId('no-match-callout')).toBeNull()
  })

  it('posts nothing to the directory endpoint from this panel', async () => {
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()
    await screen.findByTestId('no-match-callout')
    expect(posted.filter((p) => p.url.includes('/api/reviewer_dirs'))).toHaveLength(0)
  })

  it('still reports a directory this document already seeds', async () => {
    // Permanent and undeletable: the acknowledgement survives the retirement of
    // the button that made it.
    predictions = {
      predictions: [modelCard(1, 0.41)],
      seeded_dirs: [
        {
          dir_id: 'reviewer-dir-1',
          label: 'Unattested homily',
          status: 'awaiting_match',
          seed_query_id: QUERY_ID,
          member_query_ids: [QUERY_ID],
          created_at: '2026-08-26 00:00:00',
          created_by: 'Abigail',
          model_slug: MODEL,
          variant: 'sif_abtt',
          best_match_score: 0.31,
          has_potential_match: false,
        },
      ],
    }
    renderList()
    const notice = await screen.findByTestId('new-directory-saved')
    expect(notice.textContent).toContain('Unattested homily')
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Abigail',
    )
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })
})
