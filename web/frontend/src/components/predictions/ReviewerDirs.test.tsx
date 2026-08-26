import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import AwaitingMatchBadge from './AwaitingMatchBadge'
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

function reviewerCard(rank: number, score: number) {
  return {
    rank,
    dir_name: 'reviewer-dir-1',
    score,
    dir_files: ['query-3.txt'],
    preview_text: 'seed text',
    candidate_files: [{ filename: 'query-3.txt', text: 'seed text' }],
    source: 'reviewer' as const,
    label: 'Unattested homily',
    created_by: 'Abigail',
    seed_query_id: 3,
  }
}

/** Mutable per test, so a test can decide what the list currently contains. */
let predictions: {
  predictions: unknown[]
  seeded_dirs: unknown[]
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
      <Harness />
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
  it('renders a distinct card, labelled and attributed', async () => {
    predictions = {
      predictions: [modelCard(1, 0.44), reviewerCard(11, 0.62)],
      seeded_dirs: [],
    }
    renderList()

    const card = await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')
    // Distinct from a model card: its own testid, its own accessible name, and
    // the reviewer's label rather than the opaque directory id.
    expect(card.getAttribute('aria-label')).toBe(
      'Reviewer directory Unattested homily, rank 11',
    )
    expect(card.textContent).toContain('Reviewer directory')
    expect(card.textContent).toContain('Unattested homily')
    expect(card.textContent).toContain('Created by Abigail')
    expect(card.textContent).toContain('0.620')

    // The model's own card is still an ordinary prediction card.
    expect(
      screen.getByRole('button', { name: 'Prediction rank 1: candidate-1' }),
    ).toBeTruthy()
    expect(screen.queryByTestId('reviewer-dir-card-candidate-1')).toBeNull()
  })

  it('keeps model ranks intact and appends reviewer candidates after them', async () => {
    predictions = {
      predictions: [
        ...Array.from({ length: 10 }, (_, i) => modelCard(i + 1, 0.9 - i / 100)),
        reviewerCard(11, 0.61),
      ],
      seeded_dirs: [],
    }
    renderList()

    await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')
    // All ten model cards survive the ten-item cap, and the reviewer card is
    // additional rather than displacing rank 10.
    expect(
      screen.getByRole('button', { name: 'Prediction rank 10: candidate-10' }),
    ).toBeTruthy()
    expect(
      screen.getByTestId('reviewer-dir-card-reviewer-dir-1').getAttribute('aria-label'),
    ).toContain('rank 11')
  })

  it('arrow keys step across the gap between model and reviewer ranks', async () => {
    // Ranks are 1, 2 then 11: reviewer cards are anchored, not contiguous, so
    // stepping numerically would land on rank 3, which has no card.
    predictions = {
      predictions: [modelCard(1, 0.44), modelCard(2, 0.2), reviewerCard(11, 0.62)],
      seeded_dirs: [],
    }
    renderList()
    await screen.findByTestId('reviewer-dir-card-reviewer-dir-1')

    await userEvent.keyboard('{ArrowDown}{ArrowDown}')
    await waitFor(() => {
      expect(
        screen
          .getByTestId('reviewer-dir-card-reviewer-dir-1')
          .getAttribute('aria-pressed'),
      ).toBe('true')
    })

    await userEvent.keyboard('{ArrowUp}')
    await waitFor(() => {
      expect(
        screen
          .getByRole('button', { name: 'Prediction rank 2: candidate-2' })
          .getAttribute('aria-pressed'),
      ).toBe('true')
    })
  })

  it('selects the reviewer card on click', async () => {
    predictions = {
      predictions: [modelCard(1, 0.44), reviewerCard(11, 0.62)],
      seeded_dirs: [],
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
      seeded_dirs: [{ ...base, status: 'awaiting_match', best_match_score: 0.31 }],
    }
    renderList()
    expect((await screen.findByTestId('awaiting-match-badge')).textContent).toBe(
      'Awaiting future match',
    )
  })
})

// --- creation flow ---------------------------------------------------------

describe('new-directory creation flow', () => {
  it('is emphasised when the best labelled candidate is below the no-match band', async () => {
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()
    const cta = await screen.findByTestId('new-directory-cta')
    expect(cta.textContent).toBe('None of these — start a new directory')
  })

  it('is hidden once this document already seeds a directory', async () => {
    // The backend answers a second create on the same seed with 409, so the
    // button must not be offered in the first place.
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
    await screen.findByTestId('awaiting-match-badge')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })

  it('is a quiet escape hatch when the top candidate is above the band', async () => {
    predictions = { predictions: [modelCard(1, 0.72)], seeded_dirs: [] }
    renderList()
    const cta = await screen.findByTestId('new-directory-cta')
    expect(cta.textContent).toBe('Start a new directory')
  })

  it('posts the contract body and confirms the result', async () => {
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()

    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    await userEvent.type(
      screen.getByLabelText('Name the new directory'),
      'Unattested homily',
    )
    await userEvent.click(screen.getByTestId('new-directory-submit'))

    await waitFor(() => {
      expect(screen.getByTestId('new-directory-created')).toBeTruthy()
    })
    const created = posted.find((p) => p.url.includes('/api/reviewer_dirs'))
    expect(created?.body).toEqual({
      query_file_id: QUERY_ID,
      label: 'Unattested homily',
      model_slug: MODEL,
    })
    expect(screen.getByTestId('new-directory-created').textContent).toContain(
      'Unattested homily',
    )
  })

  it('omits the label when the reviewer leaves it blank, letting the backend default it', async () => {
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()

    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    await userEvent.click(screen.getByTestId('new-directory-submit'))

    await waitFor(() => {
      expect(posted.some((p) => p.url.includes('/api/reviewer_dirs'))).toBe(true)
    })
    const created = posted.find((p) => p.url.includes('/api/reviewer_dirs'))
    expect(created?.body).toEqual({
      query_file_id: QUERY_ID,
      model_slug: MODEL,
    })
  })

  it('surfaces a rejected creation instead of pretending it worked', async () => {
    installFetch(() =>
      jsonResponse({ detail: 'Authentication required' }, 401),
    )
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
    renderList()

    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    await userEvent.click(screen.getByTestId('new-directory-submit'))

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toBe('Authentication required')
    expect(screen.queryByTestId('new-directory-created')).toBeNull()
  })
})
