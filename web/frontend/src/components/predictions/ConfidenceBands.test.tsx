import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import NoMatchCallout from './NoMatchCallout'
import PredictionList from './PredictionList'

const MODEL = 'google_mt5-base'
const QUERY_ID = 11

/** Ranked hits whose top score decides the band under test. */
function predictionsPayload(scores: number[], fileId = QUERY_ID) {
  return {
    file_id: fileId,
    filename: `query-${fileId}.txt`,
    model: MODEL,
    variant: 'sif_abtt',
    predictions: scores.map((score, index) => ({
      rank: index + 1,
      dir_name: `CSAR.347.${index + 1}`,
      score,
      dir_files: ['a.txt'],
      preview_text: 'preview',
      candidate_files: [{ filename: 'a.txt', text: 'candidate text' }],
      source: 'model' as const,
    })),
    seeded_dirs: seededDirs,
  }
}

let scores: number[] = [0.9]
/** Reviewer directories seeded by the query under test. */
let seededDirs: unknown[] = []
/** Status the stubbed POST /api/reviewer_dirs answers with. */
let reviewerDirStatus = 201
let reviewerDirPosts: { url: string; body: unknown }[] = []
/** When set, the stubbed POST hangs until `releaseReviewerDir()` is called. */
let deferReviewerDir = false
let releaseReviewerDir: () => void = () => {}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  reviewerDirPosts = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/reviewer_dirs')) {
        reviewerDirPosts.push({
          url,
          body: init?.body ? JSON.parse(String(init.body)) : null,
        })
        if (deferReviewerDir) {
          await new Promise<void>((resolve) => {
            releaseReviewerDir = resolve
          })
        }
        if (reviewerDirStatus === 201) {
          return jsonResponse(
            { dir_id: 'rev-42', label: 'New directory 42', status: 'awaiting_match' },
            201,
          )
        }
        return jsonResponse({ detail: 'Not Found' }, reviewerDirStatus)
      }
      if (url.includes('/api/models')) {
        return jsonResponse([
          {
            slug: MODEL,
            display_name: 'mT5-base',
            layer: 4,
            pooling: 'mean',
            prediction_count: 2238,
            available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
            default_variant: 'sif_abtt',
            // Thresholds are served, not hardcoded in the bundle (#95).
            confidence_bands: { no_match: 0.5, verify: 0.7 },
            supports_reviewer_dirs: true,
          },
        ])
      }
      if (url.includes('/predictions')) {
        const fileId = Number(url.match(/\/api\/query\/(\d+)\/predictions/)?.[1] ?? QUERY_ID)
        return jsonResponse(predictionsPayload(scores, fileId))
      }
      return jsonResponse({})
    }),
  )
}

const NEXT_QUERY_ID = 12

/** Lets a test navigate to another query mid-request, as a reviewer would. */
function NavigableHarness() {
  const { activeQueryId, setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return (
    <>
      <button type="button" onClick={() => setActiveQueryId(NEXT_QUERY_ID)}>
        go to next query
      </button>
      <span data-testid="active-query">{activeQueryId}</span>
      <PredictionList />
    </>
  )
}

function Harness() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return <PredictionList />
}

function renderList(topScores: number[]) {
  scores = topScores
  return render(
    <AppProvider>
      <Harness />
    </AppProvider>,
  )
}

beforeEach(() => {
  scores = [0.9]
  seededDirs = []
  reviewerDirStatus = 201
  deferReviewerDir = false
  releaseReviewerDir = () => {}
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('confidence bands on the prediction list', () => {
  it('flags the no-match band in red at 0.49', async () => {
    renderList([0.49, 0.2])
    const callout = await screen.findByTestId('no-match-callout')
    expect(callout.textContent).toContain('Potentially no match')
    expect(callout.getAttribute('role')).toBe('alert')
    expect(screen.queryByTestId('careful-review-note')).toBeNull()
    expect(screen.queryByTestId('likely-match-note')).toBeNull()
    // The top card carries the band too, not just the banner.
    expect(screen.getByTestId('band-chip-1').getAttribute('class')).toContain('incorrect')
  })

  it('shows the careful-review note at the 0.5 boundary', async () => {
    renderList([0.5])
    expect(await screen.findByTestId('careful-review-note')).toBeTruthy()
    expect(screen.queryByTestId('no-match-callout')).toBeNull()
    expect(screen.queryByTestId('likely-match-note')).toBeNull()
  })

  it('still shows the careful-review note at 0.69', async () => {
    renderList([0.69])
    expect(await screen.findByTestId('careful-review-note')).toBeTruthy()
    expect(screen.queryByTestId('no-match-callout')).toBeNull()
  })

  it('switches to verification framing at 0.7', async () => {
    renderList([0.7])
    expect((await screen.findByTestId('likely-match-note')).textContent).toContain(
      'Likely match - verify',
    )
    expect(screen.queryByTestId('no-match-callout')).toBeNull()
    expect(screen.queryByTestId('careful-review-note')).toBeNull()
  })

  it('keeps verification framing at 0.71', async () => {
    renderList([0.71])
    expect(await screen.findByTestId('likely-match-note')).toBeTruthy()
    expect(screen.queryByTestId('careful-review-note')).toBeNull()
  })

  it('bands each card independently of the list-level banner', async () => {
    renderList([0.8, 0.6, 0.3])
    await screen.findByTestId('likely-match-note')
    expect(screen.getByTestId('band-chip-1').textContent).toBe('Likely match - verify')
    expect(screen.getByTestId('band-chip-2').textContent).toBe('Review this match carefully')
    expect(screen.getByTestId('band-chip-3').textContent).toBe('Potentially no match')
  })
})

describe('new-directory CTA', () => {
  it('is hidden above the no-match threshold', async () => {
    renderList([0.5])
    await screen.findByTestId('careful-review-note')
    expect(
      screen.queryByRole('button', { name: 'New directory / New file' }),
    ).toBeNull()
  })

  it('posts the D2 contract and reports the created directory', async () => {
    // Adapted for #95: the CTA opens a one-field naming form rather than
    // posting on the first click, because a directory's label is permanent
    // (both tables are append-only, there is no rename). The field is
    // pre-filled, so accepting the default is one more click and no typing.
    const user = userEvent.setup()
    renderList([0.31])
    const cta = await screen.findByRole('button', { name: 'New directory / New file' })

    await user.click(cta)
    await user.click(await screen.findByTestId('new-directory-submit'))

    await waitFor(() => {
      expect(reviewerDirPosts).toHaveLength(1)
    })
    expect(reviewerDirPosts[0].url).toContain('/api/reviewer_dirs')
    expect(reviewerDirPosts[0].body).toEqual({
      query_file_id: QUERY_ID,
      label: 'New directory from query-11',
      model_slug: MODEL,
    })
    expect((await screen.findByTestId('new-directory-created')).textContent).toContain(
      'New directory 42',
    )
  })

  it('surfaces a server failure to the reviewer', async () => {
    reviewerDirStatus = 500
    const user = userEvent.setup()
    renderList([0.1])

    await user.click(
      await screen.findByRole('button', { name: 'New directory / New file' }),
    )
    await user.click(await screen.findByTestId('new-directory-submit'))

    expect(await screen.findByTestId('new-directory-error')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-created')).toBeNull()
  })

  it('refuses a second directory on a document that already seeds one', async () => {
    // The backend answers that with 409, so the CTA is not offered at all.
    seededDirs = [
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
    ]
    renderList([0.2])

    await screen.findByTestId('no-match-callout')
    expect(screen.getByTestId('no-match-already-seeded')).toBeTruthy()
    expect(
      screen.queryByRole('button', { name: 'New directory / New file' }),
    ).toBeNull()
  })
})

describe('CTA staleness (reviewer navigates mid-request)', () => {
  it('does not paint a created directory onto the next query', async () => {
    // The reviewer's repro: click the CTA on query 11, move to query 12 before
    // the POST resolves. The late 201 belongs to a fragment that is no longer
    // on screen and must not be shown as query 12's outcome.
    deferReviewerDir = true
    const user = userEvent.setup()
    scores = [0.2]
    render(
      <AppProvider>
        <NavigableHarness />
      </AppProvider>,
    )

    await user.click(
      await screen.findByRole('button', { name: 'New directory / New file' }),
    )
    await user.click(await screen.findByTestId('new-directory-submit'))
    await waitFor(() => {
      expect(reviewerDirPosts).toHaveLength(1)
    })
    expect(reviewerDirPosts[0].body).toMatchObject({ query_file_id: QUERY_ID })

    await user.click(screen.getByRole('button', { name: 'go to next query' }))
    await waitFor(() => {
      expect(screen.getByTestId('active-query').textContent).toBe(String(NEXT_QUERY_ID))
    })

    // ...and only now does query 11's request come back.
    await act(async () => {
      releaseReviewerDir()
    })

    expect(screen.queryByTestId('new-directory-created')).toBeNull()
    // Query 12 shows a fresh, clickable CTA rather than a stuck "Creating...".
    expect(screen.getByRole('button', { name: 'New directory / New file' })).toBeTruthy()
  })

  it('drops a late result even when the component is reused, not remounted', async () => {
    // Directly exercises the guard inside the component, independently of the
    // `key` in PredictionList: here the same instance is handed a new query.
    deferReviewerDir = true
    const user = userEvent.setup()
    const { rerender } = render(
      <NoMatchCallout
        queryFileId={QUERY_ID}
        topScore={0.2}
        topK={10}
        model={MODEL}
        alreadySeeded={false}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'New directory / New file' }))
    await user.click(screen.getByTestId('new-directory-submit'))
    await waitFor(() => {
      expect(reviewerDirPosts).toHaveLength(1)
    })

    rerender(
      <NoMatchCallout
        queryFileId={NEXT_QUERY_ID}
        topScore={0.3}
        topK={10}
        model={MODEL}
        alreadySeeded={false}
      />,
    )
    await act(async () => {
      releaseReviewerDir()
    })

    expect(screen.queryByTestId('new-directory-created')).toBeNull()
    expect(screen.getByRole('button', { name: 'New directory / New file' })).toBeTruthy()
  })
})
