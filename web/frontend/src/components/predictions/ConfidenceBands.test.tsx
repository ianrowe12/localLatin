import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import NoMatchCallout from './NoMatchCallout'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
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
        // The seed lookup (issue #161) shares this path with creation, so the
        // two are separated by method: only a POST is a creation attempt, and
        // only a POST is deferred by `deferReviewerDir`.
        if (init?.method !== 'POST') {
          return jsonResponse(
            seededDirs.filter(
              (dir) =>
                (dir as { seed_query_id?: number }).seed_query_id ===
                Number(
                  new URL(url, 'http://test.local').searchParams.get(
                    'seed_query_id',
                  ),
                ),
            ),
          )
        }
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
            {
              dir_id: 'rev-42',
              label: 'New directory 42',
              status: 'awaiting_match',
              seed_query_id: QUERY_ID,
              member_query_ids: [QUERY_ID],
              created_at: '2026-08-26 00:00:00',
              created_by: 'Abigail',
              best_match_score: null,
              has_potential_match: false,
            },
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
      <SavedDirectoryProvider accountKey="test-account">
        <PredictionProvider>
          <Harness />
        </PredictionProvider>
      </SavedDirectoryProvider>
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

describe('the retired new-directory call to action (issue #196)', () => {
  it('offers no way to create a directory at any band', async () => {
    for (const band of [[0.2], [0.6], [0.9]]) {
      const { unmount } = renderList(band)
      await screen.findByTestId('band-chip-1')
      expect(screen.queryByTestId('new-directory-cta')).toBeNull()
      expect(screen.queryByTestId('new-directory-form')).toBeNull()
      expect(screen.queryByTestId('new-directory-submit')).toBeNull()
      expect(screen.queryByTestId('directory-creation-guidance')).toBeNull()
      expect(
        screen.queryByRole('button', { name: 'New directory / New file' }),
      ).toBeNull()
      expect(screen.queryByRole('button', { name: 'Start a new directory' })).toBeNull()
      unmount()
    }
  })

  it('keeps the red low-confidence notice as a hint, with no action in it', async () => {
    renderList([0.49, 0.2])
    const callout = await screen.findByTestId('no-match-callout')
    expect(callout.getAttribute('role')).toBe('alert')
    expect(callout.textContent).toContain('Potentially no match')
    expect(callout.textContent).toContain('Best similarity is 0.490')
    // The caveat stays: a low score is not evidence the CCL lacks the source.
    expect(screen.getByTestId('no-match-caveat').textContent).toContain('CCL')
    expect(callout.querySelector('button')).toBeNull()
  })

  it('never posts to the directory endpoint from the prediction panel', async () => {
    renderList([0.2])
    await screen.findByTestId('no-match-callout')
    expect(reviewerDirPosts).toHaveLength(0)
  })
})
