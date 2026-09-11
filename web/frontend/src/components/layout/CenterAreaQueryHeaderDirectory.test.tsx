import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import SavedDirectoryObservations from '../../contexts/SavedDirectoryObservations'
import { TokenProvider } from '../../contexts/TokenContext'
import { REVIEWER_DIRS_UPDATED_EVENT } from '../../api/reviewerDirs'
import {
  HISTORICAL_SEEDLESS_DIR,
  HISTORICAL_SEED_QUERY_ID,
} from '../../test/fixtures/historicalReviewerDir'
import CenterArea from './CenterArea'

/**
 * The directory badge on the QUERY HEADER, in the composed application
 * (issues #161 + #163).
 *
 * The badge answers "what happened to the directory I started on this
 * document". That is a fact about the document, established by the database and
 * recorded in the durable saved-directory record. It used to be read off
 * `PredictionResponse.seeded_dirs` instead -- a field of the ranking request --
 * which made a permanent grouping behave like part of a fetch: gone while the
 * ranking loaded, gone when it failed, re-asked per model, and showing rows
 * this account had never had admitted for it.
 *
 * Every case below drives the real tree through `fetch`. Nothing writes to the
 * store directly, because a state the application cannot actually reach is not
 * a state worth asserting about.
 *
 * The composition is App's own: `SavedDirectoryProvider` above
 * `PredictionProvider`, with `SavedDirectoryObservations` as the single
 * admission point. `accountKey` is a prop so an account change can be exercised
 * as a re-render of the same tree, which is what a sign-in actually does to the
 * providers underneath it.
 */

const QUERY_A = HISTORICAL_SEED_QUERY_ID
const QUERY_B = 8
const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'bowphs_PhilTa'

interface PredictionPayload {
  predictions?: unknown[]
  seeded_dirs?: unknown[]
  httpStatus?: number
}

let predictionsFor: Record<number, PredictionPayload>
let queuedPredictions: Record<number, PredictionPayload[]>
let storedDirs: Record<string, unknown>[]
let dirLookupStatus: number
let dirGets: number

function dirFixture(overrides: Record<string, unknown> = {}) {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    status: 'awaiting_match',
    seed_query_id: QUERY_A,
    member_query_ids: [QUERY_A],
    created_at: '2026-08-26 00:00:00',
    created_by: 'Abigail',
    model_slug: MODEL_A,
    variant: 'sif_abtt',
    best_match_score: 0.31,
    has_potential_match: false,
    ...overrides,
  }
}

function modelEntry(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 6,
    pooling: 'mean',
    prediction_count: 10,
    available_variants: ['sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.45, verify: 0.7 },
    has_qq_matrix: true,
  }
}

function modelCard(rank: number, score: number) {
  return {
    rank,
    dir_name: `dir-${rank}`,
    score,
    dir_files: [`dir-${rank}-a.txt`],
    preview_text: 'incipit sermo',
    candidate_files: [{ filename: `dir-${rank}-a.txt`, text: 'incipit sermo' }],
    source: 'model',
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => 'application/json' },
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response
}

function payloadFor(queryId: number): PredictionPayload {
  const next = queuedPredictions[queryId]?.shift()
  return next ?? predictionsFor[queryId] ?? { predictions: [modelCard(1, 0.9)] }
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const path = new URL(url, 'http://localhost').pathname

      if (path === '/api/reviewer_dirs') {
        dirGets += 1
        if (dirLookupStatus !== 200) {
          return jsonResponse({ detail: 'unavailable' }, dirLookupStatus)
        }
        const seed = new URL(url, 'http://localhost').searchParams.get('seed_query_id')
        return jsonResponse(
          seed === null
            ? storedDirs
            : storedDirs.filter((dir) => dir.seed_query_id === Number(seed)),
        )
      }
      if (path === '/api/models') {
        return jsonResponse([modelEntry(MODEL_A, 'LaTa'), modelEntry(MODEL_B, 'PhilTa')])
      }
      if (path.endsWith('/predictions')) {
        const queryId = Number(path.split('/')[3])
        const model = new URL(url, 'http://localhost').searchParams.get('model') ?? ''
        const payload = payloadFor(queryId)
        if (payload.httpStatus && payload.httpStatus !== 200) {
          return jsonResponse(
            { error: { code: 'internal_error', message: 'Ranking store unavailable' } },
            payload.httpStatus,
          )
        }
        return jsonResponse({
          file_id: queryId,
          filename: `query-${queryId}.txt`,
          model,
          variant: 'sif_abtt',
          status: 'ok',
          predictions: payload.predictions ?? [],
          seeded_dirs: payload.seeded_dirs ?? [],
        })
      }
      if (/^\/api\/query\/\d+$/.test(path)) {
        const queryId = Number(path.split('/')[3])
        return jsonResponse({
          file_id: queryId,
          filename: `query-${queryId}.txt`,
          text: 'uerba domini',
          tokens: [{ text: 'uerba', index: 0, category: 'content' }],
        })
      }
      if (path.includes('token_map')) {
        return jsonResponse({ error: { message: 'no artifact' } }, 404)
      }
      return jsonResponse(null)
    }),
  )
}

function Selection({ queryId, model }: { queryId: number; model: string }) {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(queryId)
    setActiveModel(model)
  }, [queryId, model, setActiveQueryId, setActiveModel])
  return null
}

function Header({
  queryId = QUERY_A,
  model = MODEL_A,
  accountKey = 2,
}: {
  queryId?: number
  model?: string
  accountKey?: number | null
}) {
  return (
    <AppProvider>
      <TokenProvider>
        <SavedDirectoryProvider accountKey={accountKey}>
          <PredictionProvider>
            <SavedDirectoryObservations />
            <Selection queryId={queryId} model={model} />
            <CenterArea />
          </PredictionProvider>
        </SavedDirectoryProvider>
      </TokenProvider>
    </AppProvider>
  )
}

function awaitingBadge(): HTMLElement | null {
  return screen.queryByTestId('awaiting-match-badge')
}

function matchedBadge(): HTMLElement | null {
  return screen.queryByTestId('matched-dir-badge')
}

/** The document really is on screen, so "no badge" is not "nothing rendered". */
async function headerReady(queryId: number): Promise<void> {
  await screen.findByText(`query-${queryId}.txt`)
}

/**
 * CenterArea's own statement that the ranking failed.
 *
 * The server's message ("Ranking store unavailable") is rendered by
 * `PredictionList`, which this screen does not mount; asserting on it here
 * would be asserting on another component's copy. This pane says the same
 * thing in its own words (issue #156), and that is the observable proof the
 * failure has actually reached the screen the badge is on.
 */
async function rankingFailedHere(): Promise<void> {
  await waitFor(() => {
    expect(screen.getByTestId('candidate-evidence-note').textContent).toContain(
      'did not load',
    )
  })
}

/** True while this pane is not reporting a failed ranking. */
function rankingLooksSettled(): boolean {
  const note = screen.queryByTestId('candidate-evidence-note')
  return !(note?.textContent ?? '').includes('did not load')
}

beforeEach(() => {
  predictionsFor = {}
  queuedPredictions = {}
  storedDirs = []
  dirLookupStatus = 200
  dirGets = 0
  localStorage.clear()
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('the query header states what is stored, not what the last request carried', () => {
  it('shows the stored status once this account has an answer for the document', async () => {
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [dirFixture()],
    }
    render(<Header />)

    await headerReady(QUERY_A)
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())
    expect(awaitingBadge()?.textContent).toBe('Awaiting future match')
    expect(awaitingBadge()?.getAttribute('title')).toContain('Unattested homily')
  })

  it('keeps the badge through a refresh whose ranking fails', async () => {
    // The grouping is permanent. A ranking that 500s says nothing about it, and
    // taking the badge off screen would report a fetch failure as the loss of a
    // directory the reviewer cannot recreate.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [dirFixture()],
    }
    render(<Header />)
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())

    // Everything that could re-establish the fact is now broken: the next
    // ranking fails and the directory endpoint fails with it. Whatever is on
    // screen afterwards is on screen because the record is durable.
    queuedPredictions[QUERY_A] = [{ httpStatus: 500 }, { httpStatus: 500 }]
    dirLookupStatus = 500
    const getsBefore = dirGets
    await act(async () => {
      window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
    })

    await rankingFailedHere()
    expect(awaitingBadge()).not.toBeNull()
    expect(awaitingBadge()?.textContent).toBe('Awaiting future match')
    // And it did not quietly re-ask: a settled identity is not looked up again.
    expect(dirGets).toBe(getsBefore)
  })

  it('keeps the badge across a model switch, without re-asking', async () => {
    // Identity does not depend on a model. The old read did: `seeded_dirs`
    // arrives per ranking, so the badge blinked out for the width of the second
    // model's request and came back saying the same thing.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [dirFixture()],
    }
    const { rerender } = render(<Header model={MODEL_A} />)
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())
    const getsBefore = dirGets

    // The second model's ranking never answers with the directory at all, which
    // is exactly the case the old read got wrong.
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    rerender(<Header model={MODEL_B} />)

    await waitFor(() => {
      expect(
        (screen.getAllByTestId('awaiting-match-badge')[0] as HTMLElement).textContent,
      ).toBe('Awaiting future match')
    })
    expect(dirGets).toBe(getsBefore)
  })

  it('shows a stored grouping even though no ranking ever arrives', async () => {
    // Cold recovery. After a reload the record starts unknown; if the only way
    // to learn about a directory were a successful ranking, a reviewer whose
    // model is down would be told nothing about the permanent thing they
    // created -- with the directory endpoint healthy the whole time.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { httpStatus: 500 }
    render(<Header />)

    await headerReady(QUERY_A)
    await rankingFailedHere()
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())
    expect(awaitingBadge()?.textContent).toBe('Awaiting future match')
    expect(dirGets).toBeGreaterThan(0)
  })

  it('says nothing about a document whose groupings are not established', async () => {
    // Absence of evidence. The lookup fails and no ranking carries anything, so
    // the honest header is a bare one: claiming "no directory" here is how a
    // reviewer gets invited to create a duplicate.
    dirLookupStatus = 500
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    render(<Header />)

    await headerReady(QUERY_A)
    await waitFor(() => expect(dirGets).toBeGreaterThan(0))
    expect(awaitingBadge()).toBeNull()
    expect(matchedBadge()).toBeNull()
  })
})

describe('the header speaks for the document and the account on screen', () => {
  it('drops the previous document’s badge the moment another is selected', async () => {
    // Keyed on the document being rendered, not on the ranking's query id,
    // which lags a selection by a request. A badge sourced from the lagging key
    // would caption the new document with the previous one's groupings.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [dirFixture()],
    }
    predictionsFor[QUERY_B] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    const { rerender } = render(<Header queryId={QUERY_A} />)
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())

    rerender(<Header queryId={QUERY_B} />)

    // Not "eventually": the first commit for the new document already carries
    // no claim about it.
    expect(awaitingBadge()).toBeNull()
    await headerReady(QUERY_B)
    expect(awaitingBadge()).toBeNull()
    expect(matchedBadge()).toBeNull()
  })

  it('does not show one account the groupings established for another', async () => {
    // Reviewer directories are global, so the rows may well come back. What may
    // not happen is this account being shown them on the strength of the
    // previous session's answer.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [dirFixture()],
    }
    const { rerender } = render(<Header accountKey={2} />)
    await waitFor(() => expect(awaitingBadge()).not.toBeNull())

    // The new session can establish nothing: both endpoints are down.
    dirLookupStatus = 500
    queuedPredictions[QUERY_A] = [{ httpStatus: 500 }, { httpStatus: 500 }]
    rerender(<Header accountKey={3} />)

    expect(awaitingBadge()).toBeNull()
    await rankingFailedHere()
    expect(awaitingBadge()).toBeNull()
  })

  it('refuses a row whose fields this client filled in on the server’s behalf', async () => {
    // The ranking validator tolerates an older backend omitting `created_by`
    // and supplies `''`. That substitute is type-valid and indistinguishable
    // afterwards, so it is named -- and a named substitute is not a stored
    // fact. The badge orders and attributes groupings from these fields, so it
    // gets the row only when the server really sent it.
    dirLookupStatus = 500
    const incomplete = dirFixture()
    delete (incomplete as Record<string, unknown>).created_by
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.9)],
      seeded_dirs: [incomplete],
    }
    render(<Header />)

    await headerReady(QUERY_A)
    await waitFor(() => expect(dirGets).toBeGreaterThan(0))
    expect(awaitingBadge()).toBeNull()
    expect(matchedBadge()).toBeNull()

    // The row itself is still a legible candidate: refusing it as durable
    // evidence is not the same as failing the response.
    expect(rankingLooksSettled()).toBe(true)
  })
})

describe('a grouping this document is not filed in does not claim that it is', () => {
  it('marks a stored grouping that does not list this document as a member', async () => {
    // A real preserved pre-#160 row: the directory is permanent, the membership
    // row never landed, and nothing here can add it. Hiding the grouping would
    // leave the reviewer unable to explain why Create is refused; implying the
    // membership would invent the one thing `matched` is derived from.
    storedDirs = [{ ...HISTORICAL_SEEDLESS_DIR }]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    render(<Header />)

    await waitFor(() => expect(awaitingBadge()).not.toBeNull())
    const badge = awaitingBadge() as HTMLElement
    expect(badge.textContent).toBe('Awaiting future match · not filed')
    expect(badge.getAttribute('title')).toContain('Historical partial grouping')
    expect(badge.getAttribute('title')).toContain(
      'does not list this document among its members',
    )
  })

  it('does not let such a grouping turn a second witness into this document’s match', async () => {
    // Two OTHER documents were filed into it, so the server derives `matched`.
    // True of the directory; not a statement about this document, which the
    // stored record does not list at all. Unmarked, the green chip on this
    // document's own header reads as "your document matched".
    storedDirs = [
      { ...HISTORICAL_SEEDLESS_DIR, status: 'matched', member_query_ids: [41, 42] },
    ]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    render(<Header />)

    await waitFor(() => expect(matchedBadge()).not.toBeNull())
    const badge = matchedBadge() as HTMLElement
    expect(badge.textContent).toBe('New directory matched · not filed')
    expect(badge.getAttribute('title')).toContain(
      'does not list this document among its members',
    )
  })

  it('leaves an ordinary filed grouping unmarked', async () => {
    // The guard is about the rare preserved row, and must not put a caveat on
    // the overwhelmingly common complete one.
    storedDirs = [dirFixture({ status: 'matched', member_query_ids: [QUERY_A, 42] })]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.9)], seeded_dirs: [] }
    render(<Header />)

    await waitFor(() => expect(matchedBadge()).not.toBeNull())
    const badge = matchedBadge() as HTMLElement
    expect(badge.textContent).toBe('New directory matched')
    expect(badge.getAttribute('title')).not.toContain('does not list this document')
  })
})
