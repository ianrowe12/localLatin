import { useEffect, type ReactNode } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App, { AccountScopedSavedDirectories } from '../../App'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider, useReviewer } from '../../contexts/ReviewerContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import SavedDirectoryObservations from '../../contexts/SavedDirectoryObservations'
import { TokenProvider } from '../../contexts/TokenContext'
import { REVIEWER_DIRS_UPDATED_EVENT } from '../../api/reviewerDirs'
import PredictionList from './PredictionList'

/**
 * The durable saved-directory record, in the composed application.
 *
 * This is what survives of `DurableAcknowledgement.integration.test.tsx` after
 * issue #196 retired the creation flow it was mostly about. Every case below
 * concerns behaviour that still ships through `SavedDirectoryNotice`, and the
 * reason to keep it at THIS level is unchanged: `savedDirectoryStore.test.ts`
 * proves the store's own rules, and these prove the record is reachable from
 * the real tree -- mounted above every branch of the prediction list, scoped to
 * the signed-in account, fed only by issue #156's current request, and never
 * read straight off a raw response.
 *
 * What is gone with the red button: creating, naming, the pending and recovery
 * paths of a write, and the cases about a half-finished POST. Nothing in the
 * prediction panel writes any more.
 */

const QUERY_A = 7
const QUERY_B = 8
const MODEL_A = 'bowphs_LaTa'

interface PredictionPayload {
  predictions?: unknown[]
  seeded_dirs?: unknown[]
  status?: string | null
  httpStatus?: number
}

let predictionsFor: Record<number, PredictionPayload>
let queuedPredictions: Record<number, PredictionPayload[]>
let storedDirs: Record<string, unknown>[]
let dirLookupStatus: number
let predictionGets: number
let dirGets: number
let signedInAccount: number

function accountFixture(id: number) {
  return {
    id,
    username: id === 2 ? 'bob' : 'carol',
    display_name: id === 2 ? 'Bob Bibliothecarius' : 'Carol Cantrix',
    role: 'reviewer',
    approval_status: 'approved',
    must_change_password: false,
  }
}

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
    supports_reviewer_dirs: true,
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
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function payloadFor(queryId: number): PredictionPayload {
  const next = queuedPredictions[queryId]?.shift()
  return next ?? predictionsFor[queryId] ?? { predictions: [modelCard(1, 0.9)] }
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      const path = new URL(url, 'http://localhost').pathname

      if (path === '/api/reviewer_dirs' && method === 'POST') {
        // Nothing in this tree may write a directory any more. A POST here is a
        // regression, not a fixture case.
        throw new Error('the prediction panel must not create directories')
      }
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
      if (path === '/api/auth/me' || path === '/api/auth/signin') {
        return jsonResponse(accountFixture(signedInAccount))
      }
      if (path === '/api/queries/next') return jsonResponse({ file_id: QUERY_A })
      if (path === '/api/models') {
        return jsonResponse([modelEntry(MODEL_A, 'LaTa')])
      }
      if (path.endsWith('/predictions')) {
        predictionGets += 1
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
          status: payload.status === undefined ? 'ok' : payload.status,
          predictions: payload.predictions ?? [],
          seeded_dirs: payload.seeded_dirs ?? [],
          reviewer_dir_candidates: [],
        })
      }
      if (/^\/api\/query\/\d+$/.test(path)) {
        return jsonResponse({
          file_id: Number(path.split('/')[3]),
          filename: 'query.txt',
          text: 'uerba domini',
          tokens: [{ text: 'uerba', index: 0, category: 'content' }],
        })
      }
      if (path.includes('/api/feedback/latest')) return jsonResponse(null)
      if (path.includes('token_map')) {
        return jsonResponse({ error: { message: 'no artifact' } }, 404)
      }
      if (path === '/api/queries') {
        return jsonResponse({
          items: [
            {
              file_id: QUERY_A,
              filename: 'query-7.txt',
              text_preview: 'uerba domini',
              review_status: 'unreviewed',
              review_count: 0,
            },
          ],
          total: 1,
          page: 1,
          page_size: 50,
          has_more: false,
        })
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

/** App.tsx's own provider order around the prediction list. */
function Review({
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
      <ReviewerProvider>
        <TokenProvider>
          <SavedDirectoryProvider accountKey={accountKey}>
            <PredictionProvider>
              <SavedDirectoryObservations />
              <Selection queryId={queryId} model={model} />
              <PredictionList />
            </PredictionProvider>
          </SavedDirectoryProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}

function SignInAsOtherAccount() {
  const { signIn } = useReviewer()
  return (
    <button
      type="button"
      data-testid="switch-account"
      onClick={() => {
        signedInAccount = 3
        void signIn({ username: 'carol', password: 'irrelevant' })
      }}
    >
      switch account
    </button>
  )
}

function AuthGate({ children }: { children: ReactNode }) {
  const { loading } = useReviewer()
  return loading ? null : <>{children}</>
}

function ComposedReview() {
  return (
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <AccountScopedSavedDirectories>
            <PredictionProvider>
              <SavedDirectoryObservations />
              <AuthGate>
                <Selection queryId={QUERY_A} model={MODEL_A} />
                <SignInAsOtherAccount />
                <PredictionList />
              </AuthGate>
            </PredictionProvider>
          </AccountScopedSavedDirectories>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}

beforeEach(() => {
  predictionsFor = {}
  queuedPredictions = {}
  storedDirs = []
  dirLookupStatus = 200
  predictionGets = 0
  dirGets = 0
  signedInAccount = 2
  window.localStorage.clear()
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('the provider is mounted in the real application', () => {
  it('renders the acknowledgement inside App without a fallback store', async () => {
    // App composition is the subject: `useSavedDirectory*` throws without a
    // provider, so reaching the notice at all proves one is mounted above every
    // branch of the list. There is deliberately no process-wide fallback that
    // could make this pass by accident.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    render(<App />)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
  })

  it('offers no way to create a directory anywhere in App', async () => {
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.2)] }
    render(<App />)
    await screen.findByTestId('no-match-callout')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(screen.queryByTestId('new-directory-form')).toBeNull()
  })

  it('keeps the acknowledgement when the ranking request fails', async () => {
    // The original defect, in its surviving half: a request failure replaces
    // the ranked subtree, and the record of a permanent grouping is not part of
    // that subtree's evidence.
    storedDirs = [dirFixture()]
    queuedPredictions[QUERY_A] = [
      { predictions: [modelCard(1, 0.72)] },
      { httpStatus: 503 },
    ]
    render(<App />)
    await screen.findByTestId('new-directory-saved')

    // The refetch a directory change broadcasts -- the same event the key path
    // fires -- and this time it fails.
    await act(async () => {
      window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
    })
    await screen.findByTestId('predictions-error')
    expect(screen.getByTestId('new-directory-saved')).toBeTruthy()
  })

  it('recovers the saved grouping after a reload whose first ranking fails', async () => {
    // Identity has its own endpoint and does not depend on any model's ranking.
    // After a reload the lookup is asked regardless, so a document that plainly
    // has a directory says so even while the ranking is failing.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const first = render(<App />)
    await screen.findByTestId('new-directory-saved')
    first.unmount()

    const seedsBefore = dirGets
    predictionsFor[QUERY_A] = { httpStatus: 503 }
    render(<App />)
    await screen.findByTestId('predictions-error')

    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    expect(dirGets).toBeGreaterThan(seedsBefore)
  })

  it('reports an unanswerable lookup as unresolved, not as "no directory"', async () => {
    // When both endpoints fail the honest answer is "not established". It
    // matters less than it did -- nothing here offers to create one on the
    // strength of a silent "no" any more -- but a blank panel would still be a
    // claim nobody can vouch for.
    dirLookupStatus = 503
    predictionsFor[QUERY_A] = { httpStatus: 503 }
    render(<App />)
    await screen.findByTestId('predictions-error')

    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })
})

describe('the record is scoped to the signed-in account', () => {
  it('shows no previous account state on the first render after a switch', async () => {
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review accountKey={2} />)
    await screen.findByTestId('new-directory-saved')

    // Re-rendering with another account is the switch. The assertion runs on
    // the very next paint: an effect-based reset would leave one frame in which
    // this document looks grouped to somebody it may not be grouped for.
    view.rerender(<Review accountKey={3} />)
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })

  it('does not feed the next account the previous session\'s ranking', async () => {
    // `PredictionProvider` sits above the auth gates: its key is
    // query/model/variant with no account in it, so its settled response, its
    // `seeded_dirs` and its cache all survive a switch. Folding those into "the
    // current store" would hand the new session the old one's snapshot.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<ComposedReview />)
    await screen.findByTestId('new-directory-saved')

    // The world moves on between the two answers: the row is gone by the time
    // the next account asks.
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    storedDirs = []

    await userEvent.click(screen.getByTestId('switch-account'))
    await waitFor(() =>
      expect(screen.queryByTestId('new-directory-saved')).toBeNull(),
    )
  })
})

describe('only the current request feeds the record', () => {
  it('accepts seeded directories the current ranking reports, with attribution', async () => {
    // Observation-only recovery: this reviewer created nothing, and the
    // acknowledgement is still owed to them.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<Review />)
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
    // Somebody else's grouping, and said so.
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Abigail',
    )
  })

  it('states directory status only from the record, never from raw seeded_dirs', async () => {
    // Reading `seeded_dirs` directly for "is this document already grouped?"
    // bypasses every check the record makes on the way in: whose session the
    // response belongs to, and which query actually seeded each row. The row
    // below is a malformed response rather than one the API emits, which is the
    // point -- a stale cache or a backend change must not badge this document
    // with somebody else's grouping.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture({ seed_query_id: QUERY_B, member_query_ids: [QUERY_B] })],
    }
    render(<Review />)
    await screen.findByTestId('band-chip-1')

    expect(screen.queryByTestId('awaiting-match-badge')).toBeNull()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })

  it('does not turn a half-row into a saved identity', async () => {
    // The shared prediction validator is deliberately laxer than the durable
    // record: it fills in `created_by: ''` and friends for an older backend, so
    // a row that reaches the record having been completed by this client is a
    // guess where a stored fact is required.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [{ dir_id: 'reviewer-dir-1', label: 'Half a row', status: 'matched' }],
    }
    render(<Review />)
    // The shared validator refuses the WHOLE response rather than dropping the
    // row, so this is an unreadable ranking -- and, either way, no grouping is
    // claimed for this document.
    await screen.findByTestId('predictions-error')

    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(screen.queryByTestId('matched-dir-badge')).toBeNull()
  })

  it('does not let a failed refresh erase a recovered grouping', async () => {
    storedDirs = [dirFixture()]
    queuedPredictions[QUERY_A] = [
      { predictions: [modelCard(1, 0.72)], seeded_dirs: [dirFixture()] },
      { httpStatus: 503 },
    ]
    render(<Review />)
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')

    await act(async () => {
      window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
    })
    await screen.findByTestId('predictions-error')
    expect(screen.getByTestId('new-directory-saved').textContent).toContain(
      'Unattested homily',
    )
  })
})

describe('the acknowledgement outlives navigation and models', () => {
  it('comes back on query A -> B -> A without re-reading the server', async () => {
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    predictionsFor[QUERY_B] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review queryId={QUERY_A} />)
    await screen.findByTestId('new-directory-saved')
    const getsBefore = dirGets

    view.rerender(<Review queryId={QUERY_B} />)
    await waitFor(() =>
      expect(screen.queryByTestId('new-directory-saved')).toBeNull(),
    )

    view.rerender(<Review queryId={QUERY_A} />)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    // Query A's answer was already established; returning to it asks nothing
    // new about A.
    expect(dirGets).toBeLessThanOrEqual(getsBefore + 1)
  })

  it('survives a model change, because identity does not depend on a model', async () => {
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    const view = render(<Review model={MODEL_A} />)
    await screen.findByTestId('new-directory-saved')
    const gets = predictionGets

    view.rerender(<Review model="bowphs_PhilTa" />)
    expect(screen.getByTestId('new-directory-saved')).toBeTruthy()
    expect(predictionGets).toBeGreaterThanOrEqual(gets)
  })
})
