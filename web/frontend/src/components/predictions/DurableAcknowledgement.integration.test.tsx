import { useEffect, type ReactNode } from 'react'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AccountScopedSavedDirectories } from '../../App'
import App from '../../App'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { ReviewerProvider, useReviewer } from '../../contexts/ReviewerContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import SavedDirectoryObservations from '../../contexts/SavedDirectoryObservations'
import { TokenProvider } from '../../contexts/TokenContext'
import { REVIEWER_DIRS_UPDATED_EVENT, type ReviewerDir } from '../../api/reviewerDirs'
import {
  HISTORICAL_CONFLICT_DETAIL,
  HISTORICAL_SEEDLESS_DIR,
} from '../../test/fixtures/historicalReviewerDir'
import FeedbackPanel from '../feedback/FeedbackPanel'
import PredictionList from './PredictionList'

/**
 * The acknowledgement of a created directory, in the composed application
 * (issue #161, integration).
 *
 * The owned store and CTA are covered by `NewDirectoryCtaBehaviour` and
 * `NewDirectoryCtaInvariants`. What is proved HERE is that the durable record
 * is reachable from the real tree: mounted above every branch of the prediction
 * list, scoped to the signed-in account, fed only by #156's current request,
 * and inert with respect to the assessment panel.
 *
 * Nothing drives the store directly. Every state is reached through the
 * rendered controls a reviewer can actually press, because a control that is
 * not on screen is not a recovery path.
 */

const QUERY_A = 7
const QUERY_B = 8
const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'bowphs_PhilTa'

interface PredictionPayload {
  predictions?: unknown[]
  seeded_dirs?: unknown[]
  status?: string | null
  httpStatus?: number
  malformed?: boolean
}

// Either a row this harness builds or a real captured one: the database
// genuinely holds both shapes, and the preserved pre-#160 rows are the point of
// the finding-11 cases below.
type StoredDir = ReturnType<typeof dirFixture> | ReviewerDir

let predictionsFor: Record<number, PredictionPayload>
let queuedPredictions: Record<number, PredictionPayload[]>
let storedDirs: StoredDir[]
let dirLookupStatus: number
let postBehaviour: 'created' | 'lost' | 'refused' | 'conflict'
let posts: { query_file_id: number; label?: string }[]
let predictionGets: number
let dirGets: number
let feedbackPosts: unknown[]
let refreshEvents: number
let releasePost: (() => void) | null
/** Holds the next predictions GET open, so a response can land late. */
let holdNextPrediction: boolean
let releasePrediction: (() => void) | null
/** Whose session `/api/auth/me` and `/api/auth/signin` answer for. */
let signedInAccount: number
/** What the database holds once the POST has landed, when it is not just the new row. */
let storedAfterPost: StoredDir[] | null

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
    has_qq_matrix: true,
  }
}

function modelCard(rank: number, score: number) {
  return {
    rank,
    dir_name: `dir-${rank}`,
    score,
    dir_files: [],
    preview_text: '',
    candidate_files: null,
    source: 'model',
  }
}

/**
 * A candidate a reviewer can actually judge: issue #157 reads the witness on
 * screen (`candidate_files[0]`), so a card without one is offered struck
 * through and cannot be chosen.
 */
function readableCard(rank: number, score: number, dirName = `dir-${rank}`) {
  return {
    rank,
    dir_name: dirName,
    score,
    dir_files: [`${dirName}-a.txt`],
    preview_text: 'incipit sermo',
    candidate_files: [{ filename: `${dirName}-a.txt`, text: 'incipit sermo' }],
    source: 'model',
  }
}

/** A reviewer directory, anchored at rank 11 the way the API anchors them. */
function reviewerCard(dirName: string, score: number, label: string) {
  return {
    rank: 11,
    dir_name: dirName,
    score,
    dir_files: [`${dirName}-a.txt`],
    preview_text: 'incipit sermo',
    candidate_files: [{ filename: `${dirName}-a.txt`, text: 'incipit sermo' }],
    source: 'reviewer',
    label,
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
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      const path = new URL(url, 'http://localhost').pathname

      if (path === '/api/reviewer_dirs' && method === 'POST') {
        const body = JSON.parse(String(init?.body ?? '{}'))
        posts.push(body)
        if (releasePost) {
          await new Promise<void>((resolve) => {
            releasePost = resolve
          })
        }
        if (postBehaviour === 'refused') {
          return jsonResponse({ detail: 'Authentication required' }, 401)
        }
        if (postBehaviour === 'conflict') {
          // The server's own refusal: this seed already has a directory, so
          // nothing is written and the row named in the message is already
          // there. `storedAfterPost` is what the reconciling read then finds.
          storedDirs = storedAfterPost ?? storedDirs
          return jsonResponse(
            { detail: HISTORICAL_CONFLICT_DETAIL },
            409,
          )
        }
        const created = dirFixture({
          seed_query_id: body.query_file_id,
          label: body.label ?? 'Unattested homily',
          member_query_ids: [body.query_file_id],
        })
        // "lost" means the row IS written and the answer never arrives.
        storedDirs = storedAfterPost ?? [created]
        if (postBehaviour === 'lost') throw new Error('Failed to fetch')
        return jsonResponse(created, 201)
      }

      if (path === '/api/reviewer_dirs') {
        dirGets += 1
        if (dirLookupStatus !== 200) {
          return jsonResponse({ detail: 'unavailable' }, dirLookupStatus)
        }
        const params = new URL(url, 'http://localhost').searchParams
        const seed = params.get('seed_query_id')
        return jsonResponse(
          seed === null
            ? storedDirs
            : storedDirs.filter((dir) => dir.seed_query_id === Number(seed)),
        )
      }

      if (path === '/api/feedback' && method === 'POST') {
        feedbackPosts.push(JSON.parse(String(init?.body ?? '{}')))
        return jsonResponse({ ok: true })
      }

      if (path === '/api/auth/me') {
        return jsonResponse(accountFixture(signedInAccount))
      }
      if (path === '/api/auth/signin') {
        return jsonResponse(accountFixture(signedInAccount))
      }
      if (path === '/api/queries/next') {
        return jsonResponse({ file_id: QUERY_A })
      }
      if (path === '/api/models') {
        return jsonResponse([modelEntry(MODEL_A, 'LaTa'), modelEntry(MODEL_B, 'PhilTa')])
      }
      if (path.endsWith('/predictions')) {
        predictionGets += 1
        const queryId = Number(path.split('/')[3])
        const model = new URL(url, 'http://localhost').searchParams.get('model') ?? ''
        // Read BEFORE any hold, so a response held open carries the world as
        // it stood when the request went out.
        const payload = payloadFor(queryId)
        if (holdNextPrediction) {
          holdNextPrediction = false
          await new Promise<void>((resolve) => {
            releasePrediction = resolve
          })
        }
        if (payload.httpStatus && payload.httpStatus !== 200) {
          return jsonResponse(
            { error: { code: 'internal_error', message: 'Ranking store unavailable' } },
            payload.httpStatus,
          )
        }
        if (payload.malformed) return jsonResponse({ nonsense: true })
        return jsonResponse({
          file_id: queryId,
          filename: `query-${queryId}.txt`,
          model,
          variant: 'sif_abtt',
          status: payload.status === undefined ? 'ok' : payload.status,
          predictions: payload.predictions ?? [],
          seeded_dirs: payload.seeded_dirs ?? [],
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

/**
 * App.tsx's own provider order, with the two panels whose independence matters.
 * `accountKey` is a prop so an account switch can be exercised as a re-render
 * of the same tree, which is what a sign-out actually is.
 */
function Review({
  queryId = QUERY_A,
  model = MODEL_A,
  accountKey = 2,
  withFeedback = false,
}: {
  queryId?: number
  model?: string
  accountKey?: number | null
  withFeedback?: boolean
}) {
  return (
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <SavedDirectoryProvider accountKey={accountKey}>
            <PredictionProvider>
              <SavedDirectoryObservations />
              <FeedbackProvider>
                <Selection queryId={queryId} model={model} />
                <PredictionList />
                {withFeedback && <FeedbackPanel />}
              </FeedbackProvider>
            </PredictionProvider>
          </SavedDirectoryProvider>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}

/**
 * The same tree, scoped by the REAL `AccountScopedSavedDirectories` reading the
 * real `ReviewerContext`, with a control that signs a different reviewer in.
 * An account transition here is what one actually is: a new user object from
 * the auth endpoints, with the prediction provider, its cache and its settled
 * ranking all still mounted underneath it.
 */
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

/**
 * App's own gate: nothing below the auth boundary renders until the session is
 * known, so no document is selected and nothing is on screen to act on while
 * the account is still resolving.
 */
function AuthGate({ children }: { children: ReactNode }) {
  const { loading } = useReviewer()
  return loading ? null : <>{children}</>
}

function ComposedReview({
  queryId = QUERY_A,
  model = MODEL_A,
}: {
  queryId?: number
  model?: string
}) {
  return (
    <AppProvider>
      <ReviewerProvider>
        <TokenProvider>
          <AccountScopedSavedDirectories>
            <PredictionProvider>
              <SavedDirectoryObservations />
              <FeedbackProvider>
                <AuthGate>
                  <Selection queryId={queryId} model={model} />
                  <SignInAsOtherAccount />
                  <PredictionList />
                </AuthGate>
              </FeedbackProvider>
            </PredictionProvider>
          </AccountScopedSavedDirectories>
        </TokenProvider>
      </ReviewerProvider>
    </AppProvider>
  )
}

/** The note box, read as a control rather than as state. */
function noteValue(): string {
  return (
    screen.getByPlaceholderText('Add notes for this query...') as HTMLTextAreaElement
  ).value
}

/** Submit, whose enabled state is issue #157's verdict on the whole draft. */
function submitButton(): HTMLElement {
  return screen.getByRole('button', { name: /Submit/ })
}

/**
 * Which choice the document is filed under, read from the sentence issue #157
 * actually shows the reviewer. Asserting the rank that must NOT be canonical is
 * the point: with choices in rank order, a draft that had been re-sorted would
 * be indistinguishable from one that was preserved.
 */
function expectCanonicalChoice(rank: number, notRank = rank === 1 ? 2 : 1): void {
  const note = screen.getByTestId('canonical-choice-note').textContent ?? ''
  expect(note).toContain(`first choice, #${rank},`)
  expect(note).not.toContain(`first choice, #${notRank},`)
}

async function createDirectory(
  user: ReturnType<typeof userEvent.setup>,
  label = 'Unattested homily',
) {
  await user.click(await screen.findByTestId('new-directory-cta'))
  const field = await screen.findByLabelText('Name the new directory')
  await user.clear(field)
  await user.type(field, label)
  await user.click(await screen.findByTestId('new-directory-submit'))
}

beforeEach(() => {
  predictionsFor = {}
  queuedPredictions = {}
  storedDirs = []
  dirLookupStatus = 200
  postBehaviour = 'created'
  posts = []
  predictionGets = 0
  dirGets = 0
  feedbackPosts = []
  refreshEvents = 0
  releasePost = null
  holdNextPrediction = false
  releasePrediction = null
  signedInAccount = 2
  storedAfterPost = null
  window.localStorage.clear()
  installFetch()
  window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, countRefresh)
})

afterEach(() => {
  window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, countRefresh)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function countRefresh() {
  refreshEvents += 1
}

describe('the provider is mounted in the real application', () => {
  it('renders the creation flow inside App without a fallback store', async () => {
    // App composition is the subject: `useSavedDirectory*` throws without a
    // provider, so reaching the CTA at all proves one is mounted above every
    // branch of the list. There is deliberately no process-wide fallback that
    // could make this pass by accident.
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    render(<App />)
    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
  })

  it('keeps the acknowledgement in App when the refetch it caused fails', async () => {
    // The original bug, end to end: the creation broadcasts a refresh, the
    // refresh fails, and the only evidence of a permanent write used to go with
    // it.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    queuedPredictions[QUERY_A] = [
      { predictions: [modelCard(1, 0.72)] },
      { httpStatus: 503 },
    ]
    render(<App />)

    await createDirectory(user)

    await screen.findByTestId('predictions-error')
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })

  it('records a grouping the App\'s own ranking reports', async () => {
    // The observation bridge is part of App's composition, not of the list.
    // `PredictionList` no longer ingests anything itself, so an acknowledgement
    // appearing here from `seeded_dirs` alone is proof that the single
    // admission point is mounted in the real tree.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<App />)
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
  })

  it('recovers the saved grouping after a reload whose first ranking fails', async () => {
    // Identity has its own endpoint and does not depend on any model's ranking,
    // but the only thing that ASKED it used to be the CTA -- and after a reload
    // neither CTA position mounts while the ranking is loading or has failed.
    // So a document that plainly had a directory showed nothing at all until
    // some ranking recovered, with the directory endpoint healthy throughout.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const first = render(<App />)
    await screen.findByTestId('new-directory-saved')
    first.unmount()

    // The reload. Same healthy directory endpoint, same unchanged row, and a
    // ranking that will not load.
    const seedsBefore = dirGets
    predictionsFor[QUERY_A] = { httpStatus: 503 }
    render(<App />)
    await screen.findByTestId('predictions-error')

    // Asked, answered, and acknowledged through the error.
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    expect(dirGets).toBeGreaterThan(seedsBefore)
    // Still no fresh creation on offer: that decision needs a settled ranking
    // and is untouched by this.
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(screen.queryByTestId('new-directory-form')).toBeNull()
  })

  it('reports an unanswerable lookup as unresolved, not as room to create', async () => {
    // The other half of asking early: when BOTH endpoints fail the honest
    // answer is "not established", offered with its own re-check. That control
    // is deliberately not the one a half-finished write shows -- nothing has
    // been written here.
    dirLookupStatus = 503
    predictionsFor[QUERY_A] = { httpStatus: 503 }
    render(<App />)
    await screen.findByTestId('predictions-error')

    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.getByTestId('new-directory-recheck')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-check-again')).toBeNull()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })
})

describe('observed evidence has to be readable as a directory', () => {
  // The rows here go through the REAL shared prediction validator, which is
  // where the danger is. Its contract for a directory inside a ranking is
  // deliberately laxer than the durable record's: it accepts a row with no
  // `member_query_ids`, `created_at` or `created_by` and fills in `[]` and `''`
  // so an older backend still renders a candidate. Those substitutes are
  // type-valid, so nothing downstream can tell them from a server that really
  // said "no members, no creator, no timestamp".
  const incomplete = {
    dir_id: 'reviewer-dir-incomplete',
    label: 'Incomplete row',
    seed_query_id: QUERY_A,
    status: 'matched',
  }

  it('does not turn a half-row into a saved identity', async () => {
    dirLookupStatus = 503
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [incomplete],
    }
    render(<App />)
    await screen.findByTestId('match-pill-1')

    // No acknowledgement, and in particular none of the blanks: a reviewer
    // reading "Created by ." learns nothing and cannot tell that it is a fault.
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(document.body.textContent).not.toContain('Created by .')
    // Not authoritative absence either. The question is open, and the way back
    // to an answer is on screen.
    expect(screen.getByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.getByTestId('new-directory-recheck')).toBeTruthy()
  })

  it('does not let a half-row overwrite the history already recovered', async () => {
    // Two complete groups are known from the server, oldest first. A later
    // half-row for the younger one carries a blank timestamp, which sorts
    // before any real one -- so accepting it would silently rename the group
    // this document is filed under and blank out who made it.
    const older = dirFixture({
      dir_id: 'reviewer-dir-older',
      label: 'Oldest group',
      created_by: 'Older creator',
      created_at: '2026-09-01 12:00:00',
      status: 'matched',
      member_query_ids: [QUERY_A, 9],
    })
    storedDirs = [older, dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review model={MODEL_A} />)
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Oldest group')
    expect(saved.textContent).toContain('Older creator')

    // A model switch is a new request, which is how a later response actually
    // reaches the record. Identity does not depend on a model, so this is the
    // ordinary way a second opinion about the same document arrives.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [{ ...incomplete, dir_id: dirFixture().dir_id, status: 'awaiting_match' }],
    }
    const seen = predictionGets
    view.rerender(<Review model={MODEL_B} />)
    await waitFor(() => expect(predictionGets).toBeGreaterThan(seen))

    // Everything the server actually said is still here, in the order it said
    // it, with the attribution it gave.
    const after = await screen.findByTestId('new-directory-saved')
    expect(after.textContent).toContain('Oldest group')
    expect(after.textContent).toContain('Older creator')
    expect(after.textContent).toContain('Unattested homily')
    expect(after.textContent).not.toContain('Created by .')
  })

  it('does not let a half-row settle a write whose outcome is unknown', async () => {
    // The most damaging version. A write is outstanding with no outcome, so the
    // reviewer has Check again in front of them. A half-row would close that
    // form, retire the recovery control and announce a confirmed save on
    // evidence that never met the bar the 201 and the lookup are held to.
    const user = userEvent.setup()
    postBehaviour = 'lost'
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review model={MODEL_A} />)
    await screen.findByTestId('new-directory-cta')
    dirLookupStatus = 503
    await createDirectory(user)
    await screen.findByTestId('new-directory-check-again')
    const eventsBefore = refreshEvents

    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [incomplete],
    }
    const seen = predictionGets
    view.rerender(<Review model={MODEL_B} />)
    await waitFor(() => expect(predictionGets).toBeGreaterThan(seen))

    // Still unknown, still recoverable, still unannounced, and the name the
    // reviewer typed is still theirs to resubmit.
    expect(screen.getByTestId('new-directory-check-again')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(screen.queryByTestId('new-directory-created')).toBeNull()
    expect(refreshEvents).toBe(eventsBefore)
    expect(posts).toHaveLength(1)
  })
})

describe('the record is scoped to the signed-in account', () => {
  it('shows no previous account state on the first render after a switch', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review accountKey={2} />)
    await createDirectory(user)
    await screen.findByTestId('new-directory-saved')

    // Re-rendering with another account is the switch. The assertion runs on
    // the very next paint: an effect-based reset would leave one frame in which
    // this document looks grouped to somebody it may not be grouped for.
    view.rerender(<Review accountKey={3} />)
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })

  it('does not feed the next account the previous session\'s ranking', async () => {
    // The gap this closes. The saved store is replaced on an account change,
    // but `PredictionProvider` is not: it sits above the auth gates, its key is
    // query/model/variant with no account in it, and its settled response, its
    // `seeded_dirs` and its cache all survive the switch. Folding those into
    // "the current store" hands the new session the old one's snapshot.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<ComposedReview />)
    await screen.findByTestId('new-directory-saved')
    const getsBefore = predictionGets

    // The world moves on between the two answers. Nothing about this is
    // account-private: it is simply a LATER answer, and the record is supposed
    // to follow the latest one rather than the snapshot it was mounted over.
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }

    await user.click(screen.getByTestId('switch-account'))

    // The new session asks for itself rather than reading what was on screen.
    await waitFor(() => expect(predictionGets).toBeGreaterThan(getsBefore))
    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    // Re-establishing provenance is not a directory write, so nothing is
    // announced to the rest of the app.
    expect(refreshEvents).toBe(0)
    expect(posts).toHaveLength(0)

    // And it settles: one refetch per account, not a loop.
    const getsAfter = predictionGets
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(predictionGets).toBe(getsAfter)
  })

  it('lets the same global grouping reappear once this session has seen it', async () => {
    // Reviewer directories are global. Withholding the previous session's
    // snapshot is about provenance, not secrecy: the moment this account's own
    // request answers, the same row, the same creator and the same history are
    // shown again.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<ComposedReview />)
    await screen.findByTestId('new-directory-saved')

    await user.click(screen.getByTestId('switch-account'))

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Abigail',
    )
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })

  it('drops a ranking that was still in flight when the account changed', async () => {
    // The late case. A request issued by the previous session answers after the
    // switch; its `seeded_dirs` are that session's evidence and must not be
    // recorded as this one's.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<ComposedReview />)
    await screen.findByTestId('new-directory-cta')

    // A request goes out under the old account and is held open.
    holdNextPrediction = true
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    view.rerender(<ComposedReview model={MODEL_B} />)
    await waitFor(() => expect(releasePrediction).not.toBeNull())

    // The account changes while it is still unanswered, and the world it was
    // describing is gone by the time it lands.
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    await user.click(screen.getByTestId('switch-account'))
    await act(async () => {
      releasePrediction?.()
      await Promise.resolve()
    })

    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
    await waitFor(() => expect(screen.queryByTestId('predictions-loading')).toBeNull())
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })

  it('does not turn the previous session\'s uncertainty into permission to create', async () => {
    // Identity before the single owned event, across an account change. The old
    // session's write may have landed; that uncertainty is its own, and the new
    // session inherits neither the write nor an answer to it. Because it cannot
    // establish anything for itself either, it must say so rather than offer a
    // fresh Create for a document that may already be spoken for.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    postBehaviour = 'lost'

    render(<ComposedReview />)
    await user.click(await screen.findByTestId('new-directory-cta'))
    const field = await screen.findByLabelText('Name the new directory')
    await user.type(field, 'Unattested homily')
    // From here nothing can answer: the write is issued, its response is lost
    // and the reconciling lookup fails.
    dirLookupStatus = 503
    await user.click(await screen.findByTestId('new-directory-submit'))

    await screen.findByTestId('new-directory-check-again')
    expect(posts).toHaveLength(1)
    // Nothing was confirmed, so nothing was announced.
    expect(refreshEvents).toBe(0)

    // The account changes. The new session inherits neither the
    // acknowledgement nor the write, and its own lookup still cannot answer.
    await user.click(screen.getByTestId('switch-account'))

    // So it says it does not know, instead of offering a fresh Create for a
    // document that may already be spoken for.
    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(screen.queryByTestId('new-directory-form')).toBeNull()
    expect(posts).toHaveLength(1)
    expect(refreshEvents).toBe(0)
  })

  it('states directory status only from the record, never from raw seeded_dirs', async () => {
    // `seeded_dirs` is raw response data. Reading it directly for "is this
    // document already grouped?" bypasses every check the record makes on the
    // way in: whose session the response belongs to, and which query actually
    // seeded each row. The current backend does filter this list by
    // `seed_query_id == file_id` (`web/routers/predictions.py`), so this
    // foreign-seed row is a MALFORMED response rather than one the API emits
    // today -- which is the point. The raw read trusted the field instead of
    // checking it, so anything that ever put a foreign row there, a backend
    // change or a stale cache, silently suppressed this document's Create
    // button and badged it with somebody else's grouping.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture({ seed_query_id: QUERY_B, member_query_ids: [QUERY_B] })],
    }
    render(<Review />)

    // The record keeps directories under the query that seeded them, so this
    // document is offered its own, and is not badged for a grouping it did not
    // seed.
    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
    expect(screen.queryByTestId('awaiting-match-badge')).toBeNull()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
  })
})

describe('only the current request feeds the record', () => {
  it('accepts seeded directories the current ranking reports', async () => {
    // Observation-only recovery: this reviewer created nothing in this session,
    // and the acknowledgement is still owed to them after a reload.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    render(<Review />)
    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
    // Somebody else's wording, and said so.
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Abigail',
    )
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })

  it('does not let a failed refresh erase the groups or their members', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    // A second, historical grouping for the same seed, with a human-filed
    // second witness. Nothing may hide either fact.
    storedAfterPost = [
      dirFixture({
        dir_id: 'reviewer-dir-0',
        label: 'Older grouping',
        created_at: '2026-08-20 00:00:00',
        member_query_ids: [QUERY_A, 41],
        status: 'matched',
      }),
      dirFixture(),
    ]
    queuedPredictions[QUERY_A] = [
      { predictions: [modelCard(1, 0.72)] },
      { httpStatus: 503 },
    ]
    postBehaviour = 'lost'

    render(<Review />)
    await createDirectory(user)

    // Recovered through the reconciling lookup, not through the ranking.
    const saved = await screen.findByTestId('new-directory-saved')
    await screen.findByTestId('predictions-error')
    // The oldest grouping is the one the backend resolves for this seed, and
    // the later duplicate is named rather than hidden.
    expect(saved.textContent).toContain('Older grouping')
    expect(screen.getByTestId('new-directory-other-groups').textContent).toContain(
      'Unattested homily',
    )
    // A human filed a second witness into the older group. That cannot be lost
    // to a failed refresh either.
    expect(screen.getByTestId('matched-dir-badge')).toBeTruthy()
  })

  it('does not let an empty ranking erase a saved identity', async () => {
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    queuedPredictions[QUERY_A] = [
      { predictions: [modelCard(1, 0.72)], seeded_dirs: [dirFixture()] },
      { predictions: [], seeded_dirs: [], status: 'ok' },
    ]
    const view = render(<Review />)
    await screen.findByTestId('new-directory-saved')

    // A later request for the same document comes back with nothing at all.
    view.rerender(<Review model={MODEL_B} />)
    await screen.findByTestId('predictions-empty')
    expect(screen.getByTestId('new-directory-saved').textContent).toContain(
      'Unattested homily',
    )
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })
})

describe('the acknowledgement outlives navigation, models and reloads', () => {
  it('comes back on query A -> B -> A without re-reading the server', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    predictionsFor[QUERY_B] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review queryId={QUERY_A} />)
    await createDirectory(user)
    await screen.findByTestId('new-directory-saved')

    view.rerender(<Review queryId={QUERY_B} />)
    // B is a different document and gets its own decision, not A's answer.
    expect(await screen.findByTestId('new-directory-cta')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()

    const getsAfterB = dirGets
    view.rerender(<Review queryId={QUERY_A} />)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    expect(dirGets).toBe(getsAfterB)
  })

  it('survives a model change, because identity does not depend on a model', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review model={MODEL_A} />)
    await createDirectory(user)
    await screen.findByTestId('new-directory-saved')

    view.rerender(<Review model={MODEL_B} />)
    expect(screen.getByTestId('new-directory-saved').textContent).toContain(
      'Unattested homily',
    )
  })

  it('recovers from the server after a reload', async () => {
    // A reload is a new tree with no memory: nothing is persisted to browser
    // storage, so the seed-filtered lookup is the only thing that can answer.
    storedDirs = [dirFixture()]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    const view = render(<Review />)
    await screen.findByTestId('new-directory-saved')
    view.unmount()

    render(<Review />)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })
})

describe('a write in flight is not lost by closing the panel', () => {
  it('keeps the lock, the name and the single POST', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    releasePost = () => {}
    render(<Review />)

    await user.click(await screen.findByTestId('new-directory-cta'))
    const field = screen.getByLabelText('Name the new directory')
    await user.clear(field)
    await user.type(field, 'Dubious capitulary')
    await user.click(screen.getByTestId('new-directory-submit'))
    await waitFor(() => expect(posts).toHaveLength(1))

    // Close, which hides a form and settles nothing.
    await user.click(screen.getByTestId('new-directory-cancel'))
    expect(screen.getByTestId('new-directory-pending-closed')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()

    const release = releasePost
    releasePost = null
    await act(async () => {
      release?.()
      await Promise.resolve()
    })

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Dubious capitulary')
    expect(posts).toHaveLength(1)
  })
})

describe('every recovery path reaches the same acknowledgement', () => {
  it('confirms a 201 and announces it once', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    render(<Review />)
    await createDirectory(user)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    await waitFor(() => expect(refreshEvents).toBe(1))
  })

  it('recovers immediately when the response is lost but the row is not', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    postBehaviour = 'lost'
    render(<Review />)
    await createDirectory(user)

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain('Unattested homily')
    // Neutral wording: nothing here establishes that the server refused.
    expect(screen.getByTestId('new-directory-recovered')).toBeTruthy()
    await waitFor(() => expect(refreshEvents).toBe(1))
  })

  it('recovers on a delayed Check again, and only then announces', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    postBehaviour = 'lost'
    render(<Review />)
    // The seed is free and readable when the reviewer opens the form; the
    // server becomes unreadable only once the write is on its way.
    await screen.findByTestId('new-directory-cta')
    dirLookupStatus = 503
    await createDirectory(user)

    // Both the write and the reconciliation failed, so the outcome is genuinely
    // unknown. No Create button, no second submission, and nothing announced.
    await screen.findByTestId('new-directory-outcome')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect((screen.getByTestId('new-directory-submit') as HTMLButtonElement).disabled).toBe(
      true,
    )
    expect(refreshEvents).toBe(0)

    dirLookupStatus = 200
    await user.click(screen.getByTestId('new-directory-check-again'))
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    await waitFor(() => expect(refreshEvents).toBe(1))
    expect(posts).toHaveLength(1)
  })

  it('recovers from the next ranking alone when no lookup ever answers', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    postBehaviour = 'lost'
    const view = render(<Review />)
    await screen.findByTestId('new-directory-cta')
    dirLookupStatus = 503
    await createDirectory(user)
    await screen.findByTestId('new-directory-outcome')

    // The reviewer switches model, and the ranking for the new one comes back
    // carrying the directory. That is the same class of evidence a lookup
    // returns -- positive, seed-matched and server-sourced -- so it settles the
    // write even though no lookup ever answered.
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [dirFixture()],
    }
    view.rerender(<Review model={MODEL_B} />)

    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
    await waitFor(() => expect(refreshEvents).toBe(1))
    expect(posts).toHaveLength(1)
  })

  it('mounts the CTA once below the no-match band', async () => {
    // The callout owns the offer up top; the foot CTA owns the acknowledgement.
    // Never both.
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.21)] }
    render(<Review />)

    expect(screen.queryAllByTestId('new-directory-cta')).toHaveLength(0)
    await within(await screen.findByTestId('no-match-callout')).findByTestId(
      'new-directory-cta',
    )
    expect(screen.queryAllByTestId('new-directory-cta')).toHaveLength(1)

    await createDirectory(user)
    await waitFor(() =>
      expect(screen.queryAllByTestId('new-directory-saved')).toHaveLength(1),
    )
    expect(screen.queryAllByTestId('new-directory-cta')).toHaveLength(0)
  })
})

describe('the assessment is not touched by any of this', () => {
  it('submits no feedback, presses no control and keeps the note draft', async () => {
    const user = userEvent.setup()
    predictionsFor[QUERY_A] = { predictions: [readableCard(1, 0.72)] }
    queuedPredictions[QUERY_A] = [
      { predictions: [readableCard(1, 0.72)] },
      { httpStatus: 503 },
    ]
    postBehaviour = 'lost'
    render(<Review withFeedback />)

    const notes = await screen.findByPlaceholderText('Add notes for this query...')
    await user.type(notes, 'Possibly Hincmar')
    // The controls exist and are untouched before any of this starts.
    expect(
      screen.getByRole('button', { name: 'Match prediction #1' }).getAttribute('aria-pressed'),
    ).toBe('false')

    await createDirectory(user)
    await screen.findByTestId('new-directory-saved')

    // The creation, the lost response, the recovery and the failed refetch it
    // triggered are all over, and nothing was answered on the reviewer's behalf.
    expect(feedbackPosts).toHaveLength(0)
    expect(noteValue()).toBe('Possibly Hincmar')
    expect(screen.queryByTestId('canonical-choice-note')).toBeNull()
  })

  it('keeps an existing ordered answer and note across the refresh it causes', async () => {
    // The combined case. A reviewer has already answered -- two ranks, in an
    // order that decides which directory the document is filed under, plus
    // prose -- and only THEN creates a directory. The creation broadcasts a
    // refresh, which goes loading, fails, and finally returns the SAME ranking.
    // None of that is evidence about their answer, so all of it survives.
    const user = userEvent.setup()
    const ranking = [readableCard(1, 0.72), readableCard(2, 0.6)]
    predictionsFor[QUERY_A] = { predictions: ranking }
    queuedPredictions[QUERY_A] = [
      { predictions: ranking },
      { httpStatus: 503 },
      { predictions: ranking },
    ]
    render(<Review withFeedback />)

    await user.click(await screen.findByLabelText('Select multiple'))
    // Order is the answer, and it is deliberately NOT rank order: the reviewer
    // picked #2 first, so #2 is the canonical filing. Ascending choices would
    // agree with an accidental sort and prove nothing, which is why every
    // assertion below also states which rank must NOT be canonical.
    await user.click(screen.getByRole('button', { name: 'Match prediction #2' }))
    await user.click(screen.getByRole('button', { name: 'Match prediction #1' }))
    await user.type(
      screen.getByPlaceholderText('Add notes for this query...'),
      'Shares the Hincmar incipit',
    )
    expectCanonicalChoice(2)
    // A deliberate, currently valid answer -- issue #157 only enables Submit
    // for one, so this is the precondition that there is an outcome here to
    // preserve, asserted without recording it.
    expect(submitButton().getAttribute('disabled')).toBeNull()

    await createDirectory(user)

    // Loading and failure are statements about a request, not about the
    // answer. Issue #157 draws no pills without a ranking -- inventing them is
    // what let Submit post against a ranking nobody had seen -- so what matters
    // here is that the draft is not RECONCILED against the absence: nothing is
    // reported as dropped, and the panel says the evidence is unavailable
    // rather than that the choices were wrong.
    await screen.findByTestId('predictions-error')
    expect(screen.getByTestId('assessment-unavailable')).toBeTruthy()
    expect(screen.queryByTestId('assessment-notice')).toBeNull()
    expect(noteValue()).toBe('Shares the Hincmar incipit')

    // The ranking comes back unchanged.
    await user.click(screen.getByTestId('predictions-retry'))
    await waitFor(() =>
      expect(screen.getByTestId('match-pill-1').getAttribute('disabled')).toBeNull(),
    )
    expect(screen.getByTestId('match-pill-2').getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByTestId('match-pill-1').getAttribute('aria-pressed')).toBe('true')
    // Still theirs, still in their order, still with their reasoning, and the
    // directory acknowledgement is on screen beside it. Canonical is #2 and not
    // #1: a refresh that quietly re-sorted the answer into rank order would
    // change which directory the document is filed under.
    expectCanonicalChoice(2)
    expect(noteValue()).toBe('Shares the Hincmar incipit')
    expect(screen.queryByTestId('assessment-notice')).toBeNull()
    // The outcome survived as an outcome: still submittable, still unsaved.
    expect(submitButton().getAttribute('disabled')).toBeNull()
    await screen.findByTestId('new-directory-saved')

    // Nothing was saved and nothing advanced: creating a directory is not an
    // assessment, and a refresh is not a submission.
    expect(feedbackPosts).toHaveLength(0)
    expect(screen.getByTestId('match-pill-1')).toBeTruthy()
  })

  it('drops only the choice a reassignment invalidated, and says why', async () => {
    // The companion case, and the reason rank 11 needs one: reviewer
    // directories are anchored there and `web/services/reviewer_dirs.py` skips
    // the ones the query already belongs to, so the directory behind rank 11
    // genuinely changes between responses. Per issue #157 that choice is
    // removed rather than silently re-pointed at whatever now sits there.
    const user = userEvent.setup()
    const before = [
      readableCard(1, 0.72),
      readableCard(2, 0.6),
      reviewerCard('reviewer-dir-early', 0.66, 'Early grouping'),
    ]
    const after = [
      readableCard(1, 0.72),
      readableCard(2, 0.6),
      reviewerCard('reviewer-dir-other', 0.64, 'Another grouping'),
    ]
    predictionsFor[QUERY_A] = { predictions: before }
    queuedPredictions[QUERY_A] = [{ predictions: before }, { predictions: after }]
    render(<Review withFeedback />)

    await user.click(await screen.findByLabelText('Select multiple'))
    // Three choices, with the one about to be invalidated in the MIDDLE of a
    // non-ascending order. Removing it therefore has to splice it out and leave
    // [#2, #1] as it was; both re-sorting the remainder and collapsing to the
    // first surviving rank would read as #1 and be caught.
    await user.click(
      await screen.findByRole('button', { name: 'Match prediction #2' }),
    )
    await user.click(screen.getByRole('button', { name: 'Match reviewer directory #11' }))
    await user.click(screen.getByRole('button', { name: 'Match prediction #1' }))
    await user.type(
      screen.getByPlaceholderText('Add notes for this query...'),
      'Second witness looks like the same scribe',
    )
    // All three are genuinely chosen, in that order, before anything happens to
    // the ranking, and the draft is a valid answer.
    expect(screen.getByTestId('match-pill-1').getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByTestId('match-pill-2').getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByTestId('match-pill-11').getAttribute('aria-pressed')).toBe('true')
    expectCanonicalChoice(2)
    expect(submitButton().getAttribute('disabled')).toBeNull()

    await createDirectory(user, 'Unattested homily')

    // The refresh the creation caused returns a ranking in which rank 11 holds
    // a different directory.
    await waitFor(() =>
      expect(screen.getByTestId('match-pill-11').getAttribute('aria-pressed')).toBe(
        'false',
      ),
    )
    const notice = screen.getByTestId('assessment-notice')
    expect(notice.textContent).toContain('Rank #11 now holds reviewer-dir-other')
    expect(notice.textContent).toContain('removed rather than moved')

    // Only that one. The valid choices keep their identity AND their order:
    // canonical is still #2, the choice the reviewer made first, not the lowest
    // surviving rank and not whatever the new ranking puts first.
    expect(screen.getByTestId('match-pill-1').getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByTestId('match-pill-2').getAttribute('aria-pressed')).toBe('true')
    expectCanonicalChoice(2)
    expect(noteValue()).toBe('Second witness looks like the same scribe')
    // What is left is still a deliberate valid answer, so dropping the invalid
    // choice narrowed the draft rather than invalidating the whole of it.
    expect(submitButton().getAttribute('disabled')).toBeNull()
    expect(feedbackPosts).toHaveLength(0)
    await screen.findByTestId('new-directory-saved')
  })
})

/**
 * Review finding 11, through the real application.
 *
 * `HISTORICAL_SEEDLESS_DIR` is not invented: an independent reviewer produced
 * it by executing the pre-#160 creation method against a disposable database,
 * letting an ordinary authenticated request commit the shared connection
 * between its two INSERTs, and then failing the creation. The current server
 * preserves that row, serves it on both endpoints and refuses a second
 * directory for the same seed. Nothing can repair or remove it.
 *
 * So the client has to read it. The alternative -- calling it unreadable -- is
 * a dead end for the reviewer: no acknowledgement of a grouping that exists, no
 * Create (409), and a re-check that can only fail the same way.
 */
describe('a directory the server preserved without its seed membership', () => {
  const seedless = { ...HISTORICAL_SEEDLESS_DIR }
  const completeSibling = dirFixture({
    dir_id: 'reviewer-dir-complete',
    label: 'Complete historical grouping',
    created_at: '2026-09-12 00:00:00',
    created_by: 'Another reviewer',
    status: 'matched',
    member_query_ids: [QUERY_A, 9],
  })

  it('acknowledges it on a cold reload without inventing the missing membership', async () => {
    storedDirs = [seedless]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)], seeded_dirs: [seedless] }
    render(<App />)
    await screen.findByTestId('match-pill-1')

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain(seedless.label)
    expect(saved.textContent).toContain(seedless.created_by)
    // Said, not papered over. The grouping is stored and the document is not
    // recorded in it, and the reviewer is told both rather than the notice
    // being left to imply the second.
    expect(within(saved).getByTestId('new-directory-membership-gap')).toBeTruthy()
    // No Create: the server would refuse it with a 409, and offering it would
    // invite a reviewer to try.
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(posts).toEqual([])
    expect(refreshEvents).toBe(0)
  })

  it('does not hide a complete sibling behind the partial one', async () => {
    // The list is validated as a whole and rejected as a whole, so treating the
    // preserved row as unreadable hid every grouping this document has.
    storedDirs = [seedless, completeSibling]
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [seedless, completeSibling],
    }
    render(<App />)
    await screen.findByTestId('match-pill-1')

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain(seedless.label)
    expect(saved.textContent).toContain(completeSibling.label)
    expect(saved.textContent).toContain('Another reviewer')
    // The sibling's status reaches the reviewer through the acknowledgement,
    // which is the one place that does not depend on a prediction refresh.
    expect(within(saved).getByTestId('matched-dir-badge')).toBeTruthy()
    // The caveat attaches to the row it is true of, and only that one.
    const gaps = within(saved).getAllByText(/does not list this document among its members/)
    expect(gaps).toHaveLength(1)
  })

  it('recovers structured identity from a 409 that names it', async () => {
    // The sequence a reviewer actually hits on one of these documents: the
    // lookup finds nothing yet (a stale cache, another tab, a slow replica),
    // they create, the server refuses because the preserved row already seeds
    // this query, and the reconciling read returns that row.
    const user = userEvent.setup()
    postBehaviour = 'conflict'
    storedAfterPost = [seedless]
    predictionsFor[QUERY_A] = { predictions: [modelCard(1, 0.72)] }
    render(<App />)
    await createDirectory(user)

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain(seedless.label)
    expect(saved.textContent).toContain(seedless.created_by)
    expect(within(saved).getByTestId('new-directory-membership-gap')).toBeTruthy()
    // Neutral about whose write it was, because this layer cannot tell a
    // refusal from a lost response.
    expect(within(saved).getByTestId('new-directory-recovered')).toBeTruthy()
    expect(posts).toHaveLength(1)
    // One refresh, for the one write this session owns.
    expect(refreshEvents).toBe(1)
  })

  it('still refuses a row the ranking validator filled in for the server', async () => {
    // The half of the distinction that must NOT move. This row reaches the
    // record through the real shared validator with `member_query_ids` absent
    // rather than empty, which normalises to the same `[]` the preserved row
    // carries. Accepting it because the preserved row is acceptable would
    // reopen finding 10 by the back door.
    const { member_query_ids: _absent, ...noMembers } = dirFixture()
    dirLookupStatus = 503
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [noMembers],
    }
    render(<App />)
    await screen.findByTestId('match-pill-1')

    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(screen.getByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.getByTestId('new-directory-recheck')).toBeTruthy()
  })

  it('reads the same row back when the lookup recovers', async () => {
    // The way out of the state above: the field was missing from the ranking,
    // not from the database, and the authoritative read settles it.
    dirLookupStatus = 503
    const { member_query_ids: _absent, ...noMembers } = dirFixture()
    predictionsFor[QUERY_A] = {
      predictions: [modelCard(1, 0.72)],
      seeded_dirs: [noMembers],
    }
    const user = userEvent.setup()
    render(<App />)
    await screen.findByTestId('new-directory-recheck')

    dirLookupStatus = 200
    storedDirs = [seedless]
    await user.click(screen.getByTestId('new-directory-recheck'))

    const saved = await screen.findByTestId('new-directory-saved')
    expect(saved.textContent).toContain(seedless.label)
    expect(posts).toEqual([])
    // A read, not a write of ours: nothing to announce.
    expect(refreshEvents).toBe(0)
  })
})
