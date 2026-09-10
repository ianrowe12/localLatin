import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { REVIEWER_DIRS_UPDATED_EVENT, type ReviewerDir } from '../api/reviewerDirs'
import { SavedDirectoryStore } from './savedDirectoryStore'

/**
 * Issue #161: the acknowledgement of a permanent write must not depend on a
 * prediction request succeeding.
 *
 * These exercise the real client (`api/reviewerDirs.ts` -> `apiFetch` ->
 * `fetch`) rather than a mocked hook, so the request shapes asserted here are
 * the ones the deployment actually issues.
 */

const QUERY_A = 7
const QUERY_B = 12
const MODEL = 'bowphs_LaTa'

function dirFixture(overrides: Partial<ReviewerDir> = {}): ReviewerDir {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    status: 'awaiting_match',
    seed_query_id: QUERY_A,
    member_query_ids: [QUERY_A],
    created_at: '2026-08-26 00:00:00',
    created_by: 'Abigail',
    model_slug: MODEL,
    variant: 'sif_abtt',
    best_match_score: 0.31,
    has_potential_match: false,
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

interface Call {
  url: string
  method: string
  body: unknown
}

let calls: Call[] = []

type Route = (url: string, init?: RequestInit) => Response | Promise<Response>

/** Default routes: creation succeeds, the seed lookup finds nothing. */
function installFetch(routes: { post?: Route; get?: Route } = {}): void {
  calls = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      calls.push({
        url,
        method,
        body: init?.body ? JSON.parse(String(init.body)) : null,
      })
      if (method === 'POST') {
        return routes.post
          ? routes.post(url, init)
          : jsonResponse(dirFixture(), 201)
      }
      return routes.get ? routes.get(url, init) : jsonResponse([])
    }),
  )
}

function getCalls(): Call[] {
  return calls.filter((call) => call.method === 'GET')
}

/** A promise plus its resolvers, for interleaving two in-flight requests. */
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('seed-filtered recovery lookup', () => {
  it('asks the database about this seed, not about a model request', async () => {
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A, { model: MODEL })

    const [call] = getCalls()
    expect(call.url).toContain('/api/reviewer_dirs?')
    expect(call.url).toContain('seed_query_id=7')
    expect(call.url).toContain(`model=${MODEL}`)
    expect(store.getRecord(QUERY_A).identity.status).toBe('absent')
  })

  it('rehydrates a saved grouping after a full remount, with its real label', async () => {
    installFetch({
      get: () => jsonResponse([dirFixture({ label: 'Server-side wording' })]),
    })
    // A brand new store is exactly what a reload produces: nothing in memory,
    // nothing persisted, everything recovered from the server.
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A)

    const identity = store.getRecord(QUERY_A).identity
    expect(identity.status).toBe('saved')
    if (identity.status !== 'saved') throw new Error('unreachable')
    expect(identity.primary.dir_id).toBe('reviewer-dir-1')
    expect(identity.primary.label).toBe('Server-side wording')
    expect(identity.primary.seed_query_id).toBe(QUERY_A)
    expect(identity.confirmedBy).toBe('server-list')
  })

  it('lists every historical group for the seed, oldest first', async () => {
    installFetch({
      get: () =>
        jsonResponse([
          dirFixture({ dir_id: 'newer', created_at: '2026-08-27 00:00:00' }),
          dirFixture({ dir_id: 'older', created_at: '2026-08-26 00:00:00' }),
        ]),
    })
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A)

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    // Legacy duplicates are preserved and surfaced; the primary matches the
    // backend's own oldest-record choice rather than response order.
    expect(identity.dirs.map((dir) => dir.dir_id)).toEqual(['older', 'newer'])
    expect(identity.primary.dir_id).toBe('older')
  })

  it('reports a failed lookup as unresolved, never as "not saved"', async () => {
    installFetch({ get: () => jsonResponse({ detail: 'boom' }, 500) })
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A)

    const identity = store.getRecord(QUERY_A).identity
    expect(identity.status).toBe('unresolved')
    expect(identity.status === 'unresolved' && identity.error).toBe('boom')
  })

  it('does not repeat a resolved lookup, and retries an unresolved one', async () => {
    installFetch({ get: () => jsonResponse({ detail: 'boom' }, 500) })
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A)
    await store.ensureLookup(QUERY_A)
    expect(getCalls()).toHaveLength(2)

    installFetch({ get: () => jsonResponse([dirFixture()]) })
    await store.ensureLookup(QUERY_A)
    await store.ensureLookup(QUERY_A)
    expect(getCalls()).toHaveLength(1)
  })
})

describe('confirmed creation is durable', () => {
  it('records the saved identity before the prediction refresh is broadcast', async () => {
    const order: string[] = []
    const store = new SavedDirectoryStore()
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, () => {
      // The refresh unmounts the creating component. By the time it fires the
      // identity has to be recorded somewhere that outlives it.
      order.push(`event:${store.getRecord(QUERY_A).identity.status}`)
    })

    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })
    order.push(`after:${store.getRecord(QUERY_A).identity.status}`)

    expect(order).toEqual(['event:saved', 'after:saved'])
  })

  it('survives a later failed refresh of the same query', async () => {
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })

    installFetch({ get: () => jsonResponse({ detail: 'network down' }, 503) })
    await store.ensureLookup(QUERY_A, { force: true })

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    expect(identity.refreshing).toBe(false)
    expect(identity.refreshError).toBe('network down')
    expect(identity.primary.dir_id).toBe('reviewer-dir-1')
  })

  it('is not erased by a stale empty lookup that resolves after the write', async () => {
    const pendingLookup = deferred<Response>()
    installFetch({
      get: () => pendingLookup.promise,
      post: () => jsonResponse(dirFixture(), 201),
    })
    const store = new SavedDirectoryStore()
    const lookup = store.ensureLookup(QUERY_A)
    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })
    // The lookup started before the write and answers with the pre-write world.
    pendingLookup.resolve(jsonResponse([]))
    await lookup

    expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
  })

  it('records a late completion against its own query, not the visible one', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise })
    const store = new SavedDirectoryStore()

    const creating = store.createDirectory(QUERY_A, { label: 'Unattested homily' })
    // The reviewer moves on to B (and back again, which a query-only guard
    // cannot tell apart from never having left).
    store.beginNaming(QUERY_B, 'New directory from query-12')
    pendingCreate.resolve(jsonResponse(dirFixture(), 201))
    await creating

    expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
    expect(store.getRecord(QUERY_B).identity.status).toBe('unknown')
    const creationB = store.getRecord(QUERY_B).creation
    expect(creationB.status).toBe('editing')
  })

  it('accepts a predictions response as confirmation but never as a denial', async () => {
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })

    // seeded_dirs is empty in every response for another model, and in any
    // response served from a cache that predates the write.
    store.observeSeededDirs(QUERY_A, [])
    expect(store.getRecord(QUERY_A).identity.status).toBe('saved')

    store.observeSeededDirs(QUERY_B, [dirFixture({ seed_query_id: QUERY_B })])
    expect(store.getRecord(QUERY_B).identity.status).toBe('saved')
  })

  it('drops everything when the account changes', async () => {
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })
    store.clear()
    expect(store.getRecord(QUERY_A).identity.status).toBe('unknown')
  })
})

describe('uncertain creation outcomes', () => {
  it('recovers the existing grouping when the seed is already taken', async () => {
    const existing = dirFixture({
      dir_id: 'reviewer-dir-9',
      label: "Somebody else's wording",
      created_by: 'James',
      status: 'matched',
      member_query_ids: [QUERY_A, 40],
    })
    installFetch({
      post: () => jsonResponse({ detail: 'Query 7 already seeds ...' }, 409),
      get: () => jsonResponse([existing]),
    })
    const store = new SavedDirectoryStore()

    const result = await store.createDirectory(QUERY_A, { label: 'My wording' })

    expect(result.outcome).toBe('recovered')
    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    // The stored label, creator and human-confirmed status, not the rejected
    // proposal and not an invented one-member 201.
    expect(identity.primary.label).toBe("Somebody else's wording")
    expect(identity.primary.created_by).toBe('James')
    expect(identity.primary.status).toBe('matched')
    expect(identity.confirmedBy).toBe('recovered')
    // The 409's prose is never parsed: the database answered the question.
    expect(getCalls()[0].url).toContain('seed_query_id=7')
  })

  it('keeps the proposed name when reconciliation proves nothing was written', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Rate limited' }, 429),
      get: () => jsonResponse([]),
    })
    const store = new SavedDirectoryStore()

    const result = await store.createDirectory(QUERY_A, { label: 'My wording' })

    expect(result).toEqual({ outcome: 'failed', error: 'Rate limited' })
    const creation = store.getRecord(QUERY_A).creation
    if (creation.status !== 'failed') throw new Error('expected failed')
    expect(creation.proposedLabel).toBe('My wording')
    expect(creation.outcome).toBe('not-created')
    expect(store.getRecord(QUERY_A).identity.status).toBe('absent')
  })

  it('refuses to guess when reconciliation also fails', async () => {
    installFetch({
      post: () => Promise.reject(new Error('Failed to fetch')),
      get: () => Promise.reject(new Error('Failed to fetch')),
    })
    const store = new SavedDirectoryStore()

    const result = await store.createDirectory(QUERY_A, { label: 'My wording' })

    expect(result.outcome).toBe('unresolved')
    const creation = store.getRecord(QUERY_A).creation
    if (creation.status !== 'failed') throw new Error('expected failed')
    expect(creation.outcome).toBe('unknown')
    expect(creation.proposedLabel).toBe('My wording')
    // A lost response is not proof of absence, so the identity stays open
    // rather than inviting a second permanent directory.
    expect(store.getRecord(QUERY_A).identity.status).toBe('unknown')
  })

  it('recovers a lost 201 response instead of creating a second directory', async () => {
    let posts = 0
    installFetch({
      post: () => {
        posts += 1
        return Promise.reject(new Error('Failed to fetch'))
      },
      // The write landed; only the response was lost.
      get: () => jsonResponse([dirFixture()]),
    })
    const store = new SavedDirectoryStore()

    const first = await store.createDirectory(QUERY_A, { label: 'My wording' })
    expect(first.outcome).toBe('recovered')

    const retry = await store.createDirectory(QUERY_A, { label: 'My wording' })
    expect(retry.outcome).toBe('already-saved')
    expect(posts).toBe(1)
  })

  it('broadcasts a recovery once, and not at all for an already-known group', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'conflict' }, 409),
      get: () => jsonResponse([dirFixture()]),
    })
    let events = 0
    const bump = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, bump)
    const store = new SavedDirectoryStore()

    await store.createDirectory(QUERY_A, { label: 'My wording' })
    expect(events).toBe(1)

    // Already saved: no POST, no second refresh, no loop.
    await store.createDirectory(QUERY_A, { label: 'My wording' })
    window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, bump)
    expect(events).toBe(1)
  })
})

describe('pending write controls', () => {
  it('sends one POST per attempt however often create is called', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise })
    const store = new SavedDirectoryStore()

    const first = store.createDirectory(QUERY_A, { label: 'My wording' })
    const second = await store.createDirectory(QUERY_A, { label: 'My wording' })
    expect(second).toEqual({ outcome: 'already-pending' })

    pendingCreate.resolve(jsonResponse(dirFixture(), 201))
    await first
    expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1)
  })

  it('cancels an unsent form but never claims to cancel an issued write', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise })
    const store = new SavedDirectoryStore()

    store.beginNaming(QUERY_A, 'New directory from query-7')
    expect(store.cancelNaming(QUERY_A)).toBe(true)
    expect(store.getRecord(QUERY_A).creation.status).toBe('idle')

    const creating = store.createDirectory(QUERY_A, { label: 'My wording' })
    expect(store.cancelNaming(QUERY_A)).toBe(false)
    expect(store.getRecord(QUERY_A).creation.status).toBe('pending')

    pendingCreate.resolve(jsonResponse(dirFixture(), 201))
    await creating
  })

  it('lets the form be dismissed without rollback, and stays quiet afterwards', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise, get: () => jsonResponse([]) })
    const store = new SavedDirectoryStore()

    const creating = store.createDirectory(QUERY_A, { label: 'My wording' })
    store.dismissPendingForm(QUERY_A)
    expect(store.getRecord(QUERY_A).creation.status).toBe('idle')

    pendingCreate.resolve(jsonResponse({ detail: 'nope' }, 500))
    await creating

    // The abandoned attempt does not re-open a form the reviewer closed...
    expect(store.getRecord(QUERY_A).creation.status).toBe('idle')
    // ...and its reconciliation still answers the question it was asked.
    expect(store.getRecord(QUERY_A).identity.status).toBe('absent')
  })

  it('keeps the reviewer-typed name across a re-open and a retry', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Rate limited' }, 429),
      get: () => jsonResponse([]),
    })
    const store = new SavedDirectoryStore()

    store.beginNaming(QUERY_A, 'New directory from query-7')
    store.setProposedLabel(QUERY_A, 'Council of Aachen, canon 4')
    await store.createDirectory(QUERY_A, { label: 'Council of Aachen, canon 4' })

    store.retryNaming(QUERY_A)
    const creation = store.getRecord(QUERY_A).creation
    if (creation.status !== 'editing') throw new Error('expected editing')
    expect(creation.proposedLabel).toBe('Council of Aachen, canon 4')

    // Re-opening the form must not silently reinstate the suggestion.
    store.beginNaming(QUERY_A, 'New directory from query-7')
    const reopened = store.getRecord(QUERY_A).creation
    if (reopened.status !== 'editing') throw new Error('expected editing')
    expect(reopened.proposedLabel).toBe('Council of Aachen, canon 4')
  })

  it('never touches feedback or an assessment outcome', async () => {
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'My wording' })
    await store.ensureLookup(QUERY_B)

    expect(calls.every((call) => !call.url.includes('/api/feedback'))).toBe(true)
    expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1)
  })
})
