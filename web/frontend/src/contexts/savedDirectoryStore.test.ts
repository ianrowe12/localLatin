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
  const base = {
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
  } as ReviewerDir
  // A directory usually contains the document that seeded it -- since issue
  // #160 creation writes both rows in one transaction -- so moving the seed
  // moves the membership with it unless a case is deliberately setting both.
  // Deliberately NOT an invariant: see the preserved-partial-write cases.
  if (overrides.seed_query_id !== undefined && overrides.member_query_ids === undefined) {
    return { ...base, member_query_ids: [overrides.seed_query_id] }
  }
  return base
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
    // A lost response is not proof of absence, so the identity is marked
    // unresolved rather than inviting a second permanent directory: whatever
    // was believed before the attempt is now stale, because the write may well
    // have landed.
    expect(store.getRecord(QUERY_A).identity.status).toBe('unresolved')
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
    // Closing the panel closes the PANEL. The write is still out there, so the
    // record keeps it -- and keeps the name it was issued with (finding 1).
    expect(store.getRecord(QUERY_A).formOpen).toBe(false)
    expect(store.getRecord(QUERY_A).creation.status).toBe('pending')

    pendingCreate.resolve(jsonResponse({ detail: 'nope' }, 500))
    await creating

    // The abandoned attempt does not re-open a form the reviewer closed...
    expect(store.getRecord(QUERY_A).formOpen).toBe(false)
    // ...it settles where the evidence puts it, keeping the typed name...
    const settled = store.getRecord(QUERY_A).creation
    if (settled.status !== 'failed') throw new Error('expected failed')
    expect(settled.outcome).toBe('not-created')
    expect(settled.proposedLabel).toBe('My wording')
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

describe('an unreadable answer is a failure, not an empty one', () => {
  // The endpoint answers one question -- "does this document already have a
  // directory?" -- and an empty list is permission to write a permanent,
  // unremovable record. A body this client cannot parse is not that permission.
  const MALFORMED_BODIES: [string, unknown][] = [
    ['an object where a list belongs', { detail: 'ok' }],
    ['a bare null', null],
    ['a string', 'reviewer_dirs'],
    ['a list whose entry has no identity', [{ label: 'Unattested homily' }]],
    ['a list whose entry has no seed', [{ ...dirFixture(), seed_query_id: null }]],
  ]

  it.each(MALFORMED_BODIES)(
    'leaves the initial lookup unresolved when the server sends %s',
    async (_name, body) => {
      installFetch({ get: () => jsonResponse(body) })
      const store = new SavedDirectoryStore()
      await store.ensureLookup(QUERY_A)

      const identity = store.getRecord(QUERY_A).identity
      // Not `absent`: that would put a Create button in front of a reviewer on
      // the strength of a response nobody could read.
      expect(identity.status).toBe('unresolved')
      expect(identity.status === 'unresolved' && identity.error).toMatch(
        /could not be read/,
      )
    },
  )

  it('does not un-save a known grouping when a refresh comes back unreadable', async () => {
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'Unattested homily' })

    installFetch({ get: () => jsonResponse({ unexpected: true }) })
    await store.ensureLookup(QUERY_A, { force: true })

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    expect(identity.primary.dir_id).toBe('reviewer-dir-1')
    expect(identity.refreshError).toMatch(/could not be read/)
  })

  it('refuses to say "nothing was created" when reconciliation is unreadable', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => jsonResponse({ detail: 'Gateway timeout' }),
    })
    const store = new SavedDirectoryStore()
    store.beginNaming(QUERY_A, 'New directory from query-7')
    store.setProposedLabel(QUERY_A, 'Council of Aachen, canon 4')
    const result = await store.createDirectory(QUERY_A, {
      label: 'Council of Aachen, canon 4',
    })

    expect(result.outcome).toBe('unresolved')
    const record = store.getRecord(QUERY_A)
    if (record.creation.status !== 'failed') throw new Error('expected failed')
    // The write may well have landed, so the reviewer is told the truth and
    // keeps the name they chose rather than retyping it into a duplicate.
    expect(record.creation.outcome).toBe('unknown')
    expect(record.creation.proposedLabel).toBe('Council of Aachen, canon 4')
    expect(record.creation.error).toBe('Gateway timeout')
    // And the identity is open again: closing the form must not fall back to
    // whatever was believed before the write was attempted.
    expect(record.identity.status).toBe('unresolved')
  })

  it('still recognises a well-formed empty list as absence', async () => {
    installFetch({ get: () => jsonResponse([]) })
    const store = new SavedDirectoryStore()
    await store.ensureLookup(QUERY_A)

    expect(store.getRecord(QUERY_A).identity.status).toBe('absent')
  })
})

describe('known groupings are append-only', () => {
  // Review finding 4. Every source of directory rows here is a SNAPSHOT taken
  // at some moment: the predictions payload, the seed-filtered lookup, a 201.
  // None of them is a statement that what it omits was deleted, and reviewer
  // directories cannot be deleted at all. So evidence accumulates.

  it('keeps historical groupings and matched membership through a thinner snapshot', () => {
    const store = new SavedDirectoryStore()
    store.observeSeededDirs(QUERY_A, [
      dirFixture({ status: 'matched', member_query_ids: [QUERY_A, 9] }),
      dirFixture({
        dir_id: 'reviewer-dir-2',
        label: 'Historical second group',
        created_at: '2026-08-27 00:00:00',
      }),
    ])

    // A later refresh mentions only the first directory, un-matched.
    store.observeSeededDirs(QUERY_A, [dirFixture()])

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    expect(identity.dirs).toHaveLength(2)
    // `matched` is a human act -- somebody filed a second witness in here --
    // and no snapshot can undo it.
    expect(identity.primary.status).toBe('matched')
    expect(identity.primary.member_query_ids).toEqual([QUERY_A, 9])
  })

  it('does not let a lookup that was already in flight erase a newer observation', async () => {
    const slowLookup = deferred<Response>()
    installFetch({ get: () => slowLookup.promise })
    const store = new SavedDirectoryStore()

    const lookup = store.ensureLookup(QUERY_A)
    // The predictions payload arrives while the lookup is still out, carrying
    // more than the lookup will.
    store.observeSeededDirs(QUERY_A, [
      dirFixture({ status: 'matched', member_query_ids: [QUERY_A, 9] }),
      dirFixture({ dir_id: 'reviewer-dir-2', created_at: '2026-08-27 00:00:00' }),
    ])
    slowLookup.resolve(jsonResponse([dirFixture()]))
    await lookup

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    expect(identity.dirs).toHaveLength(2)
    expect(identity.primary.status).toBe('matched')
  })

  it('folds a later observation in beside a recovered directory', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'conflict' }, 409),
      get: () =>
        jsonResponse([
          dirFixture({ dir_id: 'reviewer-dir-2', created_at: '2026-08-27 00:00:00' }),
        ]),
    })
    const store = new SavedDirectoryStore()

    await store.createDirectory(QUERY_A, { label: 'My wording' })
    // A predictions payload then mentions a different, older grouping. Two
    // partial views of the same document; neither denies the other.
    store.observeSeededDirs(QUERY_A, [dirFixture()])

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    expect(identity.dirs.map((dir) => dir.dir_id)).toEqual([
      'reviewer-dir-1',
      'reviewer-dir-2',
    ])
    // How this reviewer got here does not change because a refresh landed.
    expect(identity.confirmedBy).toBe('recovered')
  })
})

describe('a stored empty member list is not a missing field', () => {
  /**
   * Review finding 11 and review finding 10 are the two halves of one
   * distinction, and both have to hold at the same time.
   *
   * A preserved pre-#160 partial write really does have no members, and the
   * server says so. A row that reached the shared prediction validator without
   * a `member_query_ids` field also arrives as `[]`, because that validator
   * supplies the default `web/models.py` declares. By the time either reaches
   * this store the VALUE is identical, so the only thing that can separate them
   * is the validator saying which one it invented.
   */
  const stored = (overrides: Partial<ReviewerDir> = {}) =>
    dirFixture({ member_query_ids: [], ...overrides })

  it('records a directory the server reports with no members', () => {
    const store = new SavedDirectoryStore()
    store.observeSeededDirs(QUERY_A, [stored()])

    const identity = store.getRecord(QUERY_A).identity
    if (identity.status !== 'saved') throw new Error('expected saved')
    // Recorded as it stands. Nothing here adds the seed to the member list: the
    // membership row does not exist and this client cannot create it.
    expect(identity.primary.member_query_ids).toEqual([])
    expect(identity.primary.created_by).toBe('Abigail')
  })

  // Every field the validator can supply, not just the ones whose substitute
  // looks odd. `best_match_score` and `has_potential_match` were the two that
  // got through review, because `null` and `false` are also what an ordinary
  // unmatched directory carries.
  const SUPPLIABLE = [
    'member_query_ids',
    'created_at',
    'created_by',
    'model_slug',
    'best_match_score',
    'has_potential_match',
  ] as const

  it.each(SUPPLIABLE.map((field) => [field]))(
    'refuses one whose %s the validator supplied',
    (field) => {
      const store = new SavedDirectoryStore()
      store.observeSeededDirs(QUERY_A, [stored({ defaulted_fields: [field] })])

      // Not evidence in either direction: no identity, and the question is
      // still open for the lookup to answer.
      expect(store.getRecord(QUERY_A).identity.status).toBe('unknown')
    },
  )

  // Each of these is the substitute's value, stated by the server. The point of
  // the marker is that the store can tell them apart; this is the half that
  // proves the refusal above is about provenance and not about the value.
  it.each([
    ['no members', { member_query_ids: [] }],
    ['nothing scored', { best_match_score: null }],
    ['no lead', { has_potential_match: false }],
  ])('records one the server really reports with %s', (_name, over) => {
    const store = new SavedDirectoryStore()
    store.observeSeededDirs(QUERY_A, [stored(over as Partial<ReviewerDir>)])

    expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
  })

  it('refuses the whole list when one row was filled in for the server', () => {
    const store = new SavedDirectoryStore()
    store.observeSeededDirs(QUERY_A, [
      stored(),
      dirFixture({ dir_id: 'reviewer-dir-2', defaulted_fields: ['created_by'] }),
    ])

    expect(store.getRecord(QUERY_A).identity.status).toBe('unknown')
  })

  it.each(SUPPLIABLE.map((field) => [field]))(
    'does not let a supplied %s settle or announce an unknown write',
    async (field) => {
      const events = vi.fn()
      window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, events)
      installFetch({
        post: () => jsonResponse({ detail: 'gateway' }, 502),
        get: () => jsonResponse({ detail: 'gateway' }, 502),
      })
      const store = new SavedDirectoryStore()
      const result = await store.createDirectory(QUERY_A, { label: 'My wording' })
      expect(result.outcome).toBe('unresolved')

      store.observeSeededDirs(QUERY_A, [stored({ defaulted_fields: [field] })])
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, events)

      // The write stays exactly as unresolved as it was. Settling it here would
      // take the one notification the recovery still owes and hide Check again
      // behind a saved identity nothing established.
      const record = store.getRecord(QUERY_A)
      expect(record.creation.status).toBe('failed')
      if (record.creation.status !== 'failed') throw new Error('expected failed')
      expect(record.creation.outcome).toBe('unknown')
      expect(record.creation.proposedLabel).toBe('My wording')
      expect(store.getRecord(QUERY_A).identity.status).not.toBe('saved')
      expect(events).not.toHaveBeenCalled()
    },
  )

  it('lets the real stored row settle and announce that same write', async () => {
    const events = vi.fn()
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, events)
    installFetch({
      post: () => jsonResponse({ detail: 'gateway' }, 502),
      get: () => jsonResponse({ detail: 'gateway' }, 502),
    })
    const store = new SavedDirectoryStore()
    await store.createDirectory(QUERY_A, { label: 'My wording' })

    store.observeSeededDirs(QUERY_A, [stored()])
    window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, events)

    expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
    expect(store.getRecord(QUERY_A).creation.status).toBe('idle')
    // Identity first, then exactly one refresh for the write this store owns.
    expect(events).toHaveBeenCalledTimes(1)
  })
})
