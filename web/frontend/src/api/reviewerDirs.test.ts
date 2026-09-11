import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  createReviewerDir,
  fetchReviewerDirs,
  listsItsSeedAsMember,
  REVIEWER_DIRS_UPDATED_EVENT,
  type ReviewerDir,
} from './reviewerDirs'
import {
  HISTORICAL_SEED_QUERY_ID,
  HISTORICAL_SEEDLESS_DIR,
} from '../test/fixtures/historicalReviewerDir'

/**
 * Issue #161: what `GET /api/reviewer_dirs` is allowed to mean.
 *
 * Its answer decides whether a reviewer is offered the button that writes a
 * permanent, unremovable directory. An empty list is a real answer -- "no such
 * grouping, go ahead" -- so anything this client cannot read has to leave the
 * failure path intact instead of being rounded down to it. A proxy error page, a
 * truncated body or a future response shape all arrive here as a 200.
 */

const QUERY_ID = 7

function dirFixture(overrides: Partial<ReviewerDir> = {}): ReviewerDir {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Unattested homily',
    status: 'awaiting_match',
    seed_query_id: QUERY_ID,
    member_query_ids: [QUERY_ID],
    created_at: '2026-08-26 00:00:00',
    created_by: 'Abigail',
    model_slug: 'bowphs_LaTa',
    variant: 'sif_abtt',
    best_match_score: 0.31,
    has_potential_match: false,
    ...overrides,
  }
}

function stubBody(body: unknown, contentType = 'application/json'): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(typeof body === 'string' ? body : JSON.stringify(body), {
          status: 200,
          headers: { 'Content-Type': contentType },
        }),
    ),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('fetchReviewerDirs', () => {
  it('returns a well-formed list, empty or not', async () => {
    stubBody([])
    expect(await fetchReviewerDirs({ seedQueryId: QUERY_ID })).toEqual([])

    stubBody([dirFixture()])
    const dirs = await fetchReviewerDirs({ seedQueryId: QUERY_ID })
    expect(dirs).toHaveLength(1)
    expect(dirs[0]?.dir_id).toBe('reviewer-dir-1')
  })

  it.each([
    ['an object', { detail: 'ok' }],
    ['null', null],
    ['a number', 3],
  ])('rejects %s where a list belongs', async (_name, body) => {
    stubBody(body)
    await expect(fetchReviewerDirs({ seedQueryId: QUERY_ID })).rejects.toThrow(
      /could not be read/,
    )
  })

  it('rejects an entry with no identity rather than showing it as a grouping', async () => {
    stubBody([{ label: 'Unattested homily' }])
    await expect(fetchReviewerDirs({ seedQueryId: QUERY_ID })).rejects.toThrow(
      /incomplete record/,
    )

    stubBody([{ ...dirFixture(), seed_query_id: 4242 }])
    await expect(fetchReviewerDirs({ seedQueryId: QUERY_ID })).rejects.toThrow(
      /seeded by 4242/,
    )
  })

  it('never turns an unparseable 200 into an empty list', async () => {
    // A proxy error page served with a 200 reaches `res.json()` and throws
    // there, before this module's own check. Either way it must not resolve.
    stubBody('<html>502 Bad Gateway</html>', 'text/html')
    await expect(
      fetchReviewerDirs({ seedQueryId: QUERY_ID }),
    ).rejects.toBeInstanceOf(Error)
  })
})

/**
 * Review finding 11: what the server preserves, this client has to be able to
 * read.
 *
 * Before issue #160, creation wrote the directory row and its seed membership
 * row on a connection shared with the rest of the request, so an unrelated
 * commit in between could make the directory permanent and leave the membership
 * behind. The rows that resulted are served by the current API and refuse a
 * second create for the same seed, and nothing in this application can repair
 * or remove them. Requiring the membership of every row that comes back turned
 * a grouping a reviewer demonstrably has into an unreadable record with no way
 * forward: no acknowledgement, no Create (the server would 409 it), and a
 * re-check that could only fail the same way.
 */
describe('preserved directories with no seed membership', () => {
  it('reads back the exact row the current server serves for one', async () => {
    stubBody([HISTORICAL_SEEDLESS_DIR])
    const dirs = await fetchReviewerDirs({ seedQueryId: HISTORICAL_SEED_QUERY_ID })
    // Unchanged, in particular the empty member list: the client reports the
    // stored record rather than adding the membership row the server lost.
    expect(dirs).toEqual([HISTORICAL_SEEDLESS_DIR])
    expect(dirs[0]?.member_query_ids).toEqual([])
    expect(listsItsSeedAsMember(dirs[0]!)).toBe(false)
  })

  it('does not let one of them make a complete sibling unreadable', async () => {
    // Rows are validated one by one and a single failure rejects the whole
    // list, so treating the historical row as broken hid every grouping this
    // document has, not only the partial one.
    const sibling = dirFixture({
      dir_id: 'reviewer-dir-complete',
      status: 'matched',
      member_query_ids: [QUERY_ID, 9],
    })
    stubBody([HISTORICAL_SEEDLESS_DIR, sibling])
    const dirs = await fetchReviewerDirs({ seedQueryId: QUERY_ID })
    expect(dirs.map((dir) => dir.dir_id)).toEqual([
      'reviewer-dir-f12cec1516c0',
      'reviewer-dir-complete',
    ])
  })

  it('still refuses a row whose fields were never on the wire', async () => {
    // The other half of the distinction. `[]` is a legitimate stored value and
    // also what a permissive parser writes for a field that was absent, so
    // relaxing the membership demand must not become "accept every default".
    // A response body is raw JSON, so absence is visible here.
    const { member_query_ids: _members, ...noMembers } = dirFixture()
    stubBody([noMembers])
    await expect(fetchReviewerDirs({ seedQueryId: QUERY_ID })).rejects.toThrow(
      /incomplete record/,
    )
  })
})

describe('createReviewerDir', () => {
  function stubCreated(body: unknown, status = 201): void {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify(body), {
            status,
            headers: { 'Content-Type': 'application/json' },
          }),
      ),
    )
  }

  it('accepts a 201 that reports the document it was created for', async () => {
    stubCreated(dirFixture())
    const dir = await createReviewerDir({ query_file_id: QUERY_ID })
    expect(dir.dir_id).toBe('reviewer-dir-1')
  })

  /**
   * The atomic guarantee, asserted at the ONE place it is actually promised.
   *
   * Issue #160 made creation write the directory and its seed membership in a
   * single dedicated transaction, so a directory this client just created
   * reports its own seed as a member. A 201 that does not is the half-write
   * #160 fixed, happening now: the right answer is to treat the write as
   * unconfirmed and reconcile it against the database, not to acknowledge a row
   * whose membership nobody can vouch for. Read-back paths are held to no such
   * standard, because they legitimately return rows written before that fix.
   */
  it('refuses a 201 that omits the document it was created for', async () => {
    stubCreated({ ...dirFixture(), member_query_ids: [] })
    await expect(createReviewerDir({ query_file_id: QUERY_ID })).rejects.toThrow(
      /without the document that seeds it/,
    )
  })

  it('does not broadcast a refresh for a 201 it could not confirm', async () => {
    const seen = vi.fn()
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, seen)
    stubCreated({ ...dirFixture(), member_query_ids: [] })
    await expect(createReviewerDir({ query_file_id: QUERY_ID })).rejects.toThrow()
    window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, seen)
    expect(seen).not.toHaveBeenCalled()
  })
})
