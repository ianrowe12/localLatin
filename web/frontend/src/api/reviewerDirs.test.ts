import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchReviewerDirs, type ReviewerDir } from './reviewerDirs'

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
