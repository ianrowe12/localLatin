import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  SavedDirectoryProvider,
  useSavedDirectoryFor,
  useSavedDirectoryStore,
} from '../../contexts/SavedDirectoryContext'
import type { SavedDirectoryStore } from '../../contexts/savedDirectoryStore'
import {
  REVIEWER_DIRS_UPDATED_EVENT,
  type ReviewerDir,
} from '../../api/reviewerDirs'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider, usePredictionState } from '../../contexts/PredictionContext'
import NewDirectoryCta from './NewDirectoryCta'

/**
 * Another query's ranking, read through the SHARED prediction state (issue
 * #156) rather than through a second `usePredictions` cache of this suite's
 * own. The point of these two probes is that a cache built before the write
 * must not keep serving a ranking without the new directory in it, and testing
 * that against a private cache would prove nothing about the one the app uses.
 */
function OtherQueryCandidates({ queryId, model }: { queryId: number; model: string }) {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(queryId)
    setActiveModel(model)
  }, [queryId, model, setActiveQueryId, setActiveModel])
  const { phase, predictions } = usePredictionState()
  return (
    <span data-testid="other-query-candidates">
      {phase === 'loading' || phase === 'idle' ? 'loading' : String(predictions.length)}
    </span>
  )
}

/**
 * The seven invariants an independent review of this module found broken, kept
 * here as regressions.
 *
 * Each one is a way for the app to tell a reviewer something about a PERMANENT
 * row that is not true: that no directory was created when one was, that a name
 * is free when it is already stored, that a document sits in one grouping when
 * the database has two. They are grouped by the finding they answer so that a
 * later change which reintroduces one is named by its failure.
 *
 * Everything here drives the real component through the real store; only
 * `fetch` is stubbed.
 */

const MODEL = 'bowphs_LaTa'
const QUERY_A = 7
/** A query whose predictions are already cached when the write happens. */
const OTHER_QUERY = 12

function dirFixture(overrides: Partial<ReviewerDir> = {}): ReviewerDir {
  return {
    dir_id: 'reviewer-dir-1',
    label: 'Saved proposal',
    status: 'awaiting_match',
    seed_query_id: QUERY_A,
    member_query_ids: [QUERY_A],
    created_at: '2026-09-10 12:00:00',
    created_by: 'Reviewer one',
    model_slug: MODEL,
    variant: 'sif_abtt',
    best_match_score: null,
    has_potential_match: false,
    ...overrides,
  } as ReviewerDir
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

let posts = 0

function installFetch(routes: {
  get?: (seedQueryId: number) => Response | Promise<Response>
  post?: () => Response | Promise<Response>
}): void {
  posts = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if ((init?.method ?? 'GET') === 'POST') {
        posts += 1
        return routes.post ? routes.post() : jsonResponse(dirFixture(), 201)
      }
      const seed = Number(
        new URL(url, 'http://test.local').searchParams.get('seed_query_id'),
      )
      return routes.get ? routes.get(seed) : jsonResponse([])
    }),
  )
}

function renderCta() {
  return render(
    <SavedDirectoryProvider accountKey="reviewer-1">
      <NewDirectoryCta queryId={QUERY_A} model={MODEL} emphasised />
    </SavedDirectoryProvider>,
  )
}

/** Open the panel, name the directory, submit: the whole reviewer gesture. */
async function startCreation(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByTestId('new-directory-cta'))
  const field = screen.getByLabelText('Name the new directory')
  await user.clear(field)
  await user.type(field, 'Saved proposal')
  await user.click(screen.getByTestId('new-directory-submit'))
}

beforeEach(() => {
  installFetch({})
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('finding 1: closing the panel must not release the write', () => {
  it('keeps the write, the name and the lock when the panel is closed mid-flight', async () => {
    const user = userEvent.setup()
    const pending = deferred<Response>()
    installFetch({ post: () => pending.promise })
    renderCta()
    await startCreation(user)
    await screen.findByTestId('new-directory-pending-note')

    await user.click(screen.getByTestId('new-directory-cancel'))

    // Closed, but not forgotten: no Create button, because the record is not
    // absent -- a POST for this document is outstanding.
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(screen.queryByTestId('new-directory-submit')).toBeNull()
    expect(screen.getByTestId('new-directory-pending-closed')).toBeTruthy()

    await act(async () => {
      pending.resolve(jsonResponse(dirFixture(), 201))
    })
    // One gesture, one permanent directory.
    expect(posts).toBe(1)
    expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
  })

  it('a closed panel over an unknown outcome offers checking, never creating', async () => {
    const user = userEvent.setup()
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => (posts ? jsonResponse({ unreadable: true }) : jsonResponse([])),
    })
    renderCta()
    await startCreation(user)
    await screen.findByTestId('new-directory-outcome')

    await user.click(screen.getByTestId('new-directory-cancel'))

    // Closing over an unknown outcome leaves the question open, not answered.
    expect(screen.getByTestId('new-directory-pending-closed')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    expect(screen.queryByTestId('new-directory-submit')).toBeNull()
    expect(screen.getByTestId('new-directory-check-again')).toBeTruthy()
  })
})

describe('finding 2: an unsettled write withholds edit, retry and cancel', () => {
  it('will not post a second time while the first outcome is unknown', async () => {
    const user = userEvent.setup()
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => (posts ? jsonResponse({ unreadable: true }) : jsonResponse([])),
    })
    renderCta()
    await startCreation(user)
    await screen.findByTestId('new-directory-outcome')

    // The obvious reviewer reaction to an error is to press the button again.
    await user.click(screen.getByTestId('new-directory-submit'))
    await user.click(screen.getByTestId('new-directory-submit'))

    // A second POST here is a second permanent directory for one document.
    expect(posts).toBe(1)
  })

  it('disables the field and the submit button while the outcome is unknown', async () => {
    const user = userEvent.setup()
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => (posts ? jsonResponse({ unreadable: true }) : jsonResponse([])),
    })
    renderCta()
    await startCreation(user)
    await screen.findByTestId('new-directory-outcome')

    const field = screen.getByLabelText('Name the new directory') as HTMLInputElement
    expect(field.disabled).toBe(true)
    expect(
      (screen.getByTestId('new-directory-submit') as HTMLButtonElement).disabled,
    ).toBe(true)
    // The name is still the reviewer's own, and still theirs to come back to.
    expect(field.value).toBe('Saved proposal')

    await user.type(field, ' edited')
    expect(field.value).toBe('Saved proposal')
  })

  it('re-enables the form only once a clean lookup proves nothing was written', async () => {
    const user = userEvent.setup()
    let readable = false
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () =>
        posts && !readable ? jsonResponse({ unreadable: true }) : jsonResponse([]),
    })
    renderCta()
    await startCreation(user)
    await screen.findByTestId('new-directory-outcome')

    readable = true
    await user.click(screen.getByTestId('new-directory-check-again'))

    await waitFor(() => {
      expect(
        (screen.getByTestId('new-directory-submit') as HTMLButtonElement).disabled,
      ).toBe(false)
    })
    // Now, and only now, the retry is honest: the server has been read with
    // nothing in flight and this document has no directory.
    expect(screen.getByTestId('new-directory-outcome').textContent).toBeTruthy()
    await user.click(screen.getByTestId('new-directory-submit'))
    await waitFor(() => expect(posts).toBe(2))
  })
})

describe('finding 3: recovery language stays neutral', () => {
  it('does not claim nothing was created when the response was merely lost', async () => {
    const user = userEvent.setup()
    let saved = false
    installFetch({
      post: () => {
        // The write LANDS; the answer never arrives. Indistinguishable, from
        // here, from a refusal -- so the copy must not pick one.
        saved = true
        throw new Error('Failed to fetch')
      },
      get: () => jsonResponse(saved ? [dirFixture()] : []),
    })
    renderCta()
    await startCreation(user)

    const notice = await screen.findByTestId('new-directory-recovered')
    expect(notice.textContent).not.toContain('the name you proposed was not saved')
    expect(notice.textContent).not.toContain('nothing new was created')
    expect(notice.textContent).toContain('could not confirm')
  })
})

describe('findings 4 and 6: recovered groups and their status all reach the CTA', () => {
  it('lists every historical grouping and reports matched membership', async () => {
    const user = userEvent.setup()
    installFetch({
      post: () => jsonResponse({ detail: 'conflict' }, 409),
      get: () =>
        posts
          ? jsonResponse([
              dirFixture({ status: 'matched', member_query_ids: [QUERY_A, 9] }),
              dirFixture({
                dir_id: 'reviewer-dir-2',
                label: 'Historical second group',
                created_at: '2026-09-11 12:00:00',
                created_by: 'Reviewer two',
              }),
            ])
          : jsonResponse([]),
    })
    renderCta()
    await startCreation(user)

    await screen.findByTestId('new-directory-recovered')
    // A second grouping exists in the database. Showing only the first one
    // would tell the reviewer the duplicate they are looking at is not there.
    expect(screen.getByTestId('new-directory-other-groups').textContent).toContain(
      'Historical second group',
    )
    // And a human has already filed a second witness into the first one.
    expect(screen.getByTestId('matched-dir-badge')).toBeTruthy()
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Reviewer one',
    )
  })
})

describe('finding 5: negative lookup authority expires when a write begins', () => {
  it('does not let a lookup issued before the write declare the document absent', async () => {
    const staleLookup = deferred<Response>()
    let gets = 0
    let store!: SavedDirectoryStore
    let status = ''
    function Probe() {
      store = useSavedDirectoryStore()
      status = useSavedDirectoryFor(QUERY_A).identity.status
      return null
    }
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => {
        gets += 1
        return gets === 1 ? staleLookup.promise : jsonResponse({ unreadable: true })
      },
    })
    render(
      <SavedDirectoryProvider accountKey="reviewer-1">
        <Probe />
      </SavedDirectoryProvider>,
    )

    await act(async () => {
      await store.createDirectory(QUERY_A, { label: 'Saved proposal' })
    })
    expect(status).toBe('unresolved')

    // The empty answer was true when it was asked and is not true now: a write
    // has happened since. Treating it as current is what puts a Create button
    // in front of a reviewer whose directory already exists.
    await act(async () => {
      staleLookup.resolve(jsonResponse([]))
    })
    expect(store.getRecord(QUERY_A).identity.status).toBe('unresolved')
  })
})

describe('finding 7: identity rows are validated before they are believed', () => {
  it('refuses a listing seeded by another document', async () => {
    let store!: SavedDirectoryStore
    function Probe() {
      store = useSavedDirectoryStore()
      return null
    }
    installFetch({
      get: () =>
        jsonResponse([dirFixture({ seed_query_id: 12, member_query_ids: [12] })]),
    })
    render(
      <SavedDirectoryProvider accountKey="reviewer-1">
        <Probe />
      </SavedDirectoryProvider>,
    )

    await act(async () => {
      await store.ensureLookup(QUERY_A)
    })
    // Another document's grouping is not this document's grouping, and a
    // mislabelled acknowledgement is worse than an unanswered question.
    expect(store.getRecord(QUERY_A).identity.status).toBe('unresolved')
  })

  it('renders no acknowledgement at all for a partial row', async () => {
    installFetch({
      get: () =>
        jsonResponse([
          { dir_id: 'reviewer-dir-1', label: 'Saved proposal', seed_query_id: QUERY_A },
        ]),
    })
    renderCta()

    // The row has no creator and no status. The old code rendered it anyway,
    // as "Created by undefined"; the fix declines to describe it.
    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-saved')).toBeNull()
    expect(document.body.textContent).not.toContain('Created by undefined')
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })
})

describe('finding 8: a recovery that arrives late still invalidates other caches', () => {
  // A reviewer directory is a candidate for every OTHER query, so learning of
  // one has to reach the prediction caches. Two of the three ways this store
  // can learn of a write already broadcast that; the third -- a `Check again`
  // that succeeds after the immediate reconciliation failed -- did not, which
  // left an already-cached query rendering a candidate list built before the
  // directory existed, with nothing to invalidate it short of a reload.

  it('broadcasts once when Check again resolves an uncertain write', async () => {
    const user = userEvent.setup()
    let saved = false
    let lookupWorks = false
    let predictionGets = 0
    let events = 0
    const listener = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.includes('/predictions')) {
          predictionGets += 1
          return jsonResponse({
            file_id: OTHER_QUERY,
            filename: `query-${OTHER_QUERY}.txt`,
            model: MODEL,
            variant: 'sif_abtt',
            seeded_dirs: [],
            // The directory only becomes a candidate for the other query once
            // it exists on the server.
            predictions: saved
              ? [
                  {
                    rank: 11,
                    dir_name: 'reviewer-dir-1',
                    score: 0.8,
                    dir_files: [],
                    preview_text: '',
                    candidate_files: null,
                    source: 'reviewer',
                    label: 'Saved proposal',
                    seed_query_id: QUERY_A,
                  },
                ]
              : [],
          })
        }
        if ((init?.method ?? 'GET') === 'POST') {
          // The write LANDS and the response is lost.
          saved = true
          throw new Error('Failed to fetch')
        }
        if (!saved) return jsonResponse([])
        // The immediate reconciliation fails, so the attempt is left uncertain
        // and only an explicit Check again can settle it.
        return lookupWorks
          ? jsonResponse([dirFixture()])
          : jsonResponse({ detail: 'unavailable' }, 503)
      }),
    )

    try {
      render(
        <AppProvider>
          <SavedDirectoryProvider accountKey="reviewer-1">
            <PredictionProvider>
              <NewDirectoryCta queryId={QUERY_A} model={MODEL} emphasised />
              <OtherQueryCandidates queryId={OTHER_QUERY} model={MODEL} />
            </PredictionProvider>
          </SavedDirectoryProvider>
        </AppProvider>,
      )
      // The other query is cached before anything is created, with no reviewer
      // candidate in it. This is the cache that must not be allowed to persist.
      await waitFor(() => {
        expect(screen.getByTestId('other-query-candidates').textContent).toBe('0')
      })
      const cachedGets = predictionGets

      await startCreation(user)
      await screen.findByTestId('new-directory-outcome')
      expect(events).toBe(0)

      lookupWorks = true
      await user.click(screen.getByTestId('new-directory-check-again'))
      await screen.findByTestId('new-directory-saved')

      // The acknowledgement is not the whole job: the rest of the app has to
      // hear about it too.
      expect(events).toBe(1)
      await waitFor(() => {
        expect(screen.getByTestId('other-query-candidates').textContent).toBe('1')
      })
      expect(predictionGets).toBeGreaterThan(cachedGets)
    } finally {
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    }
  })

  it('does not broadcast on a routine lookup or a seeded observation', async () => {
    let store!: SavedDirectoryStore
    let events = 0
    const listener = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    function Probe() {
      store = useSavedDirectoryStore()
      return null
    }
    installFetch({ get: () => jsonResponse([dirFixture()]) })

    try {
      render(
        <SavedDirectoryProvider accountKey="reviewer-1">
          <Probe />
        </SavedDirectoryProvider>,
      )
      await act(async () => {
        await store.ensureLookup(QUERY_A)
        await store.ensureLookup(QUERY_A, { force: true })
        store.observeSeededDirs(QUERY_A, [dirFixture()])
      })

      expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
      // Nothing was written here. Broadcasting on an ordinary read would clear
      // every prediction cache, which refetches, which reads again.
      expect(events).toBe(0)
    } finally {
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    }
  })

  /**
   * A `seeded_dirs` observation is the fourth way this store can learn that its
   * write landed, and for a while it was the only one that did nothing about
   * it. It recorded the identity silently, which was right for an ordinary
   * refresh and wrong for a write still waiting on an answer: the CTA's saved
   * branch then hid `Check again`, the identity was no longer unknown so
   * nothing would look it up again, and the refresh other queries were owed had
   * no remaining path to reach them. The acknowledgement was on screen and the
   * rest of the app never heard.
   *
   * These drive the real CTA. Where the reviewer has a control, the control is
   * clicked; where the evidence arrives on its own, it arrives through the
   * public observation API, as a predictions response does.
   */
  it.each([
    ['while the POST is still in flight', 'in-flight'],
    ['after the POST failed and reconciliation recovered', 'recovered'],
    ['after both requests have failed', 'both-failed'],
  ])(
    'a seeded observation %s completes the owed refresh exactly once',
    async (_name, phase) => {
      const user = userEvent.setup()
      let store!: SavedDirectoryStore
      let events = 0
      const snapshots: string[] = []
      const listener = () => {
        events += 1
        snapshots.push(store.getRecord(QUERY_A).identity.status)
      }
      window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
      function Probe() {
        store = useSavedDirectoryStore()
        return null
      }
      const pendingPost = deferred<Response>()
      installFetch({
        post: () => {
          if (phase === 'in-flight') return pendingPost.promise
          if (phase === 'recovered') {
            return jsonResponse({ detail: 'Gateway timeout' }, 504)
          }
          throw new Error('Failed to fetch')
        },
        get: () => {
          // Before the write the seed is free; after it, the reconciliation
          // either finds the row or cannot read the server at all.
          if (!posts) return jsonResponse([])
          return phase === 'recovered'
            ? jsonResponse([dirFixture()])
            : jsonResponse({ detail: 'unavailable' }, 503)
        },
      })

      try {
        render(
          <SavedDirectoryProvider accountKey="reviewer-1">
            <NewDirectoryCta queryId={QUERY_A} model={MODEL} emphasised />
            <Probe />
          </SavedDirectoryProvider>,
        )

        if (phase === 'recovered') {
          // The reconciliation finds the row itself, so the observation is the
          // late arrival and must add nothing.
          await startCreation(user)
          await screen.findByTestId('new-directory-saved')
          expect(events).toBe(1)
          await act(async () => {
            store.observeSeededDirs(QUERY_A, [dirFixture()])
          })
          expect(events).toBe(1)
          return
        }

        await startCreation(user)
        if (phase === 'both-failed') {
          // The reviewer is looking at Check again when the predictions payload
          // answers the question for them.
          await screen.findByTestId('new-directory-outcome')
          expect(screen.getByTestId('new-directory-check-again')).toBeTruthy()
        }
        expect(events).toBe(0)

        await act(async () => {
          store.observeSeededDirs(QUERY_A, [dirFixture()])
        })

        // One refresh, with the acknowledgement already readable, and no
        // half-finished write left behind a hidden control.
        expect(events).toBe(1)
        expect(snapshots).toEqual(['saved'])
        expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
        expect(store.getRecord(QUERY_A).creation.status).toBe('idle')
        expect(screen.queryByTestId('new-directory-check-again')).toBeNull()

        if (phase === 'in-flight') {
          // The write it was waiting for lands afterwards. It is the same
          // directory and it has already been announced.
          await act(async () => {
            pendingPost.resolve(jsonResponse(dirFixture(), 201))
          })
          await waitFor(() => {
            expect(posts).toBe(1)
          })
          expect(events).toBe(1)
          expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
        }

        // Nothing further from routine reads or repeated observations.
        await act(async () => {
          await store.ensureLookup(QUERY_A, { force: true })
          store.observeSeededDirs(QUERY_A, [dirFixture()])
        })
        expect(events).toBe(1)
      } finally {
        window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
      }
    },
  )

  it('refreshes a cached other query when only an observation confirms the write', async () => {
    // The reviewer's whole journey, end to end, with no direct store call
    // standing in for a control: the other query is cached before the write,
    // the write's response is lost, its reconciliation fails, and the only
    // thing that ever confirms the directory is a later predictions payload.
    const user = userEvent.setup()
    let saved = false
    let predictionGets = 0
    let store!: SavedDirectoryStore
    let events = 0
    const listener = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    function Probe() {
      store = useSavedDirectoryStore()
      return null
    }

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.includes('/predictions')) {
          predictionGets += 1
          return jsonResponse({
            file_id: OTHER_QUERY,
            filename: `query-${OTHER_QUERY}.txt`,
            model: MODEL,
            variant: 'sif_abtt',
            seeded_dirs: [],
            predictions: saved
              ? [
                  {
                    rank: 11,
                    dir_name: 'reviewer-dir-1',
                    score: 0.8,
                    dir_files: [],
                    preview_text: '',
                    candidate_files: null,
                    source: 'reviewer',
                    label: 'Saved proposal',
                    seed_query_id: QUERY_A,
                  },
                ]
              : [],
          })
        }
        if ((init?.method ?? 'GET') === 'POST') {
          saved = true
          throw new Error('Failed to fetch')
        }
        // The seed is free until the write; afterwards the server cannot be
        // read, so the app is left genuinely uncertain.
        return saved
          ? jsonResponse({ detail: 'lookup unavailable' }, 503)
          : jsonResponse([])
      }),
    )

    try {
      render(
        <AppProvider>
          <SavedDirectoryProvider accountKey="reviewer-1">
            <PredictionProvider>
              <NewDirectoryCta queryId={QUERY_A} model={MODEL} emphasised />
              <OtherQueryCandidates queryId={OTHER_QUERY} model={MODEL} />
              <Probe />
            </PredictionProvider>
          </SavedDirectoryProvider>
        </AppProvider>,
      )
      await waitFor(() => {
        expect(screen.getByTestId('other-query-candidates').textContent).toBe('0')
      })
      const cachedGets = predictionGets

      await user.click(await screen.findByTestId('new-directory-cta'))
      const field = screen.getByLabelText('Name the new directory')
      await user.clear(field)
      await user.type(field, 'Saved proposal')
      await user.click(screen.getByTestId('new-directory-submit'))
      await screen.findByTestId('new-directory-outcome')
      expect(events).toBe(0)

      // The seed's own predictions come back carrying the directory.
      await act(async () => {
        store.observeSeededDirs(QUERY_A, [dirFixture()])
      })

      expect(await screen.findByTestId('new-directory-saved')).toBeTruthy()
      expect(events).toBe(1)
      await waitFor(() => {
        expect(screen.getByTestId('other-query-candidates').textContent).toBe('1')
      })
      expect(predictionGets).toBeGreaterThan(cachedGets)
    } finally {
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    }
  })

  it('a late failure cannot reopen a write an observation already confirmed', async () => {
    const user = userEvent.setup()
    let store!: SavedDirectoryStore
    let events = 0
    const listener = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    function Probe() {
      store = useSavedDirectoryStore()
      return null
    }
    const pendingPost = deferred<Response>()
    installFetch({
      post: () => pendingPost.promise,
      // Everything after the write fails: the reconciliation cannot read the
      // server either.
      get: () => (posts ? jsonResponse({ detail: 'down' }, 503) : jsonResponse([])),
    })

    try {
      render(
        <SavedDirectoryProvider accountKey="reviewer-1">
          <NewDirectoryCta queryId={QUERY_A} model={MODEL} emphasised />
          <Probe />
        </SavedDirectoryProvider>,
      )
      await startCreation(user)
      await act(async () => {
        store.observeSeededDirs(QUERY_A, [dirFixture()])
      })
      expect(events).toBe(1)

      await act(async () => {
        pendingPost.resolve(jsonResponse({ detail: 'Gateway timeout' }, 504))
      })
      await waitFor(() => {
        expect(screen.getByTestId('new-directory-saved')).toBeTruthy()
      })

      // The attempt that failed is the same attempt the observation confirmed.
      // Its error must not reopen the form, take back the acknowledgement, or
      // spend a second notification.
      expect(store.getRecord(QUERY_A).creation.status).toBe('idle')
      expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
      expect(screen.queryByTestId('new-directory-form')).toBeNull()
      expect(screen.queryByTestId('new-directory-cta')).toBeNull()
      expect(events).toBe(1)
    } finally {
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    }
  })

  it('does not broadcast twice when a lookup and the POST find the same row', async () => {
    const pendingPost = deferred<Response>()
    let events = 0
    let store!: SavedDirectoryStore
    const listener = () => {
      events += 1
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    function Probe() {
      store = useSavedDirectoryStore()
      return null
    }
    installFetch({
      post: () => pendingPost.promise,
      get: () => jsonResponse([dirFixture()]),
    })

    try {
      render(
        <SavedDirectoryProvider accountKey="reviewer-1">
          <Probe />
        </SavedDirectoryProvider>,
      )
      let operation!: Promise<unknown>
      await act(async () => {
        operation = store.createDirectory(QUERY_A, { label: 'Saved proposal' })
        // A forced lookup overlaps the write and sees the row first.
        await store.ensureLookup(QUERY_A, { force: true })
      })
      await act(async () => {
        pendingPost.resolve(jsonResponse(dirFixture(), 201))
        await operation
      })

      // Two paths, one directory, one invalidation.
      expect(events).toBe(1)
      expect(store.getRecord(QUERY_A).identity.status).toBe('saved')
    } finally {
      window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    }
  })
})
