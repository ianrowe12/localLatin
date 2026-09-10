import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  SavedDirectoryProvider,
  useSavedDirectoryFor,
  useSavedDirectoryStore,
} from '../../contexts/SavedDirectoryContext'
import type { SavedDirectoryStore } from '../../contexts/savedDirectoryStore'
import type { ReviewerDir } from '../../api/reviewerDirs'
import NewDirectoryCta from './NewDirectoryCta'

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
