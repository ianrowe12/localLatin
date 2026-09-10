import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import type { ReviewerDir } from '../../api/reviewerDirs'
import NewDirectoryCta from './NewDirectoryCta'

/**
 * Issue #161, at the component: the acknowledgement of a permanent write has to
 * outlive the render that produced it.
 *
 * These tests mount the real CTA inside a real provider and let it talk to a
 * stubbed `fetch`, so what is under test is the whole path a reviewer travels:
 * the seed lookup, the POST, the reconciliation after a failure, and what the
 * component says while each of those is unresolved. Nothing here mocks the
 * store; the point is that the two agree.
 */

const MODEL = 'bowphs_LaTa'
const QUERY_A = 7
const QUERY_B = 12

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
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

interface Routes {
  /** Answers GET /api/reviewer_dirs?seed_query_id=N, keyed by that N. */
  get?: (seedQueryId: number) => Response | Promise<Response>
  post?: (body: Record<string, unknown>) => Response | Promise<Response>
}

let requests: { url: string; method: string; body: unknown }[] = []

function installFetch(routes: Routes = {}): void {
  requests = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      const body = init?.body ? JSON.parse(String(init.body)) : null
      requests.push({ url, method, body })
      if (method === 'POST') {
        return routes.post
          ? routes.post(body as Record<string, unknown>)
          : jsonResponse(dirFixture(), 201)
      }
      const seed = Number(
        new URL(url, 'http://test.local').searchParams.get('seed_query_id'),
      )
      return routes.get ? routes.get(seed) : jsonResponse([])
    }),
  )
}

function reviewerDirRequests(method: string) {
  return requests.filter(
    (request) =>
      request.url.includes('/api/reviewer_dirs') && request.method === method,
  )
}

function renderCta(queryId = QUERY_A) {
  return render(
    <SavedDirectoryProvider accountKey="reviewer-1">
      <NewDirectoryCta
        queryId={queryId}
        model={MODEL}
        emphasised
        filename={`query-${queryId}.txt`}
      />
    </SavedDirectoryProvider>,
  )
}

/** The button only exists once the app knows the document has no directory. */
function openButton() {
  return screen.findByTestId('new-directory-cta')
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('NewDirectoryCta: before anything is known', () => {
  it('does not offer to create while the seed lookup is in flight', async () => {
    const pending = deferred<Response>()
    installFetch({ get: () => pending.promise })
    renderCta()

    expect(await screen.findByTestId('new-directory-checking')).toBeTruthy()
    // Offering Create here is how a reviewer is invited to duplicate a
    // directory that already exists.
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()

    await act(async () => {
      pending.resolve(jsonResponse([]))
    })
    expect(await openButton()).toBeTruthy()
  })

  it('offers Check again, not Create, when the lookup fails', async () => {
    installFetch({ get: () => jsonResponse({ detail: 'Service down' }, 503) })
    renderCta()

    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()

    let listing: ReviewerDir[] = []
    installFetch({ get: () => jsonResponse(listing) })
    listing = [dirFixture()]
    await userEvent.click(screen.getByTestId('new-directory-check-again'))

    // The retry finds the directory that was there all along.
    expect(await screen.findByTestId('new-directory-created')).toBeTruthy()
  })
})

describe('NewDirectoryCta: while the write is in flight', () => {
  it('disables the field and submit, and stops claiming the save can be cancelled', async () => {
    const pending = deferred<Response>()
    installFetch({ post: () => pending.promise })
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    await user.click(screen.getByTestId('new-directory-submit'))

    const submit = await screen.findByTestId('new-directory-submit')
    await waitFor(() => {
      expect(submit.hasAttribute('disabled')).toBe(true)
    })
    expect(
      screen.getByLabelText('Name the new directory').hasAttribute('disabled'),
    ).toBe(true)
    expect(screen.getByTestId('new-directory-cancel').textContent).toBe('Close')
    // No rollback is offered, because none exists: the row may already be
    // committed and neither table supports a removal.
    expect(screen.getByTestId('new-directory-pending-note').textContent).toContain(
      'does not cancel the save',
    )
    expect(screen.queryByText('Cancel')).toBeNull()

    await act(async () => {
      pending.resolve(jsonResponse(dirFixture(), 201))
    })
    expect(await screen.findByTestId('new-directory-created')).toBeTruthy()
  })

  it('posts once however many times submit is pressed', async () => {
    const pending = deferred<Response>()
    installFetch({ post: () => pending.promise })
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    const submit = screen.getByTestId('new-directory-submit')
    await user.click(submit)
    await user.click(submit)
    await user.click(submit)

    expect(reviewerDirRequests('POST')).toHaveLength(1)
    await act(async () => {
      pending.resolve(jsonResponse(dirFixture(), 201))
    })
  })

  it('keeps the acknowledgement after Close dismisses the pending form', async () => {
    const pending = deferred<Response>()
    installFetch({ post: () => pending.promise })
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    await user.click(screen.getByTestId('new-directory-submit'))
    await user.click(await screen.findByTestId('new-directory-cancel'))
    expect(screen.queryByTestId('new-directory-form')).toBeNull()

    await act(async () => {
      pending.resolve(jsonResponse(dirFixture(), 201))
    })
    // Closing the form is not abandoning the write, so its result still has to
    // be reported.
    expect(await screen.findByTestId('new-directory-created')).toBeTruthy()
  })
})

describe('NewDirectoryCta: when the write does not obviously succeed', () => {
  it('keeps the proposed name and says nothing was created when the server confirms that', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Database is locked' }, 500),
      get: () => jsonResponse([]),
    })
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    const field = screen.getByLabelText('Name the new directory')
    await user.clear(field)
    await user.type(field, 'Council of Meaux fragment')
    await user.click(screen.getByTestId('new-directory-submit'))

    const error = await screen.findByTestId('new-directory-error')
    // The server's own words, on their own: nothing else may be folded into
    // this node, since other suites assert its exact text.
    expect(error.textContent).toBe('Database is locked')
    expect(screen.getByTestId('new-directory-outcome').textContent).toContain(
      'Nothing was created',
    )
    // Retyping a name is how a second directory ends up under a near-duplicate
    // of the first one's label.
    expect(
      (screen.getByLabelText('Name the new directory') as HTMLInputElement).value,
    ).toBe('Council of Meaux fragment')
    expect(
      screen.getByTestId('new-directory-submit').hasAttribute('disabled'),
    ).toBe(false)
  })

  it('refuses to guess when neither the write nor the check resolves', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
      get: () => jsonResponse({ detail: 'Gateway timeout' }, 504),
    })
    const user = userEvent.setup()
    renderCta()

    // The initial lookup fails too, so the reviewer starts from Check again.
    await user.click(await screen.findByTestId('new-directory-check-again'))
    expect(await screen.findByTestId('new-directory-unresolved')).toBeTruthy()
  })

  it('reports a recovery honestly instead of dressing it up as a creation', async () => {
    const existing = dirFixture({
      label: 'Stored under another name',
      created_by: 'Another reviewer',
    })
    let seeded: ReviewerDir[] = []
    installFetch({
      get: () => jsonResponse(seeded),
      post: () => {
        seeded = [existing]
        return jsonResponse({ detail: 'Query 7 already seeds a directory' }, 409)
      },
    })
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    const field = screen.getByLabelText('Name the new directory')
    await user.clear(field)
    await user.type(field, 'My proposed name')
    await user.click(screen.getByTestId('new-directory-submit'))

    const notice = await screen.findByTestId('new-directory-created')
    // The stored label, not the one this reviewer typed.
    expect(notice.textContent).toContain('Stored under another name')
    expect(notice.textContent).not.toContain('My proposed name')
    // Neutral about authorship: this evidence is also what a lost 201 looks
    // like, so the app must not tell the reviewer their write was refused.
    const recovered = screen.getByTestId('new-directory-recovered').textContent ?? ''
    expect(recovered).toContain('could not confirm')
    expect(recovered).not.toContain('nothing new was created')
    expect(recovered).not.toContain('was not saved')
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Created by Another reviewer',
    )
  })
})

describe('NewDirectoryCta: durability', () => {
  it('survives a remount, which is what a creation refresh does to the list', async () => {
    // The provider sits above the list, so it stays put; the CTA is what the
    // list's loading branch unmounts while the creation's own refresh is in
    // flight. That refresh is allowed to fail here, as it does in issue #161.
    function Harness({ visible }: { visible: boolean }) {
      return (
        <SavedDirectoryProvider accountKey="reviewer-1">
          {visible ? (
            <NewDirectoryCta
              queryId={QUERY_A}
              model={MODEL}
              emphasised
              filename="query-7.txt"
            />
          ) : (
            <p>Loading predictions…</p>
          )}
        </SavedDirectoryProvider>
      )
    }

    const user = userEvent.setup()
    const view = render(<Harness visible />)

    await user.click(await openButton())
    await user.click(screen.getByTestId('new-directory-submit'))
    expect(await screen.findByTestId('new-directory-created')).toBeTruthy()

    installFetch({ get: () => jsonResponse({ detail: 'Refresh failed' }, 503) })
    view.rerender(<Harness visible={false} />)
    expect(screen.queryByTestId('new-directory-created')).toBeNull()
    view.rerender(<Harness visible />)

    expect(screen.getByTestId('new-directory-created')).toBeTruthy()
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
    // A save is not re-litigated by a lookup that could only ever fail.
    expect(reviewerDirRequests('GET')).toHaveLength(0)
  })

  it('rehydrates from the server after a reload', async () => {
    installFetch({
      get: (seed) => jsonResponse(seed === QUERY_A ? [dirFixture()] : []),
    })
    renderCta()

    const notice = await screen.findByTestId('new-directory-created')
    expect(notice.textContent).toContain('Unattested homily')
    // Found rather than made here, so it is attributed rather than claimed.
    expect(screen.getByTestId('new-directory-attribution').textContent).toContain(
      'Created by Abigail',
    )
    expect(screen.queryByTestId('new-directory-recovered')).toBeNull()
  })

  it('records a completion against its own query while another is on screen', async () => {
    const pending = deferred<Response>()
    installFetch({ post: () => pending.promise })
    const user = userEvent.setup()
    const view = renderCta(QUERY_A)

    await user.click(await openButton())
    await user.click(screen.getByTestId('new-directory-submit'))

    // The reviewer moves on before the server answers.
    view.rerender(
      <SavedDirectoryProvider accountKey="reviewer-1">
        <NewDirectoryCta
          queryId={QUERY_B}
          model={MODEL}
          emphasised
          filename={`query-${QUERY_B}.txt`}
        />
      </SavedDirectoryProvider>,
    )
    await act(async () => {
      pending.resolve(jsonResponse(dirFixture(), 201))
    })

    // Query 12 gets none of query 7's acknowledgement...
    expect(screen.queryByTestId('new-directory-created')).toBeNull()
    expect(await openButton()).toBeTruthy()

    // ...and query 7's is intact when the reviewer comes back to it.
    view.rerender(
      <SavedDirectoryProvider accountKey="reviewer-1">
        <NewDirectoryCta
          queryId={QUERY_A}
          model={MODEL}
          emphasised
          filename={`query-${QUERY_A}.txt`}
        />
      </SavedDirectoryProvider>,
    )
    expect(screen.getByTestId('new-directory-created')).toBeTruthy()
  })

  it('never touches the feedback endpoint', async () => {
    const user = userEvent.setup()
    renderCta()

    await user.click(await openButton())
    await user.click(screen.getByTestId('new-directory-submit'))
    await screen.findByTestId('new-directory-created')

    // Creating a directory is not an assessment, and the copy says so; the
    // request log has to agree with the copy.
    expect(requests.some((request) => request.url.includes('/feedback'))).toBe(
      false,
    )
  })
})
