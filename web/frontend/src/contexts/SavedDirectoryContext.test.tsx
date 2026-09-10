
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { REVIEWER_DIRS_UPDATED_EVENT, type ReviewerDir } from '../api/reviewerDirs'
import {
  SavedDirectoryProvider,
  useSavedDirectoryFor,
  useSavedDirectoryStore,
} from './SavedDirectoryContext'
import { SavedDirectoryStore } from './savedDirectoryStore'

/**
 * Issue #161, mounted: the acknowledgement has to survive the component that
 * created it being unmounted, which is exactly what the creation's own refresh
 * broadcast does to `PredictionList` today.
 *
 * The harness below stands in for that mounting behaviour without editing the
 * externally-owned list: `visible` toggles the consumer the way the list's
 * loading branch does, and re-rendering a fresh `SavedDirectoryProvider` stands
 * in for a full page reload.
 */

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
    model_slug: 'bowphs_LaTa',
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

let requests: { url: string; method: string }[] = []

function installFetch(
  routes: {
    post?: () => Response | Promise<Response>
    get?: () => Response | Promise<Response>
  } = {},
): void {
  requests = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      requests.push({ url, method })
      if (method === 'POST') {
        return routes.post ? routes.post() : jsonResponse(dirFixture(), 201)
      }
      return routes.get ? routes.get() : jsonResponse([])
    }),
  )
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

/** Everything a reviewer-facing surface needs, in one testable widget. */
function DirectoryState({ queryId }: { queryId: number }) {
  const store = useSavedDirectoryStore()
  const record = useSavedDirectoryFor(queryId)
  const { identity, creation } = record
  return (
    <div>
      <div data-testid="identity">{identity.status}</div>
      <div data-testid="label">
        {identity.status === 'saved' ? identity.primary.label : ''}
      </div>
      <div data-testid="dir-id">
        {identity.status === 'saved' ? identity.primary.dir_id : ''}
      </div>
      <div data-testid="creation">{creation.status}</div>
      <div data-testid="proposed">
        {creation.status === 'editing' ||
        creation.status === 'pending' ||
        creation.status === 'failed'
          ? creation.proposedLabel
          : ''}
      </div>
      <button
        type="button"
        data-testid="create"
        onClick={() => {
          void store.createDirectory(queryId, { label: 'Council of Aachen, canon 4' })
        }}
      >
        Create directory
      </button>
    </div>
  )
}

function Harness({
  queryId,
  visible = true,
  accountKey = 'account-1',
  store,
}: {
  queryId: number
  visible?: boolean
  accountKey?: string | null
  store?: SavedDirectoryStore
}) {
  return (
    <SavedDirectoryProvider accountKey={accountKey} store={store}>
      {visible ? <DirectoryState queryId={queryId} /> : <p>loading</p>}
    </SavedDirectoryProvider>
  )
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('SavedDirectoryProvider', () => {
  it('resolves the seed from the server on mount', async () => {
    installFetch({ get: () => jsonResponse([dirFixture()]) })
    render(<Harness queryId={QUERY_A} />)

    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })
    expect(screen.getByTestId('label').textContent).toBe('Unattested homily')
    expect(requests[0].url).toContain('seed_query_id=7')
  })

  it('keeps the acknowledgement when the consumer is unmounted and remounted', async () => {
    const { rerender } = render(<Harness queryId={QUERY_A} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('absent')
    })

    await userEvent.click(screen.getByTestId('create'))
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })

    // What the creation's own refresh broadcast does to the prediction list.
    rerender(<Harness queryId={QUERY_A} visible={false} />)
    expect(screen.getByText('loading')).toBeTruthy()
    rerender(<Harness queryId={QUERY_A} visible />)

    expect(screen.getByTestId('identity').textContent).toBe('saved')
    expect(screen.getByTestId('dir-id').textContent).toBe('reviewer-dir-1')
    // No second lookup was needed to know it: identity is not request state.
    expect(requests.filter((r) => r.method === 'GET')).toHaveLength(1)
  })

  it('rehydrates from the database after a full reload', async () => {
    installFetch({ get: () => jsonResponse([dirFixture({ label: 'Saved earlier' })]) })
    // A fresh provider with a fresh store: nothing carried over in memory, and
    // nothing was written to browser storage to carry over.
    const { unmount } = render(<Harness queryId={QUERY_A} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })
    unmount()

    render(<Harness queryId={QUERY_A} />)
    await waitFor(() => {
      expect(screen.getByTestId('label').textContent).toBe('Saved earlier')
    })
  })

  it('does not claim a known saved grouping is unsaved when a refresh fails', async () => {
    const store = new SavedDirectoryStore()
    render(<Harness queryId={QUERY_A} store={store} />)
    await userEvent.click(screen.getByTestId('create'))
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })

    installFetch({ get: () => jsonResponse({ detail: 'Service unavailable' }, 503) })
    await act(async () => {
      await store.ensureLookup(QUERY_A, { force: true })
    })

    expect(screen.getByTestId('identity').textContent).toBe('saved')
    expect(screen.getByTestId('dir-id').textContent).toBe('reviewer-dir-1')
  })

  it('records a completion for the query it was started on, never the one on screen', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise })
    const store = new SavedDirectoryStore()
    const { rerender } = render(<Harness queryId={QUERY_A} store={store} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('absent')
    })
    await userEvent.click(screen.getByTestId('create'))

    // A -> B while the write is in flight.
    rerender(<Harness queryId={QUERY_B} store={store} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('absent')
    })
    await act(async () => {
      pendingCreate.resolve(jsonResponse(dirFixture(), 201))
      await Promise.resolve()
    })

    // B is not painted with A's success...
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('absent')
    })
    expect(screen.getByTestId('label').textContent).toBe('')

    // ...and going back to A finds the acknowledgement waiting.
    rerender(<Harness queryId={QUERY_A} store={store} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })
    expect(screen.getByTestId('label').textContent).toBe('Unattested homily')
  })

  it('issues one write for a double-click and keeps the proposed name pending', async () => {
    const pendingCreate = deferred<Response>()
    installFetch({ post: () => pendingCreate.promise })
    render(<Harness queryId={QUERY_A} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('absent')
    })

    const button = screen.getByTestId('create')
    await userEvent.click(button)
    await userEvent.click(button)

    expect(screen.getByTestId('creation').textContent).toBe('pending')
    expect(screen.getByTestId('proposed').textContent).toBe(
      'Council of Aachen, canon 4',
    )
    expect(requests.filter((r) => r.method === 'POST')).toHaveLength(1)

    await act(async () => {
      pendingCreate.resolve(jsonResponse(dirFixture(), 201))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })
  })

  it('recovers the existing grouping when the seed is already taken', async () => {
    installFetch({
      post: () => jsonResponse({ detail: 'Query 7 already seeds ...' }, 409),
      get: () =>
        jsonResponse([
          dirFixture({ dir_id: 'reviewer-dir-9', label: "James's grouping" }),
        ]),
    })
    render(<Harness queryId={QUERY_A} />)
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })

    // Nothing to create: the reviewer is shown the stored grouping rather than
    // being left in a failed naming form.
    expect(screen.getByTestId('label').textContent).toBe("James's grouping")
    expect(screen.getByTestId('creation').textContent).toBe('idle')
  })

  it('drops another account\'s state on sign-out', async () => {
    const store = new SavedDirectoryStore()
    const { rerender } = render(<Harness queryId={QUERY_A} store={store} />)
    await userEvent.click(screen.getByTestId('create'))
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })

    rerender(<Harness queryId={QUERY_A} accountKey={null} store={store} />)

    // Query ids are bare integers: what one account could see is no evidence
    // about what the next one can.
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).not.toBe('saved')
    })
  })

  it('broadcasts the refresh only after the identity is readable', async () => {
    const store = new SavedDirectoryStore()
    const seen: string[] = []
    const listener = () => {
      const identity = store.getRecord(QUERY_A).identity
      seen.push(identity.status)
    }
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)
    render(<Harness queryId={QUERY_A} store={store} />)
    await userEvent.click(screen.getByTestId('create'))
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })
    window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, listener)

    expect(seen).toEqual(['saved'])
  })

  it('submits no feedback and changes no assessment outcome', async () => {
    render(<Harness queryId={QUERY_A} />)
    await userEvent.click(screen.getByTestId('create'))
    await waitFor(() => {
      expect(screen.getByTestId('identity').textContent).toBe('saved')
    })

    expect(requests.some((r) => r.url.includes('/api/feedback'))).toBe(false)
    expect(requests.filter((r) => r.method === 'POST')).toHaveLength(1)
  })
})

describe('useSavedDirectory outside a provider', () => {
  it('fails loudly rather than silently reporting "not saved"', () => {
    function Orphan() {
      useSavedDirectoryStore()
      return null
    }
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<Orphan />)).toThrow(/SavedDirectoryProvider/)
    spy.mockRestore()
  })
})
