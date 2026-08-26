import { apiUrl } from './client'

/**
 * Client for the reviewer-created-directory endpoint (issue #95, "D2").
 *
 * The UI side of the loop ships first (issue #94): the no-match CTA is written
 * against this contract and must not break on a deployment whose backend does
 * not serve it yet. Hence the deliberate feature detection below rather than
 * `apiFetch`, which collapses every failure into a bare Error message.
 *
 *   POST /api/reviewer_dirs {query_file_id, label?}
 *     -> 201 {dir_id, label, status: 'awaiting_match'}
 */

export interface ReviewerDir {
  dir_id: string
  label: string
  status: string
}

export type CreateReviewerDirResult =
  | { status: 'created'; dir: ReviewerDir }
  /** Endpoint not deployed yet: show the "coming with the next update" state. */
  | { status: 'unavailable' }
  | { status: 'error'; message: string }

/** 404/405/501 all mean "this deployment has no such endpoint", not "failed". */
function isMissingEndpoint(status: number): boolean {
  return status === 404 || status === 405 || status === 501
}

export async function createReviewerDir(
  queryFileId: number,
  label?: string,
): Promise<CreateReviewerDirResult> {
  let res: Response
  try {
    res = await fetch(apiUrl('/api/reviewer_dirs'), {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        label ? { query_file_id: queryFileId, label } : { query_file_id: queryFileId },
      ),
    })
  } catch {
    return { status: 'error', message: 'Could not reach the server.' }
  }

  if (isMissingEndpoint(res.status)) return { status: 'unavailable' }

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const message =
      (body as { error?: { message?: string }; detail?: string } | null)?.error?.message ??
      (body as { detail?: string } | null)?.detail ??
      res.statusText ??
      'Request failed'
    return { status: 'error', message }
  }

  const dir = (await res.json().catch(() => null)) as ReviewerDir | null
  if (!dir || typeof dir.dir_id !== 'string') {
    // A 2xx that does not match the contract is the same practical situation
    // as no endpoint at all: nothing was created that the reviewer can use.
    return { status: 'unavailable' }
  }
  return { status: 'created', dir }
}
