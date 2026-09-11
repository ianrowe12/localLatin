const BASE = normalizeBase(
  import.meta.env.VITE_API_BASE_PATH || import.meta.env.BASE_URL || '/',
)

function normalizeBase(base: string): string {
  if (!base || base === '/') return ''
  return `/${base.replace(/^\/+|\/+$/g, '')}`
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${BASE}${normalizedPath}`
}

/**
 * How a request failed, at the coarsest level a caller can actually act on.
 *
 * - `http`: the server answered, and the answer was a failure. `status` is set.
 * - `network`: no answer at all (offline, DNS, CORS, aborted socket).
 * - `malformed`: a 2xx whose body is not the JSON this app can read.
 */
export type ApiErrorKind = 'http' | 'network' | 'malformed'

export interface ApiErrorInfo {
  kind: ApiErrorKind
  /** HTTP status for `kind: 'http'`, and for a `malformed` 2xx. Null otherwise. */
  status: number | null
  /** FastAPI domain code from `{error: {code, message}}`, when the route sends one. */
  code: string | null
  /** Server-supplied prose, or a neutral fallback. Never a raw body dump. */
  message: string
}

/**
 * The error every `apiFetch` rejection is (issue #156).
 *
 * The old client threw `new Error(message)`, which discarded the status and the
 * domain code. Callers could then only render one undifferentiated failure --
 * an expired session, a 500 and an unreachable server all reaching the reviewer
 * as the same silence. It still extends `Error` and still carries the same
 * `message`, so every existing `catch (err: Error)` keeps working unchanged;
 * the structure is additive.
 *
 * `message` is either the server's own `error.message` / string `detail`, or a
 * neutral fallback. A response body is never interpolated into it wholesale:
 * prediction and query bodies carry manuscript text, reviewer notes and
 * reviewer-created labels, and none of that belongs in an error string that
 * ends up on screen or in a log.
 */
export class ApiError extends Error implements ApiErrorInfo {
  readonly kind: ApiErrorKind
  readonly status: number | null
  readonly code: string | null

  constructor(info: ApiErrorInfo) {
    super(info.message)
    this.name = 'ApiError'
    this.kind = info.kind
    this.status = info.status
    this.code = info.code
  }

  get info(): ApiErrorInfo {
    return {
      kind: this.kind,
      status: this.status,
      code: this.code,
      message: this.message,
    }
  }
}

/** Structured view of anything a `catch` can receive, including non-Errors. */
export function toApiErrorInfo(err: unknown, fallback: string): ApiErrorInfo {
  if (err instanceof ApiError) return err.info
  if (err instanceof Error && err.message) {
    return { kind: 'network', status: null, code: null, message: err.message }
  }
  return { kind: 'network', status: null, code: null, message: fallback }
}

/**
 * FastAPI emits two failure shapes on this app: `{error: {code, message}}` from
 * the domain handlers in `web/app.py`, and `{detail: ...}` from `HTTPException`
 * and request validation. `detail` is a string for the former and a list of
 * validation objects for the latter, which is why it is only used when it is a
 * string -- `String(detail)` on the list would render `[object Object]`.
 */
function readErrorBody(body: unknown): { code: string | null; message: string | null } {
  if (typeof body !== 'object' || body === null) return { code: null, message: null }
  const record = body as Record<string, unknown>
  const error = record.error
  if (typeof error === 'object' && error !== null) {
    const detail = error as Record<string, unknown>
    return {
      code: typeof detail.code === 'string' ? detail.code : null,
      message: typeof detail.message === 'string' ? detail.message : null,
    }
  }
  if (typeof record.detail === 'string') {
    return { code: null, message: record.detail }
  }
  return { code: null, message: null }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(apiUrl(path), {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json', ...init?.headers },
      ...init,
    })
  } catch {
    // `fetch` rejects only when no response arrived at all.
    throw new ApiError({
      kind: 'network',
      status: null,
      code: null,
      message: 'Could not reach the server.',
    })
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const { code, message } = readErrorBody(body)
    throw new ApiError({
      kind: 'http',
      status: res.status,
      code,
      message: message || res.statusText || `Request failed (${res.status})`,
    })
  }

  try {
    return (await res.json()) as T
  } catch {
    throw new ApiError({
      kind: 'malformed',
      status: res.status,
      code: null,
      message: 'The server sent a response this app could not read.',
    })
  }
}
