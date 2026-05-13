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

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: { message: res.statusText } }))
    throw new Error(err.error?.message ?? err.detail ?? res.statusText)
  }
  return res.json()
}
