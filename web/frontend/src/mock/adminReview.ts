import type { AdminReviewItem, AdminReviewQueue, QueueFilter } from '../api/adminReview'
import type { AuthUser } from '../api/auth'
import type { QueryListItem } from '../api/queries'
import { unsafeTextError } from '../utils/inputSafety'

let items: AdminReviewItem[] | null = null
let nextId = 1

function response(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
}

function sampleItems(queries: QueryListItem[]): AdminReviewItem[] {
  return queries.slice(0, 27).map((query, index) => {
    const status = index % 3 === 0 ? 'conflict' : index % 3 === 1 ? 'needs_check' : 'agreed'
    const first = {
      reviewer: 'Example Reviewer A', model_slug: 'google_mt5-base', variant: 'sif_abtt',
      keys: status === 'needs_check' ? [] : ['CTOU.567.16'],
      none_of_these: false, unresolved: status === 'needs_check',
      notes: index % 2 === 0 ? 'Please check the closing words against the reference.' : '',
      timestamp: '2026-11-16T10:00:00Z',
    }
    return {
      xml_id: query.filename.replace(/\.txt$/, ''), query_id: query.file_id,
      status, reviewer_status: status, reviewer_count: status === 'conflict' ? 2 : 1,
      answers: status === 'conflict'
        ? [first, { ...first, reviewer: 'Example Reviewer B', keys: [], none_of_these: true, notes: 'The ending appears different.' }]
        : [first],
      deciding_answer: null,
    }
  })
}

function matches(item: AdminReviewItem, filter: string): boolean {
  if (filter === 'all') return true
  if (filter === 'decided') return item.deciding_answer !== null
  if (filter === 'with_notes') return item.answers.some((answer) => answer.notes.trim())
  if (filter === 'reviewed_once') return item.reviewer_status === 'agreed' && item.reviewer_count === 1
  return item.status === filter
}

export function handleAdminReview(
  url: string, init: RequestInit | undefined, user: AuthUser | null, queries: QueryListItem[],
): Response | null {
  const parsed = new URL(url, window.location.origin)
  const path = parsed.pathname
  if (!path.includes('/api/feedback/review_queue') && !path.includes('/api/feedback/decisions/')) return null
  if (!user) return response({ detail: 'Authentication required' }, 401)
  if (user.role !== 'pi_admin') return response({ detail: 'PI/admin access required' }, 403)
  items ??= sampleItems(queries)
  const decision = path.match(/\/api\/feedback\/decisions\/(\d+)$/)
  const detail = path.match(/\/api\/feedback\/review_queue\/(\d+)$/)
  if (decision || detail) {
    const queryId = Number((decision ?? detail)![1])
    const query = queries.find((entry) => entry.file_id === queryId)
    if (!query) return response({ detail: 'Query not found' }, 404)
    let item = items.find((entry) => entry.query_id === queryId)
    if (!item) {
      item = {
        xml_id: query.filename.replace(/\.txt$/, ''), query_id: queryId,
        status: 'unreviewed', reviewer_status: '', reviewer_count: 0, answers: [], deciding_answer: null,
      }
      if (decision) items.push(item)
    }
    if (detail) return response(item)
    if (init?.method !== 'POST') return response({ detail: 'Method not allowed' }, 405)
    const body = JSON.parse(String(init.body))
    const key = String(body.ccl_key ?? '').trim().replace(/\s+/g, ' ')
    const notes = String(body.notes ?? '')
    const unsafe = unsafeTextError(body.ccl_key, 'A CCL key') ?? unsafeTextError(notes, 'A decision note')
    if (unsafe) return response({ error: { code: 'UNSAFE_TEXT', message: unsafe } }, 422)
    if (Boolean(key) === Boolean(body.none_of_these) || key.length > 200 || notes.length > 10000) {
      return response({ detail: 'Choose a CCL key or explicitly choose none of these' }, 422)
    }
    item.deciding_answer = {
      id: nextId++, xml_id: item.xml_id, timestamp: new Date().toISOString(),
      reviewer_account_id: user.id, reviewer: user.display_name, ccl_key: key || null, notes: notes.trim(),
    }
    item.status = key ? 'agreed' : 'no_key'
    return response(item.deciding_answer, 201)
  }
  const filter = parsed.searchParams.get('filter') ?? 'conflict'
  const page = Number(parsed.searchParams.get('page') ?? 1)
  const pageSize = Number(parsed.searchParams.get('page_size') ?? 20)
  const filters: QueueFilter[] = ['all', 'conflict', 'needs_check', 'with_notes', 'reviewed_once', 'decided']
  if (!filters.includes(filter as QueueFilter) || !Number.isInteger(page) || page < 1 || !Number.isInteger(pageSize) || pageSize < 1 || pageSize > 100) {
    return response({ detail: 'Invalid queue filters or pagination' }, 422)
  }
  const counted = items.filter((item) => item.answers.length || item.deciding_answer)
  const selected = counted.filter((item) => matches(item, filter))
  const start = (page - 1) * pageSize
  const counts = { all: 0, conflict: 0, needs_check: 0, with_notes: 0, reviewed_once: 0, decided: 0 }
  for (const kind of filters) counts[kind] = counted.filter((item) => matches(item, kind)).length
  const result: AdminReviewQueue = {
    items: selected.slice(start, start + pageSize), counts, total: selected.length,
    page, page_size: pageSize, has_more: start + pageSize < selected.length,
  }
  return response(result)
}
