import { describe, expect, it } from 'vitest'
import { handleAdminReview } from './adminReview'
import type { AuthUser } from '../api/auth'
import type { QueryListItem } from '../api/queries'

const admin: AuthUser = {
  id: 1, username: 'pi', display_name: 'Example Admin', role: 'pi_admin',
  approval_status: 'approved', must_change_password: false,
}
const queries: QueryListItem[] = Array.from({ length: 30 }, (_, index) => ({
  file_id: index, filename: `witness.${index}.txt`, text_preview: 'Text',
  review_status: 'reviewed', review_count: 1,
}))

describe('admin review mock handler', () => {
  it('supports filters, pagination, decisions, safety validation and the role gate', async () => {
    const path = '/api/feedback/review_queue'
    expect(handleAdminReview(path, undefined, null, queries)?.status).toBe(401)
    expect(handleAdminReview(path, undefined, { ...admin, role: 'reviewer' }, queries)?.status).toBe(403)
    const all = await handleAdminReview(`${path}?filter=all&page_size=20`, undefined, admin, queries)!.json()
    expect(all.total).toBe(27)
    expect(all.items).toHaveLength(20)
    expect(all.has_more).toBe(true)
    const saved = await handleAdminReview('/api/feedback/decisions/0', {
      method: 'POST', body: JSON.stringify({ ccl_key: 'CTOU.567.16', notes: 'Checked' }),
    }, admin, queries)!.json()
    expect(saved.ccl_key).toBe('CTOU.567.16')
    const decided = await handleAdminReview(`${path}?filter=decided`, undefined, admin, queries)!.json()
    expect(decided.total).toBe(1)
    expect(decided.items[0].status).toBe('agreed')
    expect(decided.items[0].reviewer_status).toBe('conflict')
    expect(handleAdminReview('/api/feedback/decisions/0', {
      method: 'POST', body: JSON.stringify({ ccl_key: '=1+1' }),
    }, admin, queries)?.status).toBe(422)
  })
})
