import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiUrl } from '../../api/client'
import type { StatsResponse } from '../../api/models'
import ExportControls from './ExportControls'

// Issue #270: the CCL key assignments download sits beside the raw export.

const STATS: StatsResponse = {
  total_queries: 4,
  reviewed_count: 1,
  skipped_count: 0,
  unreviewed_count: 3,
  unresolved_count: 0,
  feedback_count: 1,
  reviews_by_model: {},
  reviews_by_reviewer: { 'Alice Example': 1 },
  rank_distribution: {},
  outcome_distribution: {},
  recent_reviews: [],
  needs_attention: [],
  next_unreviewed_ids: [],
}

let opened: string[] = []

beforeEach(() => {
  opened = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } }),
    ),
  )
  vi.spyOn(window, 'open').mockImplementation((url?: string | URL) => {
    opened.push(String(url))
    return null
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('ExportControls', () => {
  it('downloads the resolved CCL key assignments, unfiltered', async () => {
    const user = userEvent.setup()
    render(<ExportControls stats={STATS} />)

    // A filter set for the raw export must not leak into the CCL file.
    await user.selectOptions(
      screen.getByLabelText('Filter export by reviewer'),
      'Alice Example',
    )
    await user.click(
      screen.getByRole('button', { name: 'Download CCL key assignments' }),
    )

    expect(opened).toEqual([apiUrl('/api/feedback/export/ccl_keys')])
  })

  it('keeps the raw export and its filters as they were', async () => {
    const user = userEvent.setup()
    render(<ExportControls stats={STATS} />)

    await user.selectOptions(
      screen.getByLabelText('Filter export by reviewer'),
      'Alice Example',
    )
    await user.click(screen.getByRole('button', { name: 'Export' }))

    expect(opened).toEqual([apiUrl('/api/feedback/export?reviewer=Alice+Example')])
  })
})
