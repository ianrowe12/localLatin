import { useEffect } from 'react'
import { act, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { FeedbackProvider, useFeedback } from './FeedbackContext'
import type { Prediction } from '../api/queries'
import type { PredictionVariant } from '../api/variants'

const QUERY_ID = 3
const MODEL = 'bowphs_LaTa'

const PREDICTIONS: Prediction[] = [
  {
    rank: 1,
    dir_name: 'Can.apost.42',
    score: 0.55,
    dir_files: ['a.txt'],
    preview_text: 'preview',
    candidate_files: null,
  },
]

let posted: Record<string, unknown>[] = []

function installFetch(): void {
  posted = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === 'POST' && init.body) {
        posted.push(JSON.parse(String(init.body)))
      }
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
}

/** Writes a draft under `variant`, then submits it under `variant`. */
function SubmitHarness({ variant }: { variant: PredictionVariant }) {
  const { updateDraft, submitFeedback } = useFeedback()
  useEffect(() => {
    updateDraft(QUERY_ID, MODEL, { correctRank: 1, notes: 'looks right' }, variant)
  }, [updateDraft, variant])
  const { drafts } = useFeedback()
  useEffect(() => {
    if (drafts.has(`${QUERY_ID}-${MODEL}-${variant}`)) {
      void submitFeedback(QUERY_ID, MODEL, PREDICTIONS, variant)
    }
  }, [drafts, submitFeedback, variant])
  return null
}

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('feedback carries the active variant', () => {
  it('records the reviewer’s variant on the submitted row', async () => {
    render(
      <FeedbackProvider>
        <SubmitHarness variant="abtt" />
      </FeedbackProvider>,
    )

    await waitFor(() => expect(posted.length).toBeGreaterThan(0))
    expect(posted[0]).toMatchObject({
      query_id: QUERY_ID,
      model_slug: MODEL,
      variant: 'abtt',
      outcome: 'matched_rank',
      correct_rank: 1,
      correct_dir: 'Can.apost.42',
    })
  })

  it('keys drafts per variant so answers never bleed across them', async () => {
    // Asserts the behaviour rather than the key format, so a key-format
    // refactor cannot quietly break isolation while the test still passes.
    let api: ReturnType<typeof useFeedback> | null = null
    function Capture() {
      api = useFeedback()
      return null
    }
    render(
      <FeedbackProvider>
        <Capture />
      </FeedbackProvider>,
    )

    await act(async () => {
      api!.updateDraft(QUERY_ID, MODEL, { correctRank: 4, notes: 'raw answer' }, 'raw')
    })

    expect(api!.getDraft(QUERY_ID, MODEL, 'raw')).toMatchObject({
      correctRank: 4,
      notes: 'raw answer',
    })
    // A different variant of the same query/model is a clean slate.
    expect(api!.getDraft(QUERY_ID, MODEL, 'sif')).toEqual({
      correctRank: null,
      notes: '',
    })
  })
})
