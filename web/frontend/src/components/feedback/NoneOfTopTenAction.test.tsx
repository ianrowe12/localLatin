import { useState } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NoneOfTopTenAction from './NoneOfTopTenAction'
import MatchPills from './MatchPills'
import { assertReceipt } from '../../api/cclKey'
import { noneOptionLabel } from '../../utils/reviewerDirectoryCopy'
import type { AssessmentCandidate } from '../../contexts/assessmentEligibility'

/**
 * The blue "None of the top N" action and its optional CCL key (issue #196).
 *
 * Every branch, because each one tells the reviewer a different thing about a
 * permanent record: the key named a directory the labelled corpus holds, it
 * joined a group a colleague made, it started one, it could not start one, or
 * there was no key at all. A receipt that reported the wrong branch would be
 * worse than none.
 */

const QUERY_ID = 7
const MODEL = 'bowphs_LaTa'

let posted: { url: string; body: Record<string, unknown> }[] = []
let response: { status: number; body: unknown } = { status: 201, body: null }
let networkError: Error | null = null
let recordedCount = 0

function receipt(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 12,
    query_id: QUERY_ID,
    outcome: 'none_of_top_k',
    ccl_key: null,
    ccl_key_action: null,
    ccl_key_dir: null,
    ccl_key_rank: null,
    ...over,
  }
}

function installFetch(): void {
  posted = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (networkError) throw networkError
      posted.push({ url: String(input), body: JSON.parse(String(init?.body)) })
      return new Response(JSON.stringify(response.body), {
        status: response.status,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
}

/**
 * The panel's OWN wiring, in miniature: `open` lives above the component and
 * `onRecorded` closes the form, which is exactly what hid the missing receipt
 * the first time round (these tests used to hard-code `open` and pass a no-op
 * `onRecorded`, so the component was never rendered the way the panel renders
 * it). `NoneOfTopTenPanel.test.tsx` drives the real `FeedbackPanel`; this
 * harness keeps the branch-by-branch cases fast without lying about the
 * contract.
 */
function Harness({
  recorded = null,
  initiallyOpen = true,
  ...over
}: Partial<Parameters<typeof NoneOfTopTenAction>[0]> & {
  initiallyOpen?: boolean
}) {
  const [open, setOpen] = useState(initiallyOpen)
  return (
    <NoneOfTopTenAction
      queryId={QUERY_ID}
      model={MODEL}
      variant="sif_abtt"
      notes="the incipit is unrelated"
      available
      recorded={recorded}
      {...over}
      open={open}
      onRecorded={() => {
        recordedCount += 1
        setOpen(false)
      }}
      onChange={() => setOpen(true)}
    />
  )
}

function renderAction(
  over: Partial<Parameters<typeof NoneOfTopTenAction>[0]> & {
    initiallyOpen?: boolean
  } = {},
) {
  return render(<Harness {...over} />)
}

beforeEach(() => {
  recordedCount = 0
  networkError = null
  response = { status: 201, body: receipt() }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the blue action', () => {
  it('shows nothing until the reviewer presses the control', () => {
    renderAction({ initiallyOpen: false })
    expect(screen.queryByTestId('none-of-top-k-form')).toBeNull()
    expect(screen.queryByTestId('none-of-top-k-recorded')).toBeNull()
  })

  it('offers exactly one field and one submit', () => {
    renderAction()
    const form = screen.getByTestId('none-of-top-k-form')
    expect(form.querySelectorAll('input')).toHaveLength(1)
    expect(form.querySelectorAll('button')).toHaveLength(1)
    expect(screen.getByLabelText('CCL key of the source, if known')).toBeTruthy()
    expect(screen.getByTestId('none-of-top-k-submit').textContent).toBe(
      'Record this answer',
    )
  })

  it('says the field is optional and what a key is for', () => {
    renderAction()
    const form = screen.getByTestId('none-of-top-k-form')
    expect(form.textContent).toContain('Optional')
    expect(form.textContent).toContain('saved with your answer')
    // It never asks a general evaluator to go and search the CCL by hand.
    expect(form.textContent).not.toContain('search')
    // And it no longer explains grouping mechanics (issue #221): an evaluator
    // cannot see a group anywhere in this app, so the sentence described
    // something they had no way to observe. The receipt after Record still
    // names the branch the server took.
    expect(form.textContent).not.toContain('joins them')
    expect(form.textContent).not.toContain('first member')
  })

  it('cannot be submitted when the ranking cannot be rejected', () => {
    renderAction({ available: false })
    expect(
      (screen.getByTestId('none-of-top-k-submit') as HTMLButtonElement).disabled,
    ).toBe(true)
  })
})

describe('what it posts', () => {
  it('records the non-match alone when no key is typed', async () => {
    renderAction()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0].url).toContain('/api/feedback')
    expect(posted[0].body).toEqual({
      query_id: QUERY_ID,
      model_slug: MODEL,
      variant: 'sif_abtt',
      outcome: 'none_of_top_k',
      correct_rank: 0,
      correct_dir: null,
      notes: 'the incipit is unrelated',
    })
    expect(screen.getByTestId('none-of-top-k-receipt').textContent).toBe(
      'Recorded: none of the ranked candidates match.',
    )
    expect(recordedCount).toBe(1)
  })

  it('sends the key trimmed, and nothing else changes', async () => {
    response = {
      status: 201,
      body: receipt({
        ccl_key: 'CTOU.567.16',
        ccl_key_action: 'created_reviewer_dir',
        ccl_key_dir: 'reviewer-dir-abc',
      }),
    }
    renderAction()
    await userEvent.type(screen.getByTestId('ccl-key-input'), '  CTOU.567.16  ')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0].body.ccl_key).toBe('CTOU.567.16')
    expect(posted[0].body.correct_rank).toBe(0)
  })

  it('does not post twice while the first request is open', async () => {
    const gate: { release: (() => void) | null } = { release: null }
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        posted.push({ url: String(input), body: JSON.parse(String(init?.body)) })
        await new Promise<void>((resolve) => {
          gate.release = resolve
        })
        return new Response(JSON.stringify(receipt()), { status: 201 })
      }),
    )
    renderAction()
    const submit = screen.getByTestId('none-of-top-k-submit')
    await userEvent.click(submit)
    await waitFor(() => expect(posted).toHaveLength(1))
    expect((submit as HTMLButtonElement).disabled).toBe(true)
    gate.release?.()
  })
})

describe('the receipt names the branch the server took', () => {
  it.each([
    [
      'matched_labelled_dir',
      null,
      'Recorded: matches directory Can.apost.49, which the ranking did not offer. Nothing new was created.',
    ],
    [
      // The parenthetical is only true when the server says so: the key can
      // name a directory that was on screen all along.
      'matched_labelled_dir',
      1,
      'Recorded: matches directory Can.apost.49, which is in the shortlist at #1. Nothing new was created.',
    ],
    [
      'joined_reviewer_dir',
      null,
      'Recorded, and this document joined the group under Can.apost.49.',
    ],
    [
      'already_joined',
      null,
      'Recorded. This document was already in the group under Can.apost.49, so nothing was added to it.',
    ],
    [
      'created_reviewer_dir',
      null,
      'Recorded, and a group under Can.apost.49 now holds this document.',
    ],
  ] as const)('reports %s (rank %s)', async (action, rank, sentence) => {
    response = {
      status: 201,
      body: receipt({
        ccl_key: 'Can.apost.49',
        ccl_key_action: action,
        ccl_key_dir: 'Can.apost.49',
        ccl_key_rank: rank,
      }),
    }
    renderAction()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'Can.apost.49')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    // The form closed, exactly as the panel closes it, and the receipt is
    // still on screen -- which is the whole finding this file missed.
    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(screen.queryByTestId('none-of-top-k-form')).toBeNull()
    expect(note.textContent).toBe(sentence)
    expect(
      screen.getByTestId('none-of-top-k-recorded').getAttribute('data-action'),
    ).toBe(action)
  })

  it('explains a key that could not start a second group', async () => {
    response = {
      status: 201,
      body: receipt({
        ccl_key: 'CTOU.567.16',
        ccl_key_action: 'seed_taken',
        ccl_key_dir: null,
      }),
    }
    renderAction()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toContain('already starts a group of its own')
    expect(note.textContent).toContain('CTOU.567.16')
    // It names no directory: the key resolved to nothing, and the group that
    // blocked it is not the group the key names.
    expect(note.textContent).not.toContain('reviewer-dir-')
    // The assessment itself was still recorded, so the panel moves on.
    expect(recordedCount).toBe(1)
  })
})

describe('the recorded answer outlives the form', () => {
  it('keeps the receipt on screen after the panel closes the form', async () => {
    response = {
      status: 201,
      body: receipt({
        ccl_key: 'CTOU.567.16',
        ccl_key_action: 'created_reviewer_dir',
        ccl_key_dir: 'reviewer-dir-abc',
      }),
    }
    renderAction()
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'CTOU.567.16')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const recordedBlock = await screen.findByTestId('none-of-top-k-recorded')
    expect(recordedBlock.textContent).toContain('Your recorded answer')
    expect(recordedBlock.textContent).toContain(
      'a group under CTOU.567.16 now holds this document',
    )
    expect(screen.queryByTestId('none-of-top-k-form')).toBeNull()
    expect(screen.queryByTestId('ccl-key-input')).toBeNull()
  })

  it('shows a stored answer on a revisit instead of an empty form', () => {
    // A fresh mount with nothing recorded in this session: the answer comes
    // from the server's prefill. Without this the reviewer met a blank field
    // and a live Record button over an answer they had already given, in an
    // append-only log.
    renderAction({
      initiallyOpen: false,
      recorded: {
        key: 'CTOU.567.16',
        action: 'joined_reviewer_dir',
        dir: 'reviewer-dir-abc',
        rank: null,
      },
    })
    expect(screen.getByTestId('none-of-top-k-receipt').textContent).toBe(
      'Recorded, and this document joined the group under CTOU.567.16.',
    )
    expect(screen.queryByTestId('none-of-top-k-submit')).toBeNull()
  })

  it('reopens an empty form on Change, and says what recording again does', async () => {
    renderAction({
      initiallyOpen: false,
      recorded: {
        key: 'CTOU.567.16',
        action: 'joined_reviewer_dir',
        dir: 'reviewer-dir-abc',
        rank: null,
      },
    })
    expect(screen.getByTestId('none-of-top-k-recorded').textContent).toContain(
      'does not replace this one',
    )
    await userEvent.click(screen.getByTestId('none-of-top-k-change'))

    expect((screen.getByTestId('ccl-key-input') as HTMLInputElement).value).toBe('')
    // The stored answer is still true while another is being typed, and is
    // still on screen.
    expect(screen.getByTestId('none-of-top-k-previous').textContent).toContain(
      'joined the group under CTOU.567.16',
    )
  })

  it('prefers this session\'s answer to the one the prefill carried', async () => {
    response = {
      status: 201,
      body: receipt({
        ccl_key: 'Can.apost.49',
        ccl_key_action: 'matched_labelled_dir',
        ccl_key_dir: 'Can.apost.49',
      }),
    }
    renderAction({
      recorded: {
        key: 'CTOU.567.16',
        action: 'joined_reviewer_dir',
        dir: 'reviewer-dir-abc',
        rank: null,
      },
    })
    await userEvent.type(screen.getByTestId('ccl-key-input'), 'Can.apost.49')
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const note = await screen.findByTestId('none-of-top-k-receipt')
    expect(note.textContent).toContain('matches directory Can.apost.49')
    expect(note.textContent).not.toContain('CTOU.567.16')
  })
})

describe('failures are told apart', () => {
  it('promises an empty log only when the server refused before writing', async () => {
    response = {
      status: 422,
      body: { error: { code: 'RANKING_NOT_EVALUABLE', message: 'Not evaluable here.' } },
    }
    renderAction()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const error = await screen.findByTestId('none-of-top-k-error')
    expect(error.textContent).toContain('Not evaluable here.')
    expect(error.textContent).toContain('Nothing was recorded')
    expect(recordedCount).toBe(0)
  })

  it('refuses to guess after a 500', async () => {
    response = {
      status: 500,
      body: { error: { code: 'internal_error', message: 'Server error.' } },
    }
    renderAction()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const error = await screen.findByTestId('none-of-top-k-error')
    expect(error.textContent).toContain('cannot tell whether this was recorded')
    expect(recordedCount).toBe(0)
  })

  it('refuses to guess after a dropped connection', async () => {
    networkError = new TypeError('Failed to fetch')
    renderAction()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const error = await screen.findByTestId('none-of-top-k-error')
    expect(error.textContent).toContain('cannot tell whether this was recorded')
    expect(recordedCount).toBe(0)
  })

  it('treats an unreadable 201 as an outcome nobody knows', async () => {
    response = { status: 201, body: { ok: true } }
    renderAction()
    await userEvent.click(screen.getByTestId('none-of-top-k-submit'))

    const error = await screen.findByTestId('none-of-top-k-error')
    expect(error.textContent).toContain('not with a record this app can read')
    expect(screen.queryByTestId('none-of-top-k-receipt')).toBeNull()
    expect(recordedCount).toBe(0)
  })
})

describe('the receipt validator', () => {
  it('accepts a complete row and refuses a half one', () => {
    expect(assertReceipt(receipt()).id).toBe(12)
    expect(() => assertReceipt(null)).toThrow()
    expect(() => assertReceipt({ ...receipt(), outcome: 'matched_rank' })).toThrow()
    expect(() => assertReceipt({ ...receipt(), ccl_key_action: 'invented' })).toThrow()
  })
})

describe('the label on the control', () => {
  const candidates: AssessmentCandidate[] = [
    {
      rank: 1,
      dirName: 'candidate-1',
      source: 'model',
      label: null,
      usable: true,
      evidence: 'readable',
    },
  ]

  it('matches the copy module, which the tour and the notices read', () => {
    // Drift guard: `noneOptionLabel` is a second copy of this sentence, kept so
    // that a leaf module does not have to import an assessment component.
    render(
      <MatchPills
        candidates={candidates}
        canEvaluate
        selectedRanks={[]}
        unconfirmedRanks={[]}
        noneSelected={false}
        noneAvailable
        modelCandidateCount={1}
        onToggleCandidate={() => {}}
        onToggleNone={() => {}}
      />,
    )
    expect(screen.getByTestId('match-pill-none').textContent).toBe(noneOptionLabel(1))
    expect(noneOptionLabel(1)).toBe('None of the 1 candidate')
    expect(noneOptionLabel(10)).toBe('None of the top 10')
  })

  it('draws the ten rank buttons large, with room around them', () => {
    const ten: AssessmentCandidate[] = Array.from({ length: 10 }, (_, i) => ({
      ...candidates[0],
      rank: i + 1,
      dirName: `candidate-${i + 1}`,
    }))
    const { container } = render(
      <MatchPills
        candidates={ten}
        canEvaluate
        selectedRanks={[]}
        unconfirmedRanks={[]}
        noneSelected={false}
        noneAvailable
        modelCandidateCount={10}
        onToggleCandidate={() => {}}
        onToggleNone={() => {}}
      />,
    )
    // Prof. Firey, 15 September 2026: big buttons, generous space, and nothing
    // else in the panel but these ten and the blue action.
    const grid = container.firstElementChild as HTMLElement
    expect(grid.className).toContain('gap-2.5')
    expect(screen.getByTestId('match-pill-1').className).toContain('text-base')
    expect(screen.getByTestId('match-pill-1').className).toContain('py-3')
    expect(screen.getAllByRole('button')).toHaveLength(11)
  })
})
