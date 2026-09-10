import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import PredictionList from './PredictionList'
import { getReviewTourSteps, REVIEW_TOUR_STEPS } from '../onboarding/tourSteps'
import { PROVENANCE_TERMS } from '../../utils/documentProvenance'
import MatchPills from '../feedback/MatchPills'
import {
  alreadySeededNote,
  DIRECTORY_CREATION_COPY,
  noneOptionLabel,
  NO_MATCH_GUIDANCE,
} from '../../utils/reviewerDirectoryCopy'

/**
 * Directory guidance (issue #162).
 *
 * What creating a directory costs has to be legible before it is done: it is a
 * permanent write with no rename and no removal, it is not the assessment, and
 * Submit or Skip afterwards cannot take it back. These assert the wording
 * through the real list, form and API path rather than a mocked hook, so a
 * change that quietly drops one of those facts fails here.
 *
 * Creation *behaviour* -- the request, its pending controls, and how a saved
 * acknowledgement survives a failed refresh -- is issue #161's, and is asserted
 * in its own files.
 */

const MODEL = 'bowphs_LaTa'
const QUERY_ID = 7

function modelsPayload() {
  return [
    {
      slug: MODEL,
      display_name: 'LaTa (T5)',
      layer: 4,
      pooling: 'sif',
      prediction_count: 2238,
      available_variants: ['sif_abtt'],
      default_variant: 'sif_abtt',
      confidence_bands: { no_match: 0.5, verify: 0.7 },
      supports_reviewer_dirs: true,
    },
  ]
}

function modelCard(rank: number, score: number) {
  return {
    rank,
    dir_name: `candidate-${rank}`,
    score,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'candidate text' }],
    source: 'model' as const,
  }
}

const SEEDED_DIR = {
  dir_id: 'reviewer-dir-1',
  label: 'Unattested homily',
  status: 'awaiting_match' as const,
  seed_query_id: QUERY_ID,
  member_query_ids: [QUERY_ID],
  created_at: '2026-08-26 00:00:00',
  created_by: 'Abigail',
  model_slug: MODEL,
  variant: 'sif_abtt' as const,
  best_match_score: 0.31,
  has_potential_match: false,
}

let predictions: { predictions: unknown[]; seeded_dirs: unknown[] } = {
  predictions: [modelCard(1, 0.41)],
  seeded_dirs: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'POST' && url.includes('/api/reviewer_dirs')) {
        return jsonResponse({ ...SEEDED_DIR }, 201)
      }
      if (url.includes('/api/models')) return jsonResponse(modelsPayload())
      if (url.includes('/predictions')) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: 'query-7.txt',
          model: MODEL,
          variant: 'sif_abtt',
          ...predictions,
        })
      }
      return jsonResponse({})
    }),
  )
}

function Harness() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return <PredictionList />
}

function renderList() {
  return render(
    <AppProvider>
      <Harness />
    </AppProvider>,
  )
}

beforeEach(() => {
  predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [] }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('before creating a directory', () => {
  it('warns under the button that creation is permanent and is not the assessment', async () => {
    renderList()
    const caption = await screen.findByTestId('new-directory-caption')
    expect(caption.textContent).toMatch(/permanent/i)
    expect(caption.textContent).toMatch(/does not record your assessment/i)
  })

  it('carries the same warning above the no-match band', async () => {
    predictions = { predictions: [modelCard(1, 0.82)], seeded_dirs: [] }
    renderList()
    expect((await screen.findByTestId('new-directory-caption')).textContent).toBe(
      DIRECTORY_CREATION_COPY.buttonCaption,
    )
  })

  it('spells out permanence, independence and the alternative in the form', async () => {
    renderList()
    await userEvent.click(await screen.findByTestId('new-directory-cta'))

    const guidance = await screen.findByTestId('directory-creation-guidance')
    const text = guidance.textContent ?? ''
    // Saved on confirmation, with no way back.
    expect(text).toMatch(/saved the moment you confirm/i)
    expect(text).toMatch(/no rename/i)
    expect(text).toMatch(/no removal/i)
    // A separate decision from the assessment, in both directions.
    expect(text).toMatch(/not an assessment/i)
    expect(text).toMatch(/submitting or skipping .* does not undo it/i)
    // Rejecting the ranked candidates does not require creating anything.
    expect(text).toMatch(/without starting a grouping/i)
    expect(text).toMatch(/None option/i)
  })

  it('promises no later editing', async () => {
    renderList()
    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    const guidance = await screen.findByTestId('directory-creation-guidance')
    // The absence of rename/removal is stated as the current state of the app,
    // not softened into a feature that is on its way.
    expect(guidance.textContent).not.toMatch(
      /rename it later|edit it later|can be renamed|for now|yet|coming/i,
    )
  })
})

describe('after a directory is saved', () => {
  it('names it and offers only conditional availability', async () => {
    renderList()
    await userEvent.click(await screen.findByTestId('new-directory-cta'))
    await userEvent.click(screen.getByTestId('new-directory-submit'))

    const saved = await screen.findByTestId('new-directory-created')
    expect(saved.textContent).toContain('Unattested homily')
    expect(saved.textContent).toMatch(/not on every one/i)
    // The old copy: "it is now a candidate for every other document". Scoring
    // excludes a directory's own members and any unscorable query, and caps how
    // many reviewer directories reach a list, so that was never true.
    expect(saved.textContent).not.toMatch(/every other document/i)
    expect(saved.textContent).toMatch(/does not undo it/i)
  })
})

describe('a document that already seeds a directory', () => {
  it('states the fact without attributing it to the reader', async () => {
    // seeded_dirs is global, not per-reviewer: the colleague who created it may
    // not be the one reading this.
    predictions = { predictions: [modelCard(1, 0.41)], seeded_dirs: [SEEDED_DIR] }
    renderList()

    const note = await screen.findByTestId('no-match-already-seeded')
    expect(note.textContent).not.toMatch(/you already started/i)
    expect(note.textContent).toMatch(/already seeds a provisional directory/i)
    // It must name the control the reviewer can actually press. This fixture
    // has one model candidate, so the singular is the honest form: the pills
    // are drawn from the real ranking, not from a top-ten assumption (#157).
    expect(note.textContent).toContain('“None of the 1 model candidate”')
    expect(note.textContent).not.toMatch(/None of top/i)
    expect(screen.queryByTestId('new-directory-cta')).toBeNull()
  })

  it('names the None control exactly as the assessment panel draws it', async () => {
    // The note tells a reviewer to go and press something, so the two surfaces
    // have to agree. `reviewerDirectoryCopy` cannot import the pill without a
    // leaf module reaching into a panel component, so the agreement is checked
    // against the real MatchPills here instead of assumed.
    for (const count of [1, 3, 10]) {
      const { unmount } = render(
        <MatchPills
          candidates={[]}
          canEvaluate
          selectedRanks={[]}
          unconfirmedRanks={[]}
          noneSelected={false}
          noneAvailable
          modelCandidateCount={count}
          onToggleCandidate={() => {}}
          onToggleNone={() => {}}
        />,
      )
      const pill = screen.getByTestId('match-pill-none')
      expect(noneOptionLabel(count)).toBe(pill.textContent)
      expect(alreadySeededNote(count)).toContain(`“${pill.textContent}”`)
      unmount()
    }
  })
})

describe('what a low score establishes', () => {
  it('separates "the model has no opinion" from "the source is not in the CCL"', async () => {
    renderList()
    const caveat = await screen.findByTestId('no-match-caveat')
    expect(caveat.textContent).toBe(NO_MATCH_GUIDANCE)
    expect(caveat.textContent).toMatch(/not evidence that the source is missing/i)
    expect(caveat.textContent).toMatch(/not expected to search the CCL/i)
  })

  it('does not tell the reviewer the fragment has no home in the corpus', async () => {
    renderList()
    const callout = await screen.findByTestId('no-match-callout')
    expect(callout.textContent).toContain('Potentially no match')
    expect(callout.textContent).not.toMatch(/belongs to a text that is not in the corpus/i)
  })
})

describe('the tour says the same things', () => {
  const stepsFor = (target: string) =>
    REVIEW_TOUR_STEPS.filter((step) => step.target === target)

  it('covers directory creation on a target that is always mounted', () => {
    // The creation button is absent above the band and absent again once this
    // document seeds a directory; TourOverlay skips a step whose target it
    // cannot find, so the guidance hangs off the list container instead.
    const step = REVIEW_TOUR_STEPS.find((s) => s.title === 'Starting a New Directory')
    expect(step).toBeTruthy()
    expect(step?.target).toBe('predictions')
    expect(getReviewTourSteps(false)).toContain(step)
    expect(getReviewTourSteps(true)).toContain(step)
  })

  it('repeats permanence, independence and the membership rule', () => {
    const text = stepsFor('predictions')
      .map((step) => step.description)
      .join(' ')
    expect(text).toMatch(/permanent/i)
    expect(text).toMatch(/no rename or removal/i)
    expect(text).toMatch(/does not undo it/i)
    expect(text).toMatch(/separate from your assessment/i)
    expect(text).toMatch(/second, distinct document/i)
    expect(text).toMatch(/never because a similarity crossed a threshold/i)
    expect(text).toMatch(/not evidence that the source is missing from the Carolingian Canon Law/i)
    expect(text).toMatch(/not being asked to search the CCL by hand/i)
  })

  it('keeps None independent of creating a directory', () => {
    const [step] = stepsFor('match-options')
    expect(step.description).toMatch(/needs no new directory/i)
    expect(step.description).toMatch(/not a claim that the CCL holds no match/i)
  })

  it('keeps uncertainty distinct from a definite non-match', () => {
    const [step] = stepsFor('submit-skip')
    expect(step.description).toMatch(/records uncertainty/i)
    expect(step.description).toMatch(/not the same judgement/i)
  })

  it('uses the panels\u2019 own provenance vocabulary', () => {
    const [candidate] = stepsFor('candidate-panel')
    expect(candidate.description).toContain(
      PROVENANCE_TERMS.labeled_reference.toLowerCase(),
    )
    expect(candidate.description).toContain(
      PROVENANCE_TERMS.reviewer_group.toLowerCase(),
    )
    const [query] = stepsFor('query-panel')
    expect(query.description).toContain(PROVENANCE_TERMS.query.toLowerCase())
  })

  it('promises candidate availability nowhere in the tour', () => {
    const text = REVIEW_TOUR_STEPS.map((step) => step.description).join(' ')
    expect(text).not.toMatch(/every other document/i)
  })
})

describe('creation copy does not drift between surfaces', () => {
  it('uses one module for the button, the form and the acknowledgement', async () => {
    renderList()
    const cta = await screen.findByTestId('new-directory-cta')
    expect(cta.textContent).toBe(DIRECTORY_CREATION_COPY.openEmphasised)

    await userEvent.click(cta)
    expect(screen.getByTestId('new-directory-submit').textContent).toBe(
      DIRECTORY_CREATION_COPY.submit,
    )
    expect(screen.getByLabelText(DIRECTORY_CREATION_COPY.fieldLabel)).toBeTruthy()

    await userEvent.click(screen.getByTestId('new-directory-submit'))
    await waitFor(() => {
      expect(screen.getByTestId('new-directory-created')).toBeTruthy()
    })
    expect(screen.getByTestId('new-directory-created').textContent).toContain(
      DIRECTORY_CREATION_COPY.savedAvailability,
    )
  })
})
