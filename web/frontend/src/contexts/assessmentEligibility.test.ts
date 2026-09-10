import { describe, expect, it } from 'vitest'
import type { Prediction } from '../api/queries'
import {
  assessmentEvidence,
  candidateEvidence,
  candidateIsUsable,
  evaluationReadiness,
  expectedCandidateDirs,
  reviewSelections,
  toggleSelection,
  unconfirmedRanks,
  type AssessmentCandidate,
} from './assessmentEligibility'
import type { DraftSelection, FeedbackDraft } from './feedbackDraft'

/**
 * Eligibility rules, away from the DOM (issue #157).
 *
 * These mirror `web/routers/feedback.py`: a control is offered if and only if
 * the server would accept the save behind it. Where the two disagree, either a
 * reviewer is blocked from a real answer or the panel offers one that fails.
 */

function model(rank: number, over: Partial<Prediction> = {}): Prediction {
  return {
    rank,
    dir_name: `candidate-${rank}`,
    score: 1 - rank / 100,
    dir_files: [`${rank}.txt`],
    preview_text: 'preview',
    candidate_files: [{ filename: `${rank}.txt`, text: 'lorem ipsum' }],
    source: 'model',
    ...over,
  }
}

function reviewer(rank: number, over: Partial<Prediction> = {}): Prediction {
  return {
    ...model(rank),
    dir_name: `reviewer-dir-${rank}`,
    source: 'reviewer',
    label: 'Unattested homily',
    ...over,
  }
}

const READY = (predictions: Prediction[]) =>
  assessmentEvidence({ phase: 'ready', predictions, hasReviewerAccount: true })

describe('candidateIsUsable', () => {
  it('requires identity, a finite score and readable words on screen', () => {
    expect(candidateIsUsable(model(1))).toBe(true)
    expect(candidateIsUsable(model(1, { dir_name: '  ' }))).toBe(false)
    expect(candidateIsUsable(model(1, { score: Number.NaN }))).toBe(false)
    expect(candidateIsUsable(model(1, { score: Number.POSITIVE_INFINITY }))).toBe(false)
    expect(candidateIsUsable(model(1, { candidate_files: null }))).toBe(false)
    expect(
      candidateIsUsable(
        model(1, { candidate_files: [{ filename: 'a.txt', text: '   ' }] }),
      ),
    ).toBe(false)
  })

  it('asks about the witness this build shows, not the whole directory', () => {
    // CenterArea renders candidate_files[0] and there is no witness selector,
    // so a readable second file is text the reviewer is never shown. The
    // server would accept the save; this client must not offer it, because
    // offering it invites a judgement of evidence that never reached the
    // screen. Stricter than the server, in the one safe direction.
    const hidden = model(1, {
      candidate_files: [
        { filename: 'blank.txt', text: '' },
        { filename: 'readable.txt', text: 'incipit' },
      ],
    })
    expect(candidateIsUsable(hidden)).toBe(false)
    expect(candidateEvidence(hidden)).toBe('hidden_witness')
    // Told apart from a directory with nothing readable at all, so the copy
    // can state which fact it is.
    expect(candidateEvidence(model(1, { candidate_files: [] }))).toBe('no_text')
    expect(candidateEvidence(model(1))).toBe('readable')
    expect(candidateEvidence(model(1, { dir_name: ' ' }))).toBe('unidentified')
  })

  it('scores low candidates as usable: confidence is not eligibility', () => {
    expect(candidateIsUsable(model(1, { score: 0.01 }))).toBe(true)
  })
})

describe('assessmentEvidence', () => {
  it.each(['idle', 'loading', 'error', 'empty', 'excluded'] as const)(
    'offers no candidate and no evaluation while %s',
    (phase) => {
      const evidence = assessmentEvidence({
        phase,
        // Even if a stale array were handed in, an unsettled phase describes
        // no candidates: there are no phantom pills to press.
        predictions: [model(1), model(2)],
        hasReviewerAccount: true,
      })
      expect(evidence.candidates).toEqual([])
      expect(evidence.canEvaluate).toBe(false)
      expect(evidence.noneAvailable).toBe(false)
      expect(evidence.evidenceBlock).not.toBeNull()
    },
  )

  it('refuses evaluation with no signed-in reviewer', () => {
    const evidence = assessmentEvidence({
      phase: 'ready',
      predictions: [model(1)],
      hasReviewerAccount: false,
    })
    expect(evidence.evidenceBlock).toBe('no_reviewer')
    expect(evidence.canEvaluate).toBe(false)
  })

  it('keeps actual sparse ranks, including anchored reviewer directories', () => {
    const evidence = READY([model(1), reviewer(11), reviewer(12)])
    expect(evidence.candidates.map((candidate) => candidate.rank)).toEqual([1, 11, 12])
    expect(evidence.modelCandidates.map((candidate) => candidate.rank)).toEqual([1])
    expect(evidence.candidateAt(11)?.dirName).toBe('reviewer-dir-11')
    expect(evidence.candidateAt(2)).toBeNull()
  })

  it('disables None when one model candidate cannot be read, not the readable pills', () => {
    const evidence = READY([model(1), model(2, { candidate_files: [] })])
    expect(evidence.canEvaluate).toBe(true)
    expect(evidence.candidateAt(1)?.usable).toBe(true)
    expect(evidence.candidateAt(2)?.usable).toBe(false)
    expect(evidence.noneAvailable).toBe(false)
    expect(evidence.noneBlock).toBe('partial_model_evidence')
  })

  it('lets a blank reviewer extra neither supply nor withdraw model evidence', () => {
    const evidence = READY([model(1), reviewer(11, { candidate_files: [] })])
    expect(evidence.noneAvailable).toBe(true)
    expect(evidence.candidateAt(11)?.usable).toBe(false)
  })

  it('refuses every evaluation when no model candidate is readable', () => {
    const evidence = READY([
      model(1, { candidate_files: [] }),
      reviewer(11),
    ])
    expect(evidence.canEvaluate).toBe(false)
    expect(evidence.evidenceBlock).toBe('no_usable_model_evidence')
    expect(evidence.noneAvailable).toBe(false)
  })
})

describe('reviewSelections', () => {
  const evidence = READY([model(1), model(2), reviewer(11)])

  it('keeps a choice whose directory is still at its rank', () => {
    const selections: DraftSelection[] = [
      { rank: 11, dirName: 'reviewer-dir-11', source: 'reviewer' },
      { rank: 1, dirName: 'candidate-1', source: 'model' },
    ]
    const review = reviewSelections(selections, evidence)
    expect(review.changed).toBe(false)
    expect(review.confirmed.map((selection) => selection.rank)).toEqual([11, 1])
    expect(review.issues).toEqual([])
  })

  it('drops a choice whose rank now holds a different directory', () => {
    const review = reviewSelections(
      [{ rank: 11, dirName: 'reviewer-dir-99', source: 'reviewer' }],
      evidence,
    )
    expect(review.selections).toEqual([])
    expect(review.changed).toBe(true)
    expect(review.issues).toEqual([
      {
        kind: 'reassigned',
        rank: 11,
        dirName: 'reviewer-dir-99',
        nowDirName: 'reviewer-dir-11',
      },
    ])
  })

  it('drops a choice whose rank is no longer offered', () => {
    const review = reviewSelections(
      [{ rank: 7, dirName: 'candidate-7', source: 'model' }],
      evidence,
    )
    expect(review.selections).toEqual([])
    expect(review.issues[0].kind).toBe('vanished')
  })

  it('holds an identity-less choice for reconfirmation instead of binding it', () => {
    const review = reviewSelections([{ rank: 11, dirName: null, source: null }], evidence)
    expect(review.selections).toHaveLength(1)
    expect(review.confirmed).toEqual([])
    expect(review.changed).toBe(false)
    expect(review.issues[0]).toEqual({ kind: 'unverified', rank: 11 })
    expect(unconfirmedRanks(review)).toEqual([11])
  })

  it('keeps a choice that lost its text but does not let it be submitted', () => {
    const blanked = READY([model(1, { candidate_files: [] }), model(2)])
    const review = reviewSelections(
      [{ rank: 1, dirName: 'candidate-1', source: 'model' }],
      blanked,
    )
    expect(review.selections).toHaveLength(1)
    expect(review.confirmed).toEqual([])
    expect(review.issues[0]).toEqual({
      kind: 'unreadable',
      rank: 1,
      dirName: 'candidate-1',
      evidence: 'no_text',
    })
  })

  it('reports a choice whose text moved out of view as its own fact', () => {
    const hidden = READY([
      model(1, {
        candidate_files: [
          { filename: 'blank.txt', text: '' },
          { filename: 'readable.txt', text: 'incipit' },
        ],
      }),
      model(2),
    ])
    const review = reviewSelections(
      [{ rank: 1, dirName: 'candidate-1', source: 'model' }],
      hidden,
    )
    expect(review.confirmed).toEqual([])
    expect(review.issues[0]).toMatchObject({
      kind: 'unreadable',
      evidence: 'hidden_witness',
    })
  })

  it('treats a source change at the same rank as a different candidate', () => {
    const review = reviewSelections(
      [{ rank: 11, dirName: 'reviewer-dir-11', source: 'model' }],
      evidence,
    )
    expect(review.issues[0].kind).toBe('reassigned')
  })
})

describe('evaluationReadiness', () => {
  const evidence = READY([model(1), model(2), reviewer(11)])
  const draft = (over: Partial<FeedbackDraft>): FeedbackDraft => ({
    correctRank: null,
    notes: '',
    ...over,
  })

  it('needs a deliberate choice', () => {
    const review = reviewSelections([], evidence)
    const readiness = evaluationReadiness(draft({}), evidence, review)
    expect(readiness.canSubmit).toBe(false)
    expect(readiness.block).toBe('no_choice')
  })

  it('accepts an explicit None on a fully readable model ranking', () => {
    const readiness = evaluationReadiness(
      draft({ correctRank: 0 }),
      evidence,
      reviewSelections([], evidence),
    )
    expect(readiness).toMatchObject({ canSubmit: true, outcome: 'none_of_top_k' })
  })

  it('refuses None on a partly unreadable ranking, keeping the draft answer', () => {
    const partial = READY([model(1), model(2, { candidate_files: [] })])
    const readiness = evaluationReadiness(
      draft({ correctRank: 0 }),
      partial,
      reviewSelections([], partial),
    )
    expect(readiness.canSubmit).toBe(false)
    expect(readiness.block).toBe('partial_model_evidence')
  })

  it('refuses a partly confirmed multi answer rather than saving a subset', () => {
    const selections: DraftSelection[] = [
      { rank: 1, dirName: 'candidate-1', source: 'model' },
      { rank: 11, dirName: null, source: null },
    ]
    const review = reviewSelections(selections, evidence)
    const readiness = evaluationReadiness(
      draft({ correctRank: 1, selectedRanks: [1, 11], selections }),
      evidence,
      review,
    )
    expect(readiness.canSubmit).toBe(false)
    expect(readiness.block).toBe('unconfirmed_choice')
    expect(readiness.selections).toEqual([])
  })

  it.each(['loading', 'error', 'empty', 'excluded'] as const)(
    'refuses to evaluate while %s, without touching the draft',
    (phase) => {
      const unavailable = assessmentEvidence({
        phase,
        predictions: [],
        hasReviewerAccount: true,
      })
      const selections: DraftSelection[] = [
        { rank: 1, dirName: 'candidate-1', source: 'model' },
      ]
      const review = reviewSelections(selections, unavailable)
      // The caller does not reconcile off a settled ranking, so nothing is
      // dropped; the draft simply cannot be submitted right now.
      expect(review.changed).toBe(true)
      const readiness = evaluationReadiness(
        draft({ correctRank: 0 }),
        unavailable,
        review,
      )
      expect(readiness.canSubmit).toBe(false)
      expect(readiness.outcome).toBeNull()
    },
  )
})

describe('choice bookkeeping', () => {
  const candidates: AssessmentCandidate[] = [
    {
      rank: 1,
      dirName: 'candidate-1',
      source: 'model',
      label: null,
      usable: true,
      evidence: 'readable',
    },
    {
      rank: 11,
      dirName: 'reviewer-dir-11',
      source: 'reviewer',
      label: 'Homily',
      usable: true,
      evidence: 'readable',
    },
  ]

  it('preserves the order choices were made in, not numeric order', () => {
    let selections = toggleSelection([], candidates[1], true)
    selections = toggleSelection(selections, candidates[0], true)
    expect(selections.map((selection) => selection.rank)).toEqual([11, 1])
    expect(expectedCandidateDirs(selections)).toEqual({
      '11': 'reviewer-dir-11',
      '1': 'candidate-1',
    })
  })

  it('confirms a restored choice in place rather than moving it to the end', () => {
    const restored: DraftSelection[] = [
      { rank: 11, dirName: null, source: null },
      { rank: 1, dirName: 'candidate-1', source: 'model' },
    ]
    const selections = toggleSelection(restored, candidates[1], true)
    expect(selections).toEqual([
      { rank: 11, dirName: 'reviewer-dir-11', source: 'reviewer' },
      { rank: 1, dirName: 'candidate-1', source: 'model' },
    ])
  })

  it('toggles a confirmed choice off, and replaces it in single-select', () => {
    const one = toggleSelection([], candidates[0], true)
    expect(toggleSelection(one, candidates[0], true)).toEqual([])
    expect(toggleSelection(one, candidates[1], false)).toEqual([
      { rank: 11, dirName: 'reviewer-dir-11', source: 'reviewer' },
    ])
  })

  it('omits an identity-less selection from the save precondition', () => {
    expect(expectedCandidateDirs([{ rank: 4, dirName: null, source: null }])).toEqual({})
  })
})
