import { describe, expect, it } from 'vitest'
import { validatePredictionResponse, type PredictionRequestKey } from './queries'

/**
 * `Prediction.supporting_member` at the runtime boundary (issue #163).
 *
 * `validatePredictionResponse` does not pass a payload through: it builds a new
 * response from the fields it recognises, so an additive backend field reaches
 * no consumer unless this layer carries it deliberately. These cases pin the
 * three things that has to mean -- carried when present, absent when the
 * artifact predates it, refused when it arrives in a shape this build cannot
 * read -- without letting the ranking itself change.
 *
 * The fixture is the payload the backend serialises for #163's own example: q1
 * under review, a reviewer group holding q2 (its seed, first in member order)
 * and q0, whose similarity is the group's maximum. Its shape was taken from a
 * live `GET /api/query/1/predictions`, not written from the type: same rank
 * anchor, same member order, same float, and `supporting_member: null` on the
 * model candidates above it.
 */

const KEY: PredictionRequestKey = {
  queryId: 1,
  model: 'bowphs_LaTa',
  variant: 'sif_abtt',
}

const GROUP_MAX = 0.7998046875
const REVIEWER_RANK = 11

/** The designated witness exactly as `SupportingMember.model_dump` writes it. */
const SUPPORT = { query_id: 0, filename: 'query-0.txt', score: GROUP_MAX }

function reviewerCandidate(overrides: Record<string, unknown> = {}) {
  return {
    rank: REVIEWER_RANK,
    dir_name: 'reviewer-dir-1',
    score: GROUP_MAX,
    dir_files: ['query-2.txt', 'query-0.txt'],
    preview_text: 'text of query-0.txt',
    candidate_files: [
      { filename: 'query-2.txt', text: 'text of query-2.txt' },
      { filename: 'query-0.txt', text: 'text of query-0.txt' },
    ],
    source: 'reviewer',
    label: 'Unattested homily',
    created_by: 'Abigail',
    seed_query_id: 2,
    supporting_member: SUPPORT,
    ...overrides,
  }
}

function modelCandidate() {
  return {
    rank: 1,
    dir_name: 'CANT.328.12',
    score: 0.44,
    dir_files: ['a.txt'],
    preview_text: 'labelled preview',
    candidate_files: [{ filename: 'a.txt', text: 'labelled text' }],
    source: 'model',
  }
}

function response(candidate: Record<string, unknown>) {
  return {
    file_id: KEY.queryId,
    filename: 'query-1.txt',
    model: KEY.model,
    variant: KEY.variant,
    predictions: [modelCandidate(), candidate],
    seeded_dirs: [],
  }
}

function validateWithSupport(value: unknown) {
  return validatePredictionResponse(
    response(reviewerCandidate({ supporting_member: value })),
    KEY,
  )
}

describe('supporting_member validation', () => {
  it('carries the designated witness through unchanged', () => {
    const result = validatePredictionResponse(
      response(reviewerCandidate()),
      KEY,
    )
    expect(result.ok).toBe(true)
    if (!result.ok) return

    const candidate = result.value.predictions[1]
    expect(candidate.supporting_member).toEqual(SUPPORT)
    // The float the backend sent, to the bit. This is the number the reviewer
    // judges, and the group maximum the designation speaks for.
    expect(candidate.supporting_member?.score).toBe(GROUP_MAX)
    expect(candidate.score).toBe(GROUP_MAX)
  })

  it('leaves the rest of the ranking exactly as it was', () => {
    const result = validatePredictionResponse(
      response(reviewerCandidate()),
      KEY,
    )
    expect(result.ok).toBe(true)
    if (!result.ok) return

    // Ranks, member order and the model candidate are not this field's
    // business, and stay untouched by its arrival.
    expect(result.value.predictions.map((p) => p.rank)).toEqual([1, REVIEWER_RANK])
    expect(result.value.predictions[0].supporting_member).toBeNull()
    expect(result.value.predictions[0].score).toBe(0.44)
    expect(result.value.predictions[1].dir_files).toEqual([
      'query-2.txt',
      'query-0.txt',
    ])
    expect(
      result.value.predictions[1].candidate_files?.map((f) => f.filename),
    ).toEqual(['query-2.txt', 'query-0.txt'])
  })

  it('accepts an artifact served before the field existed', () => {
    const candidate = reviewerCandidate()
    delete (candidate as Record<string, unknown>).supporting_member
    const result = validatePredictionResponse(response(candidate), KEY)
    expect(result.ok).toBe(true)
    if (!result.ok) return

    // Normalised to null, so every consumer reads one shape for "nobody was
    // named" instead of distinguishing undefined from null themselves.
    expect(result.value.predictions[1].supporting_member).toBeNull()
    expect(result.value.predictions[1].score).toBe(GROUP_MAX)
  })

  it('accepts an explicit null', () => {
    const result = validateWithSupport(null)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.predictions[1].supporting_member).toBeNull()
  })

  it.each([
    ['null filename', { query_id: 0, filename: null, score: GROUP_MAX }],
    ['absent filename', { query_id: 0, score: GROUP_MAX }],
  ])('keeps a winner whose metadata has no %s', (_name, value) => {
    const result = validateWithSupport(value)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    // Still that member. Nothing substitutes a readable filename for it.
    expect(result.value.predictions[1].supporting_member).toEqual({
      query_id: 0,
      filename: null,
      score: GROUP_MAX,
    })
  })

  it('accepts a member id of zero', () => {
    // `query-0.txt` is a real corpus file, and a falsy id must not be read as
    // a missing one.
    const result = validateWithSupport({ ...SUPPORT, query_id: 0 })
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.predictions[1].supporting_member?.query_id).toBe(0)
  })

  it('accepts a designation that ties another member, because the wire cannot say', () => {
    // `score_with_support` breaks equal maxima by the smallest member id, so a
    // tied payload is byte-identical to a strictly-won one. There is nothing
    // here to reject, and nothing that would let a consumer tell them apart.
    const result = validateWithSupport(SUPPORT)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.predictions[1].supporting_member).toEqual(SUPPORT)
  })

  describe('malformed designations', () => {
    const cases: [string, unknown][] = [
      ['a string', 'query-0.txt'],
      ['a number', 0],
      ['an array', [SUPPORT]],
      ['no query_id', { filename: 'query-0.txt', score: GROUP_MAX }],
      ['a fractional query_id', { ...SUPPORT, query_id: 0.5 }],
      ['a stringified query_id', { ...SUPPORT, query_id: '0' }],
      ['a null query_id', { ...SUPPORT, query_id: null }],
      ['no score', { query_id: 0, filename: 'query-0.txt' }],
      ['a NaN score', { ...SUPPORT, score: Number.NaN }],
      ['an infinite score', { ...SUPPORT, score: Number.POSITIVE_INFINITY }],
      ['a stringified score', { ...SUPPORT, score: '0.7998046875' }],
      ['a null score', { ...SUPPORT, score: null }],
      ['a numeric filename', { ...SUPPORT, filename: 0 }],
    ]

    it.each(cases)('rejects the response for %s', (_name, value) => {
      const result = validateWithSupport(value)
      expect(result).toEqual({ ok: false, reason: 'supporting_member is malformed' })
    })

    it('never completes a partial designation from the candidate', () => {
      // The candidate carries both a score and a first filename, so a lenient
      // normaliser has everything it needs to fill the gaps in -- and would
      // then be telling the reviewer which witness earned the number on the
      // strength of nothing.
      const result = validateWithSupport({ query_id: 0 })
      expect(result.ok).toBe(false)
      expect(JSON.stringify(result)).not.toContain('query-2.txt')
    })

    it('withholds the whole ranking rather than a ranking with no attribution', () => {
      // Silently nulling it would render the group maximum under this app's
      // honest wording for "the server named nobody", turning a build mismatch
      // into a claim about the retrieval run.
      const result = validateWithSupport({ ...SUPPORT, score: 'high' })
      expect(result.ok).toBe(false)
      expect('value' in result).toBe(false)
    })
  })
})
