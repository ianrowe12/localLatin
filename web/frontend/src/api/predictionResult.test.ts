import { describe, expect, it } from 'vitest'
import { ApiError, toApiErrorInfo } from './client'
import { isCompleteReviewerDir } from './reviewerDirs'
import {
  candidateHasText,
  classifyExclusion,
  modelCandidates,
  predictionKeyString,
  reviewerCandidates,
  samePredictionKey,
  validatePredictionResponse,
  type PredictionRequestKey,
} from './queries'

const KEY: PredictionRequestKey = {
  queryId: 7,
  model: 'bowphs_LaTa',
  variant: 'sif_abtt',
}

/** A response body as the route actually builds it. */
function body(overrides: Record<string, unknown> = {}) {
  return {
    file_id: 7,
    filename: 'query-7.txt',
    model: 'bowphs_LaTa',
    variant: 'sif_abtt',
    status: 'ok',
    predictions: [
      {
        rank: 1,
        dir_name: 'candidate-1',
        score: 0.81,
        dir_files: ['1.txt'],
        preview_text: 'preview',
        candidate_files: [{ filename: '1.txt', text: 'candidate text' }],
        source: 'model',
      },
    ],
    seeded_dirs: [],
    ...overrides,
  }
}

function card(overrides: Record<string, unknown>) {
  return {
    rank: 1,
    dir_name: 'candidate-1',
    score: 0.5,
    dir_files: [],
    preview_text: '',
    candidate_files: null,
    source: 'model',
    ...overrides,
  }
}

describe('validatePredictionResponse (issue #156)', () => {
  it('accepts the route’s own shape and keeps the status verbatim', () => {
    const result = validatePredictionResponse(body(), KEY)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.status).toBe('ok')
    expect(result.value.predictions).toHaveLength(1)
    expect(result.value.predictions[0].dir_name).toBe('candidate-1')
  })

  it('rejects a ranking for a different document', () => {
    // The reviewer's whole judgement rests on these words being about the text
    // on screen. A 200 for the wrong file_id is not "mostly right".
    const result = validatePredictionResponse(body({ file_id: 8 }), KEY)
    expect(result).toEqual({ ok: false, reason: 'file_id mismatch' })
  })

  it('rejects a ranking produced by a different model', () => {
    const result = validatePredictionResponse(body({ model: 'other' }), KEY)
    expect(result).toEqual({ ok: false, reason: 'model mismatch' })
  })

  it('rejects a ranking produced by a different variant', () => {
    const result = validatePredictionResponse(body({ variant: 'raw' }), KEY)
    expect(result).toEqual({ ok: false, reason: 'variant mismatch' })
  })

  it('keeps a sparse ranking, because rank 11 is where reviewer dirs live', () => {
    // [1, 11] is a real response: the model returned one candidate and a
    // reviewer directory was appended at its anchor rank. Demanding 1..n would
    // throw away a legitimate ranking.
    const result = validatePredictionResponse(
      body({
        predictions: [
          card({ rank: 1 }),
          card({ rank: 11, dir_name: 'reviewer-dir-1', source: 'reviewer' }),
        ],
      }),
      KEY,
    )
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.predictions.map((p) => p.rank)).toEqual([1, 11])
  })

  it('rejects the whole response on a duplicate rank', () => {
    // Two candidates at rank 3 means "rank 3" no longer names one directory.
    // Dropping one of them would renumber what the reviewer sees against what
    // the server resolves a saved rank to.
    const result = validatePredictionResponse(
      body({ predictions: [card({ rank: 3 }), card({ rank: 3, dir_name: 'b' })] }),
      KEY,
    )
    expect(result).toEqual({ ok: false, reason: 'duplicate candidate rank' })
  })

  it.each([
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
    ['a string', '0.5'],
  ])('rejects a score that is %s', (_label, score) => {
    const result = validatePredictionResponse(
      body({ predictions: [card({ score })] }),
      KEY,
    )
    expect(result).toEqual({ ok: false, reason: 'candidate score is not finite' })
  })

  it('rejects a candidate with no directory behind it', () => {
    const result = validatePredictionResponse(
      body({ predictions: [card({ dir_name: '' })] }),
      KEY,
    )
    expect(result).toEqual({ ok: false, reason: 'candidate has no directory' })
  })

  it('rejects a rank that is not a positive integer', () => {
    const result = validatePredictionResponse(
      body({ predictions: [card({ rank: 0 })] }),
      KEY,
    )
    expect(result).toEqual({
      ok: false,
      reason: 'candidate rank is not a positive integer',
    })
  })

  it('rejects an unknown candidate source rather than guessing "model"', () => {
    const result = validatePredictionResponse(
      body({ predictions: [card({ source: 'oracle' })] }),
      KEY,
    )
    expect(result).toEqual({ ok: false, reason: 'candidate source is unknown' })
  })

  it('rejects malformed candidate_files', () => {
    const result = validatePredictionResponse(
      body({ predictions: [card({ candidate_files: [{ filename: 'a.txt' }] })] }),
      KEY,
    )
    expect(result).toEqual({ ok: false, reason: 'candidate_files is malformed' })
  })

  it('rejects a non-string status instead of coercing it', () => {
    const result = validatePredictionResponse(body({ status: 3 }), KEY)
    expect(result).toEqual({ ok: false, reason: 'status is not a string' })
  })

  it('tolerates a legacy body with no status and no seeded_dirs', () => {
    // The mock server and any older deployment answer without either field.
    // Absent is not "excluded"; it is unknown.
    const legacy = body()
    delete (legacy as Record<string, unknown>).status
    delete (legacy as Record<string, unknown>).seeded_dirs
    const result = validatePredictionResponse(legacy, KEY)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.status).toBeNull()
    expect(result.value.seeded_dirs).toEqual([])
  })

  it('rejects a body that is not an object at all', () => {
    expect(validatePredictionResponse('<html>502</html>', KEY).ok).toBe(false)
    expect(validatePredictionResponse(null, KEY).ok).toBe(false)
    expect(validatePredictionResponse([], KEY).ok).toBe(false)
  })

  it('defaults a legacy candidate with no source to the model', () => {
    const result = validatePredictionResponse(
      body({ predictions: [{ rank: 1, dir_name: 'candidate-1', score: 0.5 }] }),
      KEY,
    )
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.predictions[0].source).toBe('model')
    expect(result.value.predictions[0].dir_files).toEqual([])
  })
})

describe('validatePredictionResponse: seeded_dirs (issue #156)', () => {
  const dir = (over: Record<string, unknown> = {}) => ({
    dir_id: 'reviewer-dir-3',
    label: 'Ordo de catechizandis',
    status: 'awaiting_match',
    seed_query_id: 7,
    member_query_ids: [7],
    best_match_score: 0.61,
    has_potential_match: true,
    created_at: '2026-01-04T10:00:00Z',
    created_by: 'reviewer-1',
    model_slug: 'bowphs_LaTa',
    variant: 'sif_abtt',
    ...over,
  })

  const seeded = (dirs: unknown) => validatePredictionResponse(body({ seeded_dirs: dirs }), KEY)

  it('accepts a well-formed directory unchanged', () => {
    const result = seeded([dir()])
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.seeded_dirs).toEqual([dir()])
  })

  // The badge reads `dir.status` and `dir.best_match_score.toFixed(2)` at render
  // time. Before this, `seeded_dirs` was cast, not checked, so a malformed entry
  // reached the badge and threw during render -- the whole panel went blank,
  // which is precisely the unexplained-absence failure this issue exists to end.
  it('rejects a null entry rather than casting it onto the badge', () => {
    expect(seeded([null])).toEqual({
      ok: false,
      reason: 'seeded_dirs[0] is not an object',
    })
  })

  it('rejects a directory with no id, which nothing downstream could key on', () => {
    expect(seeded([dir({ dir_id: '' })])).toEqual({
      ok: false,
      reason: 'seeded_dirs[0].dir_id is missing',
    })
  })

  it('rejects a score the badge would call .toFixed on', () => {
    expect(seeded([dir({ best_match_score: '0.61' })])).toEqual({
      ok: false,
      reason: 'seeded_dirs[0].best_match_score is not a number',
    })
  })

  // Deliberate: an unknown status fails the whole response instead of being
  // dropped. The badge is the only report a reviewer gets about a directory
  // they created, so a silently missing badge is the same unexplained absence
  // by another route. Frontend and backend ship together (web/ subtree), so a
  // status the client has never heard of means the payload is wrong, loudly.
  it('rejects a status it cannot render', () => {
    expect(seeded([dir({ status: 'archived' })])).toEqual({
      ok: false,
      reason: 'seeded_dirs[0].status is not a known status',
    })
  })

  it('rejects member ids that are not integers', () => {
    expect(seeded([dir({ member_query_ids: [7, 'eight'] })])).toEqual({
      ok: false,
      reason: 'seeded_dirs[0].member_query_ids is malformed',
    })
  })

  it('rejects seeded_dirs that is not an array', () => {
    expect(seeded({ 'reviewer-dir-3': dir() })).toEqual({
      ok: false,
      reason: 'seeded_dirs is not an array',
    })
  })

  it('fills the optional fields the backend defaults, without inventing an id', () => {
    // web/models.py gives model_slug, member_query_ids and has_potential_match
    // defaults, so an older payload can legitimately omit them.
    const result = seeded([
      {
        dir_id: 'reviewer-dir-9',
        label: 'Sparse',
        status: 'matched',
        seed_query_id: 12,
      },
    ])
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value.seeded_dirs?.[0]).toMatchObject({
      dir_id: 'reviewer-dir-9',
      member_query_ids: [],
      best_match_score: null,
      has_potential_match: false,
      model_slug: '',
    })
  })

  /**
   * Issue #161 handoff. The substitutes above are the right answer for a
   * candidate in a ranking and are not evidence of anything for the permanent
   * saved-directory record, which has to tell "the server said no members" from
   * "the field was not on the wire". Both are `[]` by the time anything
   * downstream sees them, so this validator names what it supplied rather than
   * leaving a distinction nobody can recover.
   */
  describe('names the defaults it supplied (issue #161)', () => {
    const defaultedFor = (over: Record<string, unknown>) => {
      const row = dir()
      for (const field of Object.keys(over)) delete (row as Record<string, unknown>)[field]
      const result = seeded([row])
      expect(result.ok).toBe(true)
      if (!result.ok) return undefined
      return result.value.seeded_dirs?.[0]?.defaulted_fields
    }

    // THE EXHAUSTIVE AUDIT (issue #161, finding 12).
    //
    // Every field a complete directory row carries, one row per field, with
    // what this client does when the server omits exactly that one. Two
    // substitutes were missed the first time round because their stand-in
    // values are unremarkable: an absent score became `null` and an absent
    // flag became `false`, which is what an ordinary scored-nothing row looks
    // like. Naming only the suspicious-looking defaults is what let those two
    // through, so the table below is stated for ALL of them and the test after
    // it proves the table itself is complete -- add a field to the row, or a
    // substitute to the validator, and something here fails.
    //
    // `rejected`  the validator refuses the response; nothing is substituted.
    // `marked`    a substitute is supplied and named, so the durable boundary
    //             declines the row.
    // `unmarked`  a substitute is supplied and NOT named, which is only safe
    //             because `isCompleteReviewerDir` never reads the field.
    const AUDIT = [
      { field: 'dir_id', verdict: 'rejected' },
      { field: 'label', verdict: 'rejected' },
      { field: 'status', verdict: 'rejected' },
      { field: 'seed_query_id', verdict: 'rejected' },
      { field: 'member_query_ids', verdict: 'marked' },
      { field: 'created_at', verdict: 'marked' },
      { field: 'created_by', verdict: 'marked' },
      { field: 'model_slug', verdict: 'marked' },
      { field: 'best_match_score', verdict: 'marked' },
      { field: 'has_potential_match', verdict: 'marked' },
      { field: 'variant', verdict: 'unmarked' },
    ] as const

    const withoutField = (field: string) => {
      const row = dir()
      delete (row as Record<string, unknown>)[field]
      return row
    }

    it('covers every field the row actually carries', () => {
      // The guard against this table going stale. A field added to a directory
      // fails here until someone decides which column it belongs in.
      expect(AUDIT.map((entry) => entry.field).sort()).toEqual(Object.keys(dir()).sort())
    })

    it.each(AUDIT.filter((entry) => entry.verdict === 'rejected'))(
      'refuses the whole response when $field is absent',
      ({ field }) => {
        expect(seeded([withoutField(field)]).ok).toBe(false)
      },
    )

    it.each(AUDIT.filter((entry) => entry.verdict === 'marked'))(
      'names $field when that one field is absent, and the durable boundary declines it',
      ({ field }) => {
        expect(defaultedFor({ [field]: true })).toEqual([field])
        const result = seeded([withoutField(field)])
        expect(result.ok).toBe(true)
        if (!result.ok) return
        // The half that matters: naming it is only useful if the durable
        // boundary acts on the name.
        expect(isCompleteReviewerDir(result.value.seeded_dirs![0])).toBe(false)
      },
    )

    it.each(AUDIT.filter((entry) => entry.verdict === 'unmarked'))(
      'leaves $field unnamed, and the durable boundary does not read it',
      ({ field }) => {
        const result = seeded([withoutField(field)])
        expect(result.ok).toBe(true)
        if (!result.ok) return
        expect(result.value.seeded_dirs![0]).not.toHaveProperty('defaulted_fields')
        // Marking this one would make the ranking path refuse a row the
        // directory endpoint itself accepts, for a value nothing durable reads.
        expect(isCompleteReviewerDir(result.value.seeded_dirs![0])).toBe(true)
      },
    )

    it('names every absent field, not just the first', () => {
      expect(
        defaultedFor({
          member_query_ids: true,
          created_at: true,
          created_by: true,
          best_match_score: true,
          has_potential_match: true,
        }),
      ).toEqual([
        'member_query_ids',
        'created_at',
        'created_by',
        'best_match_score',
        'has_potential_match',
      ])
    })

    // The distinction that matters. This row is the one a pre-#160 partial
    // write left behind: the server really holds a directory with no members,
    // and says so. Reporting it as defaulted would make a stored fact
    // indistinguishable from a missing field all over again.
    // The distinction, three times over. Each of these is an ordinary thing
    // for the server to say, and each has a substitute that is the same value.
    // Reporting them as defaulted would make a stored fact indistinguishable
    // from a missing field all over again -- in the other direction, and it
    // would take a real historical record off the screen.
    it.each([
      ['an explicitly empty member list', { member_query_ids: [] }],
      ['an explicit null score', { best_match_score: null }],
      ['an explicit false match flag', { has_potential_match: false }],
      [
        'all three at once, which is a real pre-#160 partial row',
        { member_query_ids: [], best_match_score: null, has_potential_match: false },
      ],
    ])('says nothing about %s', (_name, over) => {
      const result = seeded([dir(over)])
      expect(result.ok).toBe(true)
      if (!result.ok) return
      const normalized = result.value.seeded_dirs![0]!
      expect(normalized).not.toHaveProperty('defaulted_fields')
      expect(normalized).toEqual(dir(over))
      // Admissible: the server answered every one of these.
      expect(isCompleteReviewerDir(normalized)).toBe(true)
    })

    // Absent entirely on a complete row, so the normalised directory stays the
    // object the response carried -- see the first case in this describe block,
    // which compares it to the fixture with toEqual.
    it('adds no key at all when nothing was supplied', () => {
      const result = seeded([dir()])
      expect(result.ok).toBe(true)
      if (!result.ok) return
      expect(Object.keys(result.value.seeded_dirs![0]!)).not.toContain('defaulted_fields')
    })
  })
})

describe('classifyExclusion (issue #156)', () => {
  it('names the two reasons the retrieval run actually writes', () => {
    // scripts/resubmit/run_resubmit_unlabelled_retrieval.py writes these.
    expect(classifyExclusion('excluded_blank_source')).toBe('blank_source')
    expect(classifyExclusion('excluded_zero_norm')).toBe('zero_norm')
  })

  it('keeps an unrecognised exclusion an exclusion', () => {
    // The writer owns this vocabulary. A reason this build has never heard of
    // is still an exclusion, and must not be reported as an empty ranking.
    expect(classifyExclusion('excluded_something_new')).toBe('other')
  })

  it('is not an exclusion for ok, for null or for unknown values', () => {
    expect(classifyExclusion('ok')).toBeNull()
    expect(classifyExclusion(null)).toBeNull()
    expect(classifyExclusion(undefined)).toBeNull()
    expect(classifyExclusion('')).toBeNull()
  })
})

describe('candidate helpers', () => {
  it('separates model candidates from reviewer directories', () => {
    const all = [
      { ...card({ rank: 1 }), source: 'model' as const, label: null, created_by: null, seed_query_id: null, candidate_files: null },
      { ...card({ rank: 11 }), source: 'reviewer' as const, label: null, created_by: null, seed_query_id: null, candidate_files: null },
    ]
    expect(modelCandidates(all).map((p) => p.rank)).toEqual([1])
    expect(reviewerCandidates(all).map((p) => p.rank)).toEqual([11])
  })

  it('reports missing candidate text rather than inventing an excerpt', () => {
    // The route fills candidate_files with texts.get(fname, ""), so a candidate
    // really can arrive with a filename and no words.
    const base = { rank: 1, dir_name: 'd', score: 0.5, dir_files: [], preview_text: '', source: 'model' as const, label: null, created_by: null, seed_query_id: null }
    expect(candidateHasText({ ...base, candidate_files: null })).toBe(false)
    expect(candidateHasText({ ...base, candidate_files: [] })).toBe(false)
    expect(
      candidateHasText({ ...base, candidate_files: [{ filename: 'a.txt', text: '   ' }] }),
    ).toBe(false)
    expect(
      candidateHasText({ ...base, candidate_files: [{ filename: 'a.txt', text: 'uerba' }] }),
    ).toBe(true)
  })

  it('compares request keys by value, and treats null as no selection', () => {
    expect(samePredictionKey(KEY, { ...KEY })).toBe(true)
    expect(samePredictionKey(KEY, { ...KEY, model: 'other' })).toBe(false)
    expect(samePredictionKey(null, null)).toBe(true)
    expect(samePredictionKey(KEY, null)).toBe(false)
    expect(predictionKeyString(KEY)).toBe('7:bowphs_LaTa:sif_abtt')
  })
})

describe('ApiError (issue #156)', () => {
  it('carries the FastAPI domain error shape', () => {
    // web/app.py's handlers answer {"error": {"code", "message"}}.
    const err = new ApiError({
      kind: 'http',
      status: 409,
      code: 'reviewer_dir_exists',
      message: 'This document already seeds a directory.',
    })
    expect(err.kind).toBe('http')
    expect(err.status).toBe(409)
    expect(err.code).toBe('reviewer_dir_exists')
    expect(err.message).toBe('This document already seeds a directory.')
    expect(err instanceof Error).toBe(true)
  })

  it('is readable by every existing `catch (err) => err.message` caller', () => {
    // The old client threw a plain Error. Nothing downstream had to change.
    const err: unknown = new ApiError({
      kind: 'http',
      status: 401,
      code: null,
      message: 'Authentication required',
    })
    expect(err instanceof Error && err.message).toBe('Authentication required')
  })

  it('preserves a structured error through toApiErrorInfo', () => {
    const info = toApiErrorInfo(
      new ApiError({ kind: 'network', status: null, code: null, message: 'Failed to fetch' }),
      'fallback',
    )
    expect(info).toEqual({
      kind: 'network',
      status: null,
      code: null,
      message: 'Failed to fetch',
    })
  })

  it('degrades an unknown throw to a stated fallback, not to silence', () => {
    expect(toApiErrorInfo(new Error('boom'), 'fallback')).toEqual({
      kind: 'network',
      status: null,
      code: null,
      message: 'boom',
    })
    expect(toApiErrorInfo('not an error', 'fallback')).toEqual({
      kind: 'network',
      status: null,
      code: null,
      message: 'fallback',
    })
  })
})
