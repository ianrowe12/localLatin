import { describe, expect, it } from 'vitest'
import { ApiError } from '../api/client'
import { classifySaveFailure, isIdentityRejection } from './saveFailure'

/**
 * What the reviewer may be told after a save fails (issue #157 repair).
 *
 * "Nothing was saved" is a claim about the server's log. The client can only
 * make it when no request was sent, or when the server answered with a status
 * this route raises before it inserts. Everything else has to stay uncertain,
 * because `web/routers/feedback.py` commits the feedback row before the
 * reviewer-directory membership write, so a fault can arrive after the record
 * exists and a confident denial would invite a duplicate.
 */
describe('classifySaveFailure', () => {
  function httpError(status: number, code: string | null = null) {
    return new ApiError({ kind: 'http', status, code, message: 'the server said so' })
  }

  it('claims nothing was attempted only for a refusal raised here', () => {
    const local = new Error('Add a note before skipping so the PI can follow up.')
    local.name = 'AssessmentNotSavableError'
    const failure = classifySaveFailure(local, 'fallback')
    expect(failure.outcome).toBe('not_attempted')
    expect(failure.kind).toBe('local')
    expect(failure.message).toContain('Add a note before skipping')
  })

  it('treats every 4xx as a rejection, because they all precede the insert', () => {
    for (const status of [400, 409, 422, 499]) {
      expect(classifySaveFailure(httpError(status), 'fallback').outcome).toBe('rejected')
    }
    const conflict = classifySaveFailure(httpError(409, 'CANDIDATE_IDENTITY_CHANGED'), 'f')
    expect(isIdentityRejection(conflict)).toBe(true)
    expect(isIdentityRejection(classifySaveFailure(httpError(422), 'f'))).toBe(false)
  })

  it('stays uncertain for a fault, a lost answer or an unreadable one', () => {
    for (const err of [
      httpError(500, 'internal_error'),
      httpError(503),
      new ApiError({
        kind: 'network',
        status: null,
        code: null,
        message: 'connection lost',
      }),
      new ApiError({ kind: 'malformed', status: 200, code: null, message: 'not JSON' }),
      new TypeError('Failed to fetch'),
      'a thrown string',
    ]) {
      expect(classifySaveFailure(err, 'fallback').outcome).toBe('uncertain')
    }
  })

  it('keeps the structured cause, so callers need not parse prose', () => {
    const failure = classifySaveFailure(httpError(409, 'CANDIDATE_IDENTITY_CHANGED'), 'f')
    expect(failure).toMatchObject({
      kind: 'http',
      status: 409,
      code: 'CANDIDATE_IDENTITY_CHANGED',
    })
  })
})
