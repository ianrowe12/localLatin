import { toApiErrorInfo, type ApiErrorInfo } from '../api/client'

/**
 * What a failed save allows the reviewer to be told (issue #157 repair).
 *
 * The panel used to answer every failure with "Nothing was saved." That is
 * true for a local refusal and for a server rejection, both of which happen
 * before `web/services/feedback_db.py` appends anything. It is not true for a
 * lost response, an unreadable one, or a server fault: `web/routers/feedback.py`
 * commits the feedback row and THEN writes reviewer-directory membership, so a
 * 500 can arrive after the record exists. Promising a reviewer that nothing was
 * written, when the client cannot know that, invites a retry that appends a
 * second row to an append-only log.
 *
 * - `not_attempted`: refused here, no request was made.
 * - `rejected`: the server answered with a client error, which this route only
 *   does before it writes. Safe to correct and save again.
 * - `uncertain`: no answer, an unreadable answer, or a server fault. The save
 *   may or may not have been recorded.
 */
export type SaveFailureOutcome = 'not_attempted' | 'rejected' | 'uncertain'

export interface SaveFailure {
  outcome: SaveFailureOutcome
  /** Structured, so a caller can tell 409 from 500 from a dropped socket. */
  kind: ApiErrorInfo['kind'] | 'local'
  status: number | null
  code: string | null
  message: string
}

/** True when the ranking, not the request, is what has to be looked at again. */
export function isIdentityRejection(failure: SaveFailure): boolean {
  return failure.code === 'CANDIDATE_IDENTITY_CHANGED'
}

/**
 * Classify a rejected save.
 *
 * `local` is passed for a refusal raised before any request. Everything else is
 * read from the structured `ApiError` of issue #156; a 4xx is a rejection
 * because every 4xx on this route is raised before the insert, and anything
 * else -- network, malformed, 5xx, or a status this client does not recognise
 * -- is uncertain by default. Erring towards uncertainty is the safe direction:
 * an over-cautious warning costs a reviewer a moment, a false "nothing was
 * saved" costs the log a duplicated human decision.
 */
export function classifySaveFailure(err: unknown, fallback: string): SaveFailure {
  if (err instanceof Error && err.name === 'AssessmentNotSavableError') {
    return {
      outcome: 'not_attempted',
      kind: 'local',
      status: null,
      code: null,
      message: err.message,
    }
  }
  const info = toApiErrorInfo(err, fallback)
  const rejected =
    info.kind === 'http' &&
    info.status !== null &&
    info.status >= 400 &&
    info.status < 500
  return {
    outcome: rejected ? 'rejected' : 'uncertain',
    kind: info.kind,
    status: info.status,
    code: info.code,
    message: info.message,
  }
}
