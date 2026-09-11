import {
  candidateHasText,
  modelCandidates,
  type CandidateSource,
  type Prediction,
  type PredictionPhase,
} from '../api/queries'
import { draftIsNone, draftSelections, type DraftSelection, type FeedbackDraft } from './feedbackDraft'

/**
 * Why a candidate can or cannot be judged from this screen.
 *
 * - `readable`: a witness the reviewer can reach carries words.
 * - `no_text`: nothing this response delivered for the directory is readable
 *   in this deployment.
 * - `unidentified`: no directory name or no finite score, so there is nothing
 *   coherent to assign the document to.
 */
export type CandidateEvidence = 'readable' | 'no_text' | 'unidentified'

/**
 * A candidate as the assessment controls see it (issues #157, #163).
 *
 * `usable` asks whether the reviewer can actually read this candidate HERE: a
 * directory identity, a finite score, and readable words in a witness this
 * screen can reach.
 *
 * That last clause used to mean `candidate_files[0]` alone, because the panel
 * showed the first witness and offered no way to any other. Being stricter
 * than the server's `web/routers/feedback.py::_candidate_is_usable`, which
 * accepts a directory with a readable file anywhere in it, was then the safe
 * direction: everything this client offered, the server also accepted.
 *
 * Issue #163 mounts a witness selector in `CenterArea` for BOTH model and
 * reviewer candidates, so every witness the response delivered is one control
 * away and a readable second file is no longer evidence nobody can see. The
 * two rules are identical again -- identity, a finite score, and
 * `candidateHasText`, which is the same `any(file.text.strip() ...)` the
 * server applies -- so no readable candidate is refused here and no offered
 * save is refused there.
 *
 * The question is REACHABILITY, never which witness is currently displayed.
 * Opening a blank member of a directory that also has a readable one does not
 * withdraw the assessment: the evidence did not change, only the view did, and
 * a choice that came and went with a selection would make the selector an
 * editor of answers.
 */
export interface AssessmentCandidate {
  rank: number
  dirName: string
  source: CandidateSource
  label: string | null
  usable: boolean
  evidence: CandidateEvidence
}

/**
 * Why no evaluation can be recorded right now.
 *
 * `null` means the ranking supports evaluation. Every other value is a fact
 * about this moment, not a verdict on the manuscript: the panel turns it into
 * copy, and Skip with a note stays available throughout.
 */
export type EvidenceBlock =
  | 'no_query'
  | 'no_reviewer'
  | 'loading'
  | 'failed'
  | 'empty'
  | 'excluded'
  | 'no_usable_model_evidence'

/** `partial_model_evidence` blocks None only; a readable rank stays choosable. */
export type NoneBlock = EvidenceBlock | 'partial_model_evidence'

export interface AssessmentEvidence {
  phase: PredictionPhase
  /** Every current candidate, in the order the API returned them. */
  candidates: AssessmentCandidate[]
  /** The model's own candidates, which are what "None" is an answer about. */
  modelCandidates: AssessmentCandidate[]
  /** Positive choices are possible at all. */
  canEvaluate: boolean
  evidenceBlock: EvidenceBlock | null
  noneAvailable: boolean
  noneBlock: NoneBlock | null
  candidateAt: (rank: number) => AssessmentCandidate | null
}

/**
 * What this screen can tell the reviewer about a candidate.
 *
 * One question, asked of every witness the response delivered, because the
 * selector reaches all of them: are there readable words anywhere the reviewer
 * can open? `candidateHasText` already is that question, so the rule is the
 * API's own helper rather than a second copy of it living here.
 */
export function candidateEvidence(prediction: Prediction): CandidateEvidence {
  if (prediction.dir_name.trim().length === 0 || !Number.isFinite(prediction.score)) {
    return 'unidentified'
  }
  return candidateHasText(prediction) ? 'readable' : 'no_text'
}

/** Whether this candidate can be judged from the evidence on screen. */
export function candidateIsUsable(prediction: Prediction): boolean {
  return candidateEvidence(prediction) === 'readable'
}

function toCandidate(prediction: Prediction): AssessmentCandidate {
  const evidence = candidateEvidence(prediction)
  return {
    rank: prediction.rank,
    dirName: prediction.dir_name,
    source: prediction.source ?? 'model',
    label: prediction.label ?? null,
    usable: evidence === 'readable',
    evidence,
  }
}

function evidenceBlockFor(
  phase: PredictionPhase,
  hasReviewerAccount: boolean,
  usableModelCandidates: number,
): EvidenceBlock | null {
  if (!hasReviewerAccount) return 'no_reviewer'
  if (phase === 'idle') return 'no_query'
  if (phase === 'loading') return 'loading'
  if (phase === 'error') return 'failed'
  if (phase === 'excluded') return 'excluded'
  if (phase === 'empty') return 'empty'
  return usableModelCandidates > 0 ? null : 'no_usable_model_evidence'
}

/**
 * What the current prediction state supports.
 *
 * Everything the panel offers -- pills, the None control, Submit, and the
 * context's own submission methods -- is derived from this one value, so a
 * click, a restored draft and a direct context call cannot disagree about what
 * the evidence allows. Nothing here consults a caller-supplied array or flag.
 */
export function assessmentEvidence(input: {
  phase: PredictionPhase
  predictions: Prediction[]
  hasReviewerAccount: boolean
}): AssessmentEvidence {
  const ready = input.phase === 'ready'
  // Only a settled ranking describes candidates. Loading, failure, an
  // unexplained empty answer and a deliberate exclusion all offer none, so no
  // phantom rank can be rendered from an array length.
  const candidates = ready ? input.predictions.map(toCandidate) : []
  const model = ready
    ? modelCandidates(input.predictions).map(toCandidate)
    : []
  const usableModel = model.filter((candidate) => candidate.usable)
  const evidenceBlock = evidenceBlockFor(
    input.phase,
    input.hasReviewerAccount,
    usableModel.length,
  )
  // None is a judgement about every model candidate on offer. If one of them
  // cannot be read HERE, the reviewer cannot have rejected it: its text is
  // missing in this deployment, which the server also refuses
  // (RANKING_NOT_EVALUABLE). Reviewer directories are not model evidence and
  // neither supply nor withdraw it.
  const noneBlock: NoneBlock | null =
    evidenceBlock !== null
      ? evidenceBlock
      : usableModel.length === model.length
        ? null
        : 'partial_model_evidence'

  const byRank = new Map(candidates.map((candidate) => [candidate.rank, candidate]))
  return {
    phase: input.phase,
    candidates,
    modelCandidates: model,
    canEvaluate: evidenceBlock === null,
    evidenceBlock,
    noneAvailable: noneBlock === null,
    noneBlock,
    candidateAt: (rank: number) => byRank.get(rank) ?? null,
  }
}

/** What happened to a stored choice when it met the current ranking. */
export type SelectionIssue =
  | { kind: 'vanished'; rank: number; dirName: string | null }
  | { kind: 'reassigned'; rank: number; dirName: string; nowDirName: string }
  | { kind: 'unverified'; rank: number }
  | {
      kind: 'unreadable'
      rank: number
      dirName: string
      /**
       * Which kind of unreadable, so the copy states the fact rather than
       * inferring it again from the prediction.
       */
      evidence: CandidateEvidence
    }

export interface SelectionReview {
  /** The selections to keep in the draft, in their original order. */
  selections: DraftSelection[]
  /** Those that are deliberate, identified and backed by readable evidence. */
  confirmed: DraftSelection[]
  issues: SelectionIssue[]
  /** The stored selections must be rewritten (something was dropped). */
  changed: boolean
}

/**
 * Reconcile stored choices against the ranking now on screen.
 *
 * Only ever called with a `ready` evidence value: a failed, empty, excluded or
 * still-loading request is not evidence that anything about the draft is wrong,
 * and drafts must survive all four (issue #157).
 *
 * A choice is dropped only when its meaning is gone -- the rank no longer
 * exists, or another directory now occupies it. A choice whose identity was
 * never recorded is KEPT and returned as `unverified`: the reviewer is asked to
 * confirm it, because binding an identity-less rank to whatever sits there now
 * is exactly the silent reassignment this issue exists to prevent.
 */
export function reviewSelections(
  selections: DraftSelection[],
  evidence: AssessmentEvidence,
): SelectionReview {
  const kept: DraftSelection[] = []
  const confirmed: DraftSelection[] = []
  const issues: SelectionIssue[] = []
  let changed = false

  for (const selection of selections) {
    const candidate = evidence.candidateAt(selection.rank)
    if (candidate === null) {
      issues.push({ kind: 'vanished', rank: selection.rank, dirName: selection.dirName })
      changed = true
      continue
    }
    if (
      selection.dirName !== null &&
      (candidate.dirName !== selection.dirName ||
        (selection.source !== null && candidate.source !== selection.source))
    ) {
      issues.push({
        kind: 'reassigned',
        rank: selection.rank,
        dirName: selection.dirName,
        nowDirName: candidate.dirName,
      })
      changed = true
      continue
    }
    kept.push(selection)
    if (selection.dirName === null) {
      issues.push({ kind: 'unverified', rank: selection.rank })
    } else if (!candidate.usable) {
      issues.push({
        kind: 'unreadable',
        rank: selection.rank,
        dirName: candidate.dirName,
        evidence: candidate.evidence,
      })
    } else {
      confirmed.push(selection)
    }
  }

  return { selections: kept, confirmed, issues, changed }
}

/** Why Submit cannot record this draft. `null` means it can. */
export type SubmitBlock = EvidenceBlock | NoneBlock | 'no_choice' | 'unconfirmed_choice'

export interface EvaluationReadiness {
  canSubmit: boolean
  block: SubmitBlock | null
  /** The deliberate answer, if there is one. */
  outcome: 'matched_rank' | 'none_of_top_k' | null
  /** Confirmed selections in choice order; the first is canonical. */
  selections: DraftSelection[]
}

/**
 * Whether this draft is a deliberate, currently valid answer.
 *
 * A partially confirmed multi-answer is NOT submittable: saving the confirmed
 * subset would quietly change what the reviewer answered. They confirm the rest
 * or remove it.
 */
export function evaluationReadiness(
  draft: FeedbackDraft | undefined,
  evidence: AssessmentEvidence,
  review: SelectionReview,
): EvaluationReadiness {
  if (draftIsNone(draft)) {
    return evidence.noneAvailable
      ? { canSubmit: true, block: null, outcome: 'none_of_top_k', selections: [] }
      : { canSubmit: false, block: evidence.noneBlock, outcome: null, selections: [] }
  }
  if (evidence.evidenceBlock !== null) {
    return {
      canSubmit: false,
      block: evidence.evidenceBlock,
      outcome: null,
      selections: [],
    }
  }
  if (review.selections.length === 0) {
    return { canSubmit: false, block: 'no_choice', outcome: null, selections: [] }
  }
  if (review.confirmed.length !== review.selections.length) {
    return {
      canSubmit: false,
      block: 'unconfirmed_choice',
      outcome: null,
      selections: [],
    }
  }
  return {
    canSubmit: true,
    block: null,
    outcome: 'matched_rank',
    selections: review.confirmed,
  }
}

/** Draft selections used as the save-time identity precondition, by rank. */
export function expectedCandidateDirs(
  selections: DraftSelection[],
): Record<string, string> {
  const expected: Record<string, string> = {}
  for (const selection of selections) {
    if (selection.dirName === null) continue
    expected[String(selection.rank)] = selection.dirName
  }
  return expected
}

/** Selections whose identity is unknown, so the reviewer must reconfirm them. */
export function unconfirmedRanks(review: SelectionReview): number[] {
  const confirmed = new Set(review.confirmed.map((selection) => selection.rank))
  return review.selections
    .filter((selection) => !confirmed.has(selection.rank))
    .map((selection) => selection.rank)
}

/** Draft selections merged with a fresh click, preserving the original order. */
export function toggleSelection(
  selections: DraftSelection[],
  candidate: AssessmentCandidate,
  multiSelect: boolean,
): DraftSelection[] {
  const chosen: DraftSelection = {
    rank: candidate.rank,
    dirName: candidate.dirName,
    source: candidate.source,
  }
  const existing = selections.find((selection) => selection.rank === candidate.rank)
  if (!multiSelect) {
    // Clicking the confirmed active choice clears it; anything else replaces it.
    if (existing && existing.dirName === candidate.dirName) return []
    return [chosen]
  }
  if (!existing) return [...selections, chosen]
  if (existing.dirName !== candidate.dirName) {
    // Confirming a restored choice happens IN PLACE: the order of the answer is
    // the order the reviewer chose it in, and the first entry is the one the
    // server makes canonical.
    return selections.map((selection) =>
      selection.rank === candidate.rank ? chosen : selection,
    )
  }
  return selections.filter((selection) => selection.rank !== candidate.rank)
}
