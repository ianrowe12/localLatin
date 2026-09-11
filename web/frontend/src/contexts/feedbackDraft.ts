import type { CandidateSource } from '../api/queries'
import { DEFAULT_VARIANT, type PredictionVariant } from '../api/variants'

/**
 * One candidate a reviewer chose, named by what they were looking at.
 *
 * A rank number alone is not a choice (issue #157). `web/services/reviewer_dirs.py`
 * skips directories the query already belongs to, so the directory behind rank
 * 11 can differ between two responses; a draft that remembers "11" and nothing
 * else can be saved against a directory the reviewer never saw.
 *
 * `dirName` is therefore the durable half of the identity, and it is
 * deliberately NOT `PredictionContext.identityOf`, which mixes in the client
 * request generation: a refresh that returns the same directory at the same
 * rank is the same candidate, and must survive.
 *
 * `dirName: null` means the identity was never recorded -- a draft written
 * before this shape existed, or a restored answer whose directory the server
 * does not report for that rank. Such a selection is offered back for explicit
 * reconfirmation; it is never treated as a deliberate current choice.
 */
export interface DraftSelection {
  rank: number
  dirName: string | null
  source: CandidateSource | null
}

export interface FeedbackDraft {
  /** 0 = none-of-the-model-candidates, null = unset, >0 = first selected rank. */
  correctRank: number | null
  /** Selected ranks in the order they were chosen. Mirrors `selections`. */
  selectedRanks?: number[]
  /** The same choices with the directory each named when it was made. */
  selections?: DraftSelection[]
  notes: string
  /**
   * Which edit of this draft this is (issue #158).
   *
   * Monotonic per draft key and never reused, so a save that finishes late can
   * tell "the draft I sent" from "what is in the box now". Content equality is
   * not enough: a reviewer who types a word during the POST and deletes it
   * again has still moved on from the submitted revision, and clearing the box
   * under them (or navigating away) would throw away work they can see.
   *
   * Absent on drafts written before this existed; read `draftRevision()`.
   */
  revision?: number
}

export function emptyDraft(): FeedbackDraft {
  return { correctRank: null, notes: '' }
}

export function isFeedbackDraftEmpty(draft: FeedbackDraft | undefined): boolean {
  if (!draft) return true
  return (
    draft.correctRank === null &&
    (!draft.selectedRanks || draft.selectedRanks.length === 0) &&
    (!draft.selections || draft.selections.length === 0) &&
    draft.notes.trim() === ''
  )
}

export function seedDraftMapIfEmpty(
  drafts: Map<string, FeedbackDraft>,
  key: string,
  seed: FeedbackDraft,
  revision?: number,
): Map<string, FeedbackDraft> {
  if (!isFeedbackDraftEmpty(drafts.get(key))) return drafts
  const next = new Map(drafts)
  next.set(key, revision === undefined ? seed : { ...seed, revision })
  return next
}

/**
 * Which edit of this draft it is, treating anything unreadable as the oldest.
 *
 * Zero means "no recorded revision": an empty slot, or a draft written before
 * revisions existed. It is never equal to a revision this session issued, since
 * those start at 1, so a stale save can never mistake one for its own.
 */
export function draftRevision(draft: FeedbackDraft | undefined): number {
  const revision = draft?.revision
  return typeof revision === 'number' && Number.isInteger(revision) && revision > 0
    ? revision
    : 0
}

/** True when the draft's answer is "none of the model candidates". */
export function draftIsNone(draft: FeedbackDraft | undefined): boolean {
  return draft?.correctRank === 0
}

/**
 * The draft's positive choices, in the order they were made.
 *
 * A draft written before identities were stored yields selections with
 * `dirName: null`: the ranks are preserved, but what they meant is not known.
 */
export function draftSelections(draft: FeedbackDraft | undefined): DraftSelection[] {
  if (!draft || draft.correctRank === 0) return []
  if (draft.selections && draft.selections.length > 0) return draft.selections
  const ranks =
    draft.selectedRanks && draft.selectedRanks.length > 0
      ? draft.selectedRanks
      : draft.correctRank !== null && draft.correctRank > 0
        ? [draft.correctRank]
        : []
  return ranks.map((rank) => ({ rank, dirName: null, source: null }))
}

/** The draft with these selections, in this order. Notes are never touched. */
export function draftWithSelections(
  draft: FeedbackDraft,
  selections: DraftSelection[],
): FeedbackDraft {
  if (selections.length === 0) {
    return { correctRank: null, notes: draft.notes }
  }
  return {
    correctRank: selections[0].rank,
    selectedRanks: selections.map((selection) => selection.rank),
    selections,
    notes: draft.notes,
  }
}

/** The draft answering "none of the model candidates", or clearing that answer. */
export function draftWithNone(draft: FeedbackDraft, none: boolean): FeedbackDraft {
  return { correctRank: none ? 0 : null, notes: draft.notes }
}

export function draftsEqual(a: FeedbackDraft, b: FeedbackDraft): boolean {
  if (a.correctRank !== b.correctRank || a.notes !== b.notes) return false
  const as = draftSelections(a)
  const bs = draftSelections(b)
  return (
    as.length === bs.length &&
    as.every(
      (selection, index) =>
        selection.rank === bs[index].rank && selection.dirName === bs[index].dirName,
    )
  )
}

// ---------------------------------------------------------------------------
// Draft keys
// ---------------------------------------------------------------------------

/**
 * Who a draft belongs to, as well as what it is about (issue #157).
 *
 * Decisions are caller-specific -- `GET /api/feedback/latest` already refuses
 * to prefill somebody else's rank -- but local drafts were stored under one
 * query/model/variant key for the whole browser. Reviewer A could leave a rank
 * pressed, sign out, and reviewer B would find it pressed as their own answer,
 * with B's own (empty) server prefill unable to displace a non-empty draft.
 *
 * The variant stays in the key for issue #48's reason: the same query/model
 * ranks differently per variant, and a stored draft must not resurface against
 * a different ranking.
 */
export function makeDraftKey(
  queryId: number,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
  accountId: number | null = null,
): string {
  const scope = accountId === null ? 'anon' : `acct${accountId}`
  return `${scope}:${legacyDraftKey(queryId, model, variant)}`
}

/**
 * The pre-#157 key shape, still read for drafts written before accounts owned
 * them. Nothing writes it: those entries are preserved as they are, offered
 * back as unattributed text, and never adopted as a decision by whichever
 * account signs in next.
 */
export function legacyDraftKey(
  queryId: number,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): string {
  return `${queryId}-${model}-${variant}`
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

function sanitizeSelections(value: unknown): DraftSelection[] | undefined {
  if (!Array.isArray(value)) return undefined
  const selections: DraftSelection[] = []
  for (const entry of value) {
    if (typeof entry !== 'object' || entry === null) continue
    const record = entry as Record<string, unknown>
    if (typeof record.rank !== 'number' || !Number.isInteger(record.rank)) continue
    const dirName = typeof record.dirName === 'string' ? record.dirName : null
    const source =
      record.source === 'model' || record.source === 'reviewer' ? record.source : null
    selections.push({ rank: record.rank, dirName, source })
  }
  return selections.length > 0 ? selections : undefined
}

/**
 * A stored draft, or null.
 *
 * localStorage survives deployments and can hold anything an older build wrote,
 * so every field is checked rather than cast. A draft that cannot be read is
 * dropped from the in-memory map rather than repaired into a decision.
 */
export function sanitizeDraft(value: unknown): FeedbackDraft | null {
  if (typeof value !== 'object' || value === null) return null
  const record = value as Record<string, unknown>
  const notes = typeof record.notes === 'string' ? record.notes : ''
  const correctRank =
    typeof record.correctRank === 'number' && Number.isInteger(record.correctRank)
      ? record.correctRank
      : null
  const selections = sanitizeSelections(record.selections)
  const selectedRanks = Array.isArray(record.selectedRanks)
    ? record.selectedRanks.filter(
        (rank): rank is number => typeof rank === 'number' && Number.isInteger(rank),
      )
    : undefined
  const draft: FeedbackDraft = { correctRank, notes }
  if (selectedRanks && selectedRanks.length > 0) draft.selectedRanks = selectedRanks
  if (selections) draft.selections = selections
  if (
    typeof record.revision === 'number' &&
    Number.isInteger(record.revision) &&
    record.revision > 0
  ) {
    draft.revision = record.revision
  }
  return draft
}

export function serializeDrafts(map: Map<string, FeedbackDraft>): string {
  return JSON.stringify(Array.from(map.entries()))
}

export function deserializeDrafts(raw: string | null): Map<string, FeedbackDraft> {
  if (!raw) return new Map()
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return new Map()
  }
  if (!Array.isArray(parsed)) return new Map()
  const drafts = new Map<string, FeedbackDraft>()
  for (const entry of parsed) {
    if (!Array.isArray(entry) || entry.length !== 2) continue
    const [key, value] = entry
    if (typeof key !== 'string') continue
    const draft = sanitizeDraft(value)
    if (draft !== null) drafts.set(key, draft)
  }
  return drafts
}
