import type { CandidateFile, CandidateSource } from '../../api/queries'

/**
 * The witness behind a directory candidate's number (issue #163).
 *
 * A reviewer directory's similarity is the MAXIMUM over its member witnesses
 * (web/services/qq_matrix.py), so it is earned by a member rather than by
 * whichever member happens to be rendered. The candidate panel has always
 * rendered `candidate_files[0]`, normally the seed, which in the fixture
 * behind #163 scores 0.199951171875 while the directory's number is
 * 0.7998046875 and was reached by a different member entirely.
 *
 * This module turns a candidate payload into the small view model the panel
 * needs to say that honestly, and nothing else: it does not fetch, score,
 * re-rank, re-order members or decide which prediction is current.
 */

// ---------------------------------------------------------------------------
// Wire shape
// ---------------------------------------------------------------------------

/**
 * `Prediction.supporting_member` from the backend half of #163.
 *
 * `query_id` identifies the SUPPORTING MEMBER, not the query under review
 * (that stays `PredictionResponse.file_id`). `score` is `Prediction.score`
 * exactly: the current query's group maximum, never `ReviewerDir.
 * best_match_score` and never a per-member score for any other member.
 * `filename` is null when the winning member has no filename metadata; the
 * winner is still that member, so nothing here may substitute a readable one.
 *
 * It is a DESIGNATION, not a uniqueness claim. `score_with_support` resolves
 * equal maxima by the smallest member query id, and a tied payload is
 * byte-identical to a strictly-won one, so nothing on the wire says whether
 * another member reached the same number. No copy in this module may deny it.
 */
export interface SupportingMember {
  query_id: number
  filename: string | null
  score: number
}

/**
 * The part of a prediction this module reads.
 *
 * Structurally a superset of `Prediction` plus the additive
 * `supporting_member`, which cannot be declared on the shared `Prediction`
 * interface until #156 hands over `api/queries.ts`. A real `Prediction` is
 * assignable to this type, so no consumer has to wait for that handoff.
 */
export interface MemberEvidenceCandidate {
  dir_name: string
  score: number
  dir_files: string[]
  candidate_files: CandidateFile[] | null
  source?: CandidateSource
  label?: string | null
  supporting_member?: SupportingMember | null
}

// ---------------------------------------------------------------------------
// View model
// ---------------------------------------------------------------------------

export interface MemberWitness {
  filename: string
  /**
   * Stable identity of this entry, and the value the selector round-trips.
   *
   * It is the filename whenever the filename is unique in the response, which
   * is the only shape production serves (unlabelled queries live flat in one
   * directory, so two members cannot share a name). When a response does
   * repeat a name, the position disambiguates it, so choosing the second
   * entry cannot silently open the first.
   */
  key: string
  text: string
  /** Position in the API's member order, which is never re-sorted here. */
  position: number
  /** False when the response carries the member but no text for it. */
  hasText: boolean
  /** True only for the member the backend named as the score's source. */
  isSupporting: boolean
}

/**
 * What the response says about the member that produced the number.
 *
 * `absent` is a legacy or model payload with no evidence field at all, and is
 * NOT permission to nominate the first member. `unnamed` is a real winner
 * whose filename metadata is missing. `named` may still be un-openable here,
 * because a member with no readable text is skipped when the candidate files
 * are built.
 */
export type SupportState =
  | {
      kind: 'named'
      queryId: number
      filename: string
      score: number
      /** The named member is among this response's openable witnesses. */
      inspectable: boolean
      hasText: boolean
      /**
       * The response lists more than one witness under that name, so which
       * one produced the number cannot be told from the payload. Never
       * resolved by picking the first: an ambiguous winner is stated as
       * ambiguous.
       */
      ambiguous: boolean
    }
  | { kind: 'unnamed'; queryId: number; score: number }
  | { kind: 'absent' }

/**
 * What the number covers. Reviewer groups aggregate over members; a labelled
 * directory's number comes from the retrieval run and is not attributed to
 * any single file in it.
 */
export type ScoreScope = 'group-maximum' | 'single-witness' | 'directory'

export interface MemberEvidence {
  scoreScope: ScoreScope
  score: number
  /** Members carrying text, in the API's member order. */
  witnesses: MemberWitness[]
  /** Members the response names but cannot open. */
  unopenableCount: number
  /** Members the directory has, openable or not. */
  memberCount: number
  support: SupportState
  selected: MemberWitness | null
  /** True only when the rendered witness is provably the score's source. */
  displayedIsSupport: boolean
  /** More than one witness can be opened, so a selector is worth rendering. */
  selectable: boolean
}

function normaliseWitnesses(
  candidate: MemberEvidenceCandidate,
  supportFilename: string | null,
): MemberWitness[] {
  const files = candidate.candidate_files ?? []
  const occurrences = new Map<string, number>()
  for (const file of files) {
    occurrences.set(file.filename, (occurrences.get(file.filename) ?? 0) + 1)
  }
  return files.map((file, position) => {
    const repeated = (occurrences.get(file.filename) ?? 0) > 1
    return {
      filename: file.filename,
      key: repeated ? `${file.filename}#${position}` : file.filename,
      text: file.text ?? '',
      position,
      hasText: (file.text ?? '').trim().length > 0,
      // A repeated name cannot mark any single entry as the source.
      isSupporting:
        !repeated && supportFilename != null && file.filename === supportFilename,
    }
  })
}

function resolveSupport(
  candidate: MemberEvidenceCandidate,
  witnesses: MemberWitness[],
): SupportState {
  const support = candidate.supporting_member
  if (support == null) return { kind: 'absent' }
  if (support.filename == null) {
    return { kind: 'unnamed', queryId: support.query_id, score: support.score }
  }
  const matches = witnesses.filter((w) => w.filename === support.filename)
  const unique = matches.length === 1 ? matches[0] : null
  return {
    kind: 'named',
    queryId: support.query_id,
    filename: support.filename,
    score: support.score,
    inspectable: unique != null,
    hasText: unique?.hasText ?? false,
    ambiguous: matches.length > 1,
  }
}

/**
 * Build the view model for one candidate.
 *
 * `selectedFilename` is the reviewer's manual choice, already scoped to a
 * prediction identity by `useSelectedMember`. A choice that is not in this
 * candidate's members is discarded rather than honoured, which is what makes
 * a same-rank/different-directory refresh safe.
 */
export function resolveMemberEvidence(
  candidate: MemberEvidenceCandidate | null | undefined,
  selectedKey: string | null = null,
): MemberEvidence | null {
  if (!candidate) return null

  const supportFilename = candidate.supporting_member?.filename ?? null
  const witnesses = normaliseWitnesses(candidate, supportFilename)
  const support = resolveSupport(candidate, witnesses)

  const named = candidate.dir_files ?? []
  const openable = new Set(witnesses.map((w) => w.filename))
  // A named supporting member the response neither lists nor opens is still a
  // member, and proves the group is larger than the list however short the
  // list is. Counting it is what stops a witness that did not produce the
  // number from being presented as all there is.
  const supportUncounted =
    support.kind === 'named' &&
    !openable.has(support.filename) &&
    !named.includes(support.filename)
  const memberCount =
    Math.max(named.length, witnesses.length) + (supportUncounted ? 1 : 0)
  const namedButClosed = named.filter((name) => !openable.has(name)).length
  const unopenableCount = Math.max(
    namedButClosed,
    memberCount - witnesses.length,
  )

  // Defaulting to the supporting witness is the whole point; falling back to
  // the first member when it is not openable is a display fallback only, and
  // `displayedIsSupport` stays false so no copy can call it the source.
  const requested =
    selectedKey != null
      ? (witnesses.find((w) => w.key === selectedKey) ?? null)
      : null
  const supportWitness = witnesses.find((w) => w.isSupporting) ?? null
  const selected = requested ?? supportWitness ?? witnesses[0] ?? null

  const isReviewer = candidate.source === 'reviewer'
  // A response's lists are not a census of the group. The serializer drops any
  // member whose filename it cannot resolve while still scoring it, so one
  // listed file does not mean one member, and no absent or unnamed evidence
  // field can make it mean that. The only wording that survives this is a
  // claim about the response rather than about the group, and it is reserved
  // for the case where the one listed witness is also the named, unambiguous
  // source of the number.
  const provablySole =
    memberCount === 1 &&
    witnesses.length === 1 &&
    support.kind === 'named' &&
    !support.ambiguous &&
    support.inspectable &&
    selected?.isSupporting === true
  const scoreScope: ScoreScope = !isReviewer
    ? 'directory'
    : provablySole
      ? 'single-witness'
      : 'group-maximum'

  return {
    scoreScope,
    score: candidate.score,
    witnesses,
    unopenableCount,
    memberCount,
    support,
    selected,
    displayedIsSupport:
      support.kind === 'named' &&
      !support.ambiguous &&
      selected != null &&
      selected.isSupporting,
    selectable: witnesses.length > 1,
  }
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

export interface ScoreAttributionCopy {
  /** Short label for the number, e.g. "Group maximum". */
  label: string
  /** One or two sentences naming what the number covers and who produced it. */
  sentence: string
  /**
   * `attention` means the reader would otherwise credit the number to the
   * witness in front of them. Never styling alone: the sentence says it too.
   */
  tone: 'neutral' | 'attention'
}

const SCOPE_LABEL: Record<ScoreScope, string> = {
  'group-maximum': 'Group maximum',
  'single-witness': 'Similarity',
  directory: 'Retrieval score',
}

const SCOPE_LEAD: Record<ScoreScope, string> = {
  'group-maximum':
    "Highest similarity across this group's scorable member witnesses.",
  'single-witness':
    'Similarity to the witness shown below, the only member this response lists.',
  directory:
    'Score for the whole directory from the retrieval run, not for one file in it.',
}

/**
 * The sentence that keeps an aggregate from reading as an individual score.
 *
 * Shared with the reviewer candidate card, whose preview text now comes from
 * the supporting member too, so the card and the panel attribute the same
 * number to the same witness.
 */
export function describeScoreAttribution(
  evidence: MemberEvidence,
): ScoreAttributionCopy {
  const label = SCOPE_LABEL[evidence.scoreScope]
  const lead = SCOPE_LEAD[evidence.scoreScope]
  const { support, displayedIsSupport } = evidence

  if (evidence.scoreScope === 'directory') {
    return { label, sentence: lead, tone: 'neutral' }
  }

  if (support.kind === 'absent') {
    // A missing evidence field is not a licence to nominate the seed, and it
    // cannot reach the sole-witness wording, which requires named support.
    return {
      label,
      sentence: `${lead} This response does not identify which witness produced it.`,
      tone: 'attention',
    }
  }

  if (support.kind === 'unnamed') {
    return {
      label,
      sentence: `${lead} Produced by member query ${support.queryId}, which this response does not name.`,
      tone: 'attention',
    }
  }

  if (support.ambiguous) {
    return {
      label,
      sentence: `${lead} Produced by a member named ${support.filename}, and this group lists more than one witness under that name, so this response cannot say which.`,
      tone: 'attention',
    }
  }

  if (!support.inspectable) {
    return {
      label,
      sentence: `${lead} Produced by ${support.filename}, which cannot be opened here.`,
      tone: 'attention',
    }
  }

  if (!displayedIsSupport) {
    // NOT "not by the witness shown below". The backend designates one
    // maximising member and breaks equal maxima by the smallest query id, so
    // the member on screen may have reached the same number; the response
    // carries no per-member score that could tell the reviewer either way.
    return {
      label,
      sentence: `${lead} This response designates ${support.filename} as the supporting witness. The individual similarity of the witness shown below is not supplied.`,
      tone: 'attention',
    }
  }

  if (!support.hasText) {
    return {
      label,
      sentence: `${lead} The witness shown below produced it, but its text is not available.`,
      tone: 'attention',
    }
  }

  return {
    label,
    // With one member there is nobody else it could have come from, so the
    // lead already says everything and a second sentence is noise.
    sentence:
      evidence.scoreScope === 'single-witness'
        ? lead
        : `${lead} The witness shown below produced it.`,
    tone: 'neutral',
  }
}

/** Selector option text. Never colour alone: the marker is in the words. */
export function describeWitnessOption(witness: MemberWitness): string {
  const notes: string[] = []
  if (witness.isSupporting) notes.push('produced the score')
  // Two entries under one name would otherwise be indistinguishable choices.
  if (witness.key !== witness.filename) {
    notes.push(`entry ${witness.position + 1} under this name`)
  }
  if (!witness.hasText) notes.push('no text available')
  return notes.length > 0
    ? `${witness.filename} (${notes.join(', ')})`
    : witness.filename
}

/**
 * The line under the number when there is nothing to choose between.
 *
 * A response's member list is not a census: members whose filename cannot be
 * resolved are dropped from it and scored anyway. So this says "the only
 * member witness available here", about the response, and never that the
 * group has one member. It is null in the sole-witness case, where the score
 * sentence has already said it.
 */
export function describeSoleWitness(evidence: MemberEvidence): string | null {
  if (evidence.selectable || evidence.selected == null) return null
  if (evidence.scoreScope === 'single-witness') return null
  return `Showing ${evidence.selected.filename}, the only member witness available here.`
}

/** "1 of 3 member witnesses cannot be opened here", or null when all can. */
export function describeUnopenableMembers(
  evidence: MemberEvidence,
): string | null {
  if (evidence.unopenableCount <= 0) return null
  if (evidence.memberCount <= 1) {
    return 'The one member witness this response lists cannot be opened here.'
  }
  return `${evidence.unopenableCount} of ${evidence.memberCount} member witnesses cannot be opened here.`
}

// ---------------------------------------------------------------------------
// Prediction identity
// ---------------------------------------------------------------------------

/**
 * Everything a member choice belongs to.
 *
 * Rank is deliberately absent: reviewer directories are anchored at rank 11,
 * so the same rank can name a different directory after a refresh, and a
 * choice scoped to a rank would survive into a group that never contained
 * that member. `generation` comes from the shared prediction state (#156) and
 * is what makes a refresh of the same key a new identity.
 */
export interface MemberEvidenceIdentity {
  queryId: number | null
  model: string
  variant: string
  generation: number | string
  dirName: string | null
  source?: CandidateSource
}

export function memberEvidenceKey(identity: MemberEvidenceIdentity): string {
  return JSON.stringify([
    identity.queryId,
    identity.model,
    identity.variant,
    identity.generation,
    identity.dirName,
    identity.source ?? null,
  ])
}

/**
 * Identity of the exact witness on screen.
 *
 * Token-map artifacts, pins and hover state are all about a pair, so they
 * must be re-keyed when the member changes, not only when the directory does.
 */
export function displayedWitnessKey(
  identityKey: string,
  witnessKey: string | null,
): string {
  return `${identityKey}::${witnessKey ?? ''}`
}

// ---------------------------------------------------------------------------
// Model attribution applicability
// ---------------------------------------------------------------------------

export type AttributionMismatchReason =
  /** No token map for this pair at all. */
  | 'no-artifact'
  /** The artifact does not say which witness it describes. */
  | 'unverifiable'
  /** Reviewer-group members are originally unlabelled witnesses. */
  | 'reviewer-witness'
  /** The artifact describes a different file, or a different provenance. */
  | 'other-witness'

export type AttributionScope =
  | { applicable: true }
  | { applicable: false; reason: AttributionMismatchReason }

const UNLABELLED_SEGMENTS = new Set(['canon_unlabelled', 'unlabelled'])

/**
 * Does the loaded token map describe the witness actually on screen?
 *
 * `/api/token_map` is keyed by candidate DIRECTORY, so its answer is about
 * whatever file the artifact was built from. Painting that matrix over a
 * different member would attribute one witness's model evidence to another.
 * `candidate_path` is the only field that identifies the artifact's witness,
 * and it carries provenance as well as a name (`data/canon_labelled/<dir>/
 * <file>.txt`), which is why a bare basename match is not enough.
 *
 * Reviewer-group members are refused outright: their witnesses are originally
 * unlabelled query documents, no artifact pairs a query with another query,
 * and #163 does not generate any.
 */
export function attributionAppliesToWitness(params: {
  candidatePath?: string | null
  dirName?: string | null
  filename?: string | null
  source?: CandidateSource
}): AttributionScope {
  const { candidatePath, dirName, filename, source } = params

  if (source === 'reviewer') {
    return { applicable: false, reason: 'reviewer-witness' }
  }
  if (candidatePath == null) {
    return { applicable: false, reason: 'no-artifact' }
  }
  if (candidatePath.trim() === '' || !filename) {
    return { applicable: false, reason: 'unverifiable' }
  }

  const segments = candidatePath.split('/').filter((s) => s.length > 0)
  const basename = segments[segments.length - 1]
  const parent = segments.length > 1 ? segments[segments.length - 2] : null

  if (segments.some((s) => UNLABELLED_SEGMENTS.has(s))) {
    return { applicable: false, reason: 'other-witness' }
  }
  if (basename !== filename) {
    return { applicable: false, reason: 'other-witness' }
  }
  // A basename alone does not say which directory's witness was attributed,
  // and directory names repeat filenames across the corpus. Without both
  // sides of that comparison the artifact is unverified, not verified.
  if (parent == null || dirName == null) {
    return { applicable: false, reason: 'unverifiable' }
  }
  if (parent !== dirName) {
    return { applicable: false, reason: 'other-witness' }
  }
  return { applicable: true }
}

/**
 * Why the highlights are not model attribution for this witness.
 *
 * `no-artifact` returns null on purpose: the candidate panel already carries
 * one notice for a pair with no token map, and saying it twice would be
 * noise. `lexicalHighlighting` says whether word-overlap shading is on, which
 * must never be read as evidence for the cosine.
 */
export function describeAttributionScope(
  scope: AttributionScope,
  options: { lexicalHighlighting?: boolean } = {},
): string | null {
  if (scope.applicable) return null

  let sentence: string
  switch (scope.reason) {
    case 'no-artifact':
      return null
    case 'reviewer-witness':
      sentence =
        'No model attribution exists for a reviewer-group witness, so none is shown.'
      break
    case 'other-witness':
      sentence =
        'The available attribution describes a different witness, so it is not applied here.'
      break
    case 'unverifiable':
      sentence =
        'The available attribution does not identify its witness, so it is not applied here.'
      break
  }

  if (options.lexicalHighlighting) {
    sentence +=
      ' Any highlighting below is word overlap, not evidence for the similarity score.'
  }
  return sentence
}
