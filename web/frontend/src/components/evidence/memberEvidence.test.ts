import { describe, expect, it } from 'vitest'
import {
  attributionAppliesToWitness,
  describeAttributionScope,
  describeScoreAttribution,
  describeSoleWitness,
  describeUnopenableMembers,
  describeWitnessOption,
  displayedWitnessKey,
  memberEvidenceKey,
  resolveMemberEvidence,
  type MemberEvidenceCandidate,
} from './memberEvidence'

// The half-float values the backend fixture actually returns (issue #163):
// the group's maximum comes from the SECOND member, while the seed -- the
// file the panel has always rendered -- scores a quarter of it.
const GROUP_MAX = 0.7998046875
const SEED_ONLY = 0.199951171875

function reviewerGroup(
  overrides: Partial<MemberEvidenceCandidate> = {},
): MemberEvidenceCandidate {
  return {
    dir_name: 'reviewer-dir-1',
    score: GROUP_MAX,
    dir_files: ['query-2.txt', 'query-0.txt'],
    candidate_files: [
      { filename: 'query-2.txt', text: 'text of query-2.txt' },
      { filename: 'query-0.txt', text: 'text of query-0.txt' },
    ],
    source: 'reviewer',
    label: 'Unattested homily',
    supporting_member: {
      query_id: 0,
      filename: 'query-0.txt',
      score: GROUP_MAX,
    },
    ...overrides,
  }
}

describe('resolveMemberEvidence', () => {
  it('defaults to the supporting witness, not to the first member', () => {
    const evidence = resolveMemberEvidence(reviewerGroup())!
    expect(evidence.selected?.filename).toBe('query-0.txt')
    expect(evidence.displayedIsSupport).toBe(true)
    expect(evidence.scoreScope).toBe('group-maximum')
    // The number is the group's, untouched, and never the seed-only value.
    expect(evidence.score).toBe(GROUP_MAX)
    expect(evidence.score).not.toBe(SEED_ONLY)
  })

  it('keeps the API member order in the selector', () => {
    const evidence = resolveMemberEvidence(reviewerGroup())!
    expect(evidence.witnesses.map((w) => w.filename)).toEqual([
      'query-2.txt',
      'query-0.txt',
    ])
    expect(evidence.witnesses.map((w) => w.position)).toEqual([0, 1])
  })

  it('keeps the aggregate when another member is selected', () => {
    const evidence = resolveMemberEvidence(reviewerGroup(), 'query-2.txt')!
    expect(evidence.selected?.filename).toBe('query-2.txt')
    expect(evidence.displayedIsSupport).toBe(false)
    expect(evidence.score).toBe(GROUP_MAX)
    expect(describeScoreAttribution(evidence).sentence).toContain(
      'Produced by query-0.txt, not by the witness shown below',
    )
  })

  it('discards a selection that is not a member of this candidate', () => {
    const evidence = resolveMemberEvidence(reviewerGroup(), 'query-9.txt')!
    expect(evidence.selected?.filename).toBe('query-0.txt')
    expect(evidence.displayedIsSupport).toBe(true)
  })

  it('never nominates a member when the response names none', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({ supporting_member: null }),
    )!
    expect(evidence.support.kind).toBe('absent')
    expect(evidence.displayedIsSupport).toBe(false)
    // It still has to show something; it just may not call it the source.
    expect(evidence.selected?.filename).toBe('query-2.txt')
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).toContain('does not identify which witness produced it')
    expect(copy.tone).toBe('attention')
  })

  it('treats an absent evidence field the same as an explicit null', () => {
    const candidate = reviewerGroup()
    delete candidate.supporting_member
    expect(resolveMemberEvidence(candidate)!.support.kind).toBe('absent')
  })

  it('names the winning query id when the winner has no filename', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        supporting_member: { query_id: 41, filename: null, score: GROUP_MAX },
      }),
    )!
    expect(evidence.support).toMatchObject({ kind: 'unnamed', queryId: 41 })
    expect(evidence.displayedIsSupport).toBe(false)
    expect(describeScoreAttribution(evidence).sentence).toContain(
      'Produced by member query 41',
    )
  })

  it('says so when the winning witness cannot be opened here', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        candidate_files: [{ filename: 'query-2.txt', text: 'text of query-2' }],
      }),
    )!
    expect(evidence.support).toMatchObject({
      kind: 'named',
      filename: 'query-0.txt',
      inspectable: false,
    })
    expect(evidence.displayedIsSupport).toBe(false)
    expect(describeScoreAttribution(evidence).sentence).toContain(
      'Produced by query-0.txt, which cannot be opened here',
    )
    expect(describeUnopenableMembers(evidence)).toBe(
      '1 of 2 member witnesses cannot be opened here.',
    )
  })

  it('distinguishes a supporting witness with no text from a missing one', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        candidate_files: [
          { filename: 'query-2.txt', text: 'text of query-2.txt' },
          { filename: 'query-0.txt', text: '' },
        ],
      }),
    )!
    expect(evidence.support).toMatchObject({ inspectable: true, hasText: false })
    expect(evidence.displayedIsSupport).toBe(true)
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).toContain('its text is not available')
    expect(copy.tone).toBe('attention')
    expect(describeWitnessOption(evidence.witnesses[1])).toBe(
      'query-0.txt (produced the score, no text available)',
    )
  })

  it('uses plain single-witness copy for a singleton group', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['query-2.txt'],
        candidate_files: [{ filename: 'query-2.txt', text: 'only text' }],
        supporting_member: {
          query_id: 2,
          filename: 'query-2.txt',
          score: SEED_ONLY,
        },
        score: SEED_ONLY,
      }),
    )!
    expect(evidence.scoreScope).toBe('single-witness')
    expect(evidence.selectable).toBe(false)
    const copy = describeScoreAttribution(evidence)
    expect(copy.label).toBe('Similarity')
    expect(copy.tone).toBe('neutral')
    expect(copy.sentence).toBe(
      'Similarity to the witness shown below, the only member this response lists.',
    )
  })

  it('does not call a labelled directory score a member aggregate', () => {
    const evidence = resolveMemberEvidence({
      dir_name: 'CANT.328.12',
      score: 0.91,
      dir_files: ['a.txt', 'b.txt'],
      candidate_files: [
        { filename: 'a.txt', text: 'a' },
        { filename: 'b.txt', text: 'b' },
      ],
      source: 'model',
    })!
    expect(evidence.scoreScope).toBe('directory')
    const copy = describeScoreAttribution(evidence)
    expect(copy.label).toBe('Retrieval score')
    expect(copy.sentence).toContain('not for one file in it')
    expect(copy.tone).toBe('neutral')
  })

  it('returns null for a missing candidate', () => {
    expect(resolveMemberEvidence(null)).toBeNull()
    expect(resolveMemberEvidence(undefined)).toBeNull()
  })

  it('survives a candidate with no readable members at all', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({ candidate_files: null }),
    )!
    expect(evidence.witnesses).toEqual([])
    expect(evidence.selected).toBeNull()
    expect(evidence.selectable).toBe(false)
    expect(describeUnopenableMembers(evidence)).toBe(
      '2 of 2 member witnesses cannot be opened here.',
    )
  })

  it('does not call an empty response a group of one witness', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: [],
        candidate_files: [],
        supporting_member: null,
      }),
    )!
    expect(evidence.memberCount).toBe(0)
    // Nothing is known about the membership, which is not the same as knowing
    // there is exactly one member.
    expect(evidence.scoreScope).not.toBe('single-witness')
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).not.toContain('only witness')
    expect(copy.sentence).toContain('does not identify which witness')
    expect(copy.tone).toBe('attention')
  })

  it('counts a supporting member the lists leave out', () => {
    // The response lists one openable member but credits the score to another
    // file. Whatever the lists say, the group is not a singleton.
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['query-2.txt'],
        candidate_files: [{ filename: 'query-2.txt', text: 'seed text' }],
        supporting_member: {
          query_id: 9,
          filename: 'query-9.txt',
          score: GROUP_MAX,
        },
      }),
    )!
    expect(evidence.memberCount).toBe(2)
    expect(evidence.unopenableCount).toBe(1)
    expect(evidence.scoreScope).toBe('group-maximum')
    expect(evidence.displayedIsSupport).toBe(false)
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).not.toContain('only witness')
    expect(copy.sentence).toContain(
      'Produced by query-9.txt, which cannot be opened here',
    )
    expect(describeUnopenableMembers(evidence)).toBe(
      '1 of 2 member witnesses cannot be opened here.',
    )
  })

  it('refuses to pick a winner when two members share a name', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['same.txt', 'same.txt'],
        candidate_files: [
          { filename: 'same.txt', text: 'first text' },
          { filename: 'same.txt', text: 'second text' },
        ],
        supporting_member: {
          query_id: 4,
          filename: 'same.txt',
          score: GROUP_MAX,
        },
      }),
    )!
    // Opening the first entry and calling it the source would be the very
    // mistake this issue exists to remove.
    expect(evidence.witnesses.some((w) => w.isSupporting)).toBe(false)
    expect(evidence.displayedIsSupport).toBe(false)
    expect(evidence.support).toMatchObject({
      kind: 'named',
      ambiguous: true,
      inspectable: false,
    })
    const copy = describeScoreAttribution(evidence)
    expect(copy.tone).toBe('attention')
    expect(copy.sentence).toContain('more than one witness under that name')
    expect(copy.sentence).not.toContain('The witness shown below produced it')
  })

  it('keeps repeated filenames separately selectable', () => {
    const candidate = reviewerGroup({
      dir_files: ['same.txt', 'same.txt'],
      candidate_files: [
        { filename: 'same.txt', text: 'first text' },
        { filename: 'same.txt', text: 'second text' },
      ],
      supporting_member: null,
    })
    const evidence = resolveMemberEvidence(candidate)!
    const keys = evidence.witnesses.map((w) => w.key)
    expect(new Set(keys).size).toBe(2)
    // Choosing the second entry must open the second text, not the first.
    const second = resolveMemberEvidence(candidate, keys[1])!
    expect(second.selected?.text).toBe('second text')
    expect(second.selected?.position).toBe(1)
    expect(describeWitnessOption(second.selected!)).toBe(
      'same.txt (entry 2 under this name)',
    )
  })

  it('will not call a group a singleton on the word of an unnamed winner', () => {
    // One readable file, and a winner the response identifies only by query
    // id. Nothing pairs that id with the file on screen, so the group may be
    // larger than the list and this file may not have earned the number.
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['query-2.txt'],
        candidate_files: [{ filename: 'query-2.txt', text: 'seed text' }],
        supporting_member: { query_id: 41, filename: null, score: GROUP_MAX },
      }),
    )!
    expect(evidence.scoreScope).toBe('group-maximum')
    expect(evidence.displayedIsSupport).toBe(false)
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).not.toContain('only witness in this group')
    expect(copy.sentence).toContain(
      'Produced by member query 41, which this response does not name',
    )
    expect(copy.tone).toBe('attention')
    expect(describeSoleWitness(evidence)).toBe(
      'Showing query-2.txt, the only member witness available here.',
    )
  })

  it('still calls a proven single-member group what it is', () => {
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['query-2.txt'],
        candidate_files: [{ filename: 'query-2.txt', text: 'only text' }],
        supporting_member: {
          query_id: 2,
          filename: 'query-2.txt',
          score: SEED_ONLY,
        },
        score: SEED_ONLY,
      }),
    )!
    expect(evidence.scoreScope).toBe('single-witness')
    // The score sentence has already named it; a second line would repeat it.
    expect(describeSoleWitness(evidence)).toBeNull()
  })

  it('will not call a legacy payload a group of one', () => {
    // The pre-#163 serializer dropped any member whose filename it could not
    // resolve and still scored it, so a one-file list with no evidence field
    // is exactly the shape that hides a higher-scoring member.
    const evidence = resolveMemberEvidence(
      reviewerGroup({
        dir_files: ['seed.txt'],
        candidate_files: [{ filename: 'seed.txt', text: 'seed text' }],
        supporting_member: null,
      }),
    )!
    expect(evidence.scoreScope).toBe('group-maximum')
    expect(evidence.displayedIsSupport).toBe(false)
    const copy = describeScoreAttribution(evidence)
    expect(copy.sentence).not.toContain('only member this response lists')
    expect(copy.sentence).toContain('does not identify which witness produced it')
    expect(copy.tone).toBe('attention')
    expect(describeSoleWitness(evidence)).toBe(
      'Showing seed.txt, the only member witness available here.',
    )
  })

  it('leaves the key equal to the filename when names are unique', () => {
    const evidence = resolveMemberEvidence(reviewerGroup())!
    expect(evidence.witnesses.map((w) => w.key)).toEqual([
      'query-2.txt',
      'query-0.txt',
    ])
  })
})

describe('memberEvidenceKey', () => {
  const base = {
    queryId: 7,
    model: 'bowphs_LaTa',
    variant: 'sif_abtt',
    generation: 3,
    dirName: 'reviewer-dir-1',
    source: 'reviewer' as const,
  }

  it('changes with every part of the prediction identity', () => {
    const key = memberEvidenceKey(base)
    expect(memberEvidenceKey({ ...base, queryId: 8 })).not.toBe(key)
    expect(memberEvidenceKey({ ...base, model: 'google_mt5-base' })).not.toBe(key)
    expect(memberEvidenceKey({ ...base, variant: 'raw' })).not.toBe(key)
    expect(memberEvidenceKey({ ...base, generation: 4 })).not.toBe(key)
    expect(memberEvidenceKey({ ...base, dirName: 'reviewer-dir-2' })).not.toBe(key)
  })

  it('is stable for the same identity', () => {
    expect(memberEvidenceKey({ ...base })).toBe(memberEvidenceKey({ ...base }))
  })

  it('cannot be confused by a directory name that looks like a separator', () => {
    expect(
      memberEvidenceKey({ ...base, dirName: 'a', model: 'b:c' }),
    ).not.toBe(memberEvidenceKey({ ...base, dirName: 'a:b', model: 'c' }))
  })

  it('re-keys the displayed witness when only the member changes', () => {
    const key = memberEvidenceKey(base)
    expect(displayedWitnessKey(key, 'query-0.txt')).not.toBe(
      displayedWitnessKey(key, 'query-2.txt'),
    )
  })
})

describe('attributionAppliesToWitness', () => {
  it('accepts the artifact that names the displayed labelled witness', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: true })
  })

  it('refuses an artifact built from a different file in the directory', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
        dirName: 'CANT.328.12',
        filename: 'b.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'other-witness' })
  })

  it('refuses a same-named file from another directory', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/CSAR.347.17/a.txt',
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'other-witness' })
  })

  it('refuses a same-named file of a different provenance', () => {
    // Basename and parent can both match while the witness is the unlabelled
    // document rather than the labelled reference. Provenance decides.
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_unlabelled/CANT.328.12/a.txt',
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'other-witness' })
  })

  it('never applies model attribution to a reviewer-group witness', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/query-0.txt/query-0.txt',
        dirName: 'reviewer-dir-1',
        filename: 'query-0.txt',
        source: 'reviewer',
      }),
    ).toEqual({ applicable: false, reason: 'reviewer-witness' })
  })

  it('separates "no artifact" from "artifact that names nothing"', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: null,
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'no-artifact' })
    expect(
      attributionAppliesToWitness({
        candidatePath: '',
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'unverifiable' })
  })

  it('cannot verify an artifact against an unnamed witness', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
        dirName: 'CANT.328.12',
        filename: null,
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'unverifiable' })
  })

  it('tolerates a bare directory/file path from the artifact index', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'CANT.328.12/c.txt',
        dirName: 'CANT.328.12',
        filename: 'c.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: true })
  })

  it('will not verify an artifact path that names no directory', () => {
    // A basename alone cannot prove the artifact was built from the witness
    // on screen: the same filename occurs in many directories.
    expect(
      attributionAppliesToWitness({
        candidatePath: 'a.txt',
        dirName: 'CANT.328.12',
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'unverifiable' })
  })

  it('will not verify an artifact when the displayed directory is unknown', () => {
    expect(
      attributionAppliesToWitness({
        candidatePath: 'data/canon_labelled/OTHER/a.txt',
        dirName: null,
        filename: 'a.txt',
        source: 'model',
      }),
    ).toEqual({ applicable: false, reason: 'unverifiable' })
  })
})

describe('describeAttributionScope', () => {
  it('stays silent when the panel already explains a missing artifact', () => {
    expect(
      describeAttributionScope({ applicable: false, reason: 'no-artifact' }),
    ).toBeNull()
    expect(describeAttributionScope({ applicable: true })).toBeNull()
  })

  it('labels word overlap as lexical, not as evidence for the cosine', () => {
    const sentence = describeAttributionScope(
      { applicable: false, reason: 'reviewer-witness' },
      { lexicalHighlighting: true },
    )
    expect(sentence).toContain('No model attribution exists')
    expect(sentence).toContain(
      'word overlap, not evidence for the similarity score',
    )
  })

  it('says the attribution belongs to another witness', () => {
    expect(
      describeAttributionScope({ applicable: false, reason: 'other-witness' }),
    ).toContain('describes a different witness')
  })
})
