import { useEffect, useMemo, useState } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { usePredictions } from '../../api/queries'
import { REVIEWER_DIRS_UPDATED_EVENT } from '../../api/reviewerDirs'
import ModelSelector from '../predictions/ModelSelector'
import MemberEvidenceBar from './MemberEvidenceBar'
import {
  attributionAppliesToWitness,
  memberEvidenceKey,
  resolveMemberEvidence,
  type MemberEvidenceCandidate,
} from './memberEvidence'
import { useSelectedMember } from './useSelectedMember'

// Issue #163's fixture, at the values the backend actually returns. q1 is
// under review; the group holds q2 (its seed, first in member order) and q0
// (filed into it later). The seed alone scores SEED_ONLY against q1, so the
// witness the panel has always rendered is not the one behind the number.
const GROUP_MAX = 0.7998046875
const SEED_ONLY = 0.199951171875

const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'google_mt5-base'
/** Deployed without a q-q matrix, so it serves no reviewer candidates. */
const MODEL_NO_MATRIX = 'KaLM_mini'

const QUERY_ID = 1
const OTHER_QUERY_ID = 8
/** Where reviewer directories are anchored (MAX_MODEL_RANK + 1). */
const REVIEWER_RANK = 11

interface GroupFixture {
  dirName: string
  label: string
  members: { filename: string; text: string }[]
  /** Members the directory has but this response cannot open. */
  closedMembers?: string[]
  supporting: { query_id: number; filename: string | null; score: number } | null
  score?: number
}

function defaultGroup(): GroupFixture {
  return {
    dirName: 'reviewer-dir-1',
    label: 'Unattested homily',
    members: [
      { filename: 'query-2.txt', text: 'text of query-2.txt' },
      { filename: 'query-0.txt', text: 'text of query-0.txt' },
    ],
    supporting: { query_id: 0, filename: 'query-0.txt', score: GROUP_MAX },
  }
}

function groupPayload(group: GroupFixture) {
  const openable = group.members.map((m) => m.filename)
  return {
    rank: REVIEWER_RANK,
    dir_name: group.dirName,
    score: group.score ?? GROUP_MAX,
    dir_files: [...openable, ...(group.closedMembers ?? [])],
    preview_text:
      group.members.find((m) => m.filename === group.supporting?.filename)
        ?.text ?? '',
    candidate_files: group.members,
    source: 'reviewer' as const,
    label: group.label,
    created_by: 'Abigail',
    seed_query_id: 2,
    supporting_member: group.supporting,
  }
}

function modelCard(rank: number, score: number) {
  return {
    rank,
    dir_name: `CANT.328.1${rank}`,
    score,
    dir_files: ['a.txt'],
    preview_text: 'labelled preview',
    candidate_files: [{ filename: 'a.txt', text: 'labelled text' }],
    source: 'model' as const,
    supporting_member: null,
  }
}

/** Per `${model}:${queryId}`; a missing entry means "no reviewer group". */
let groups: Record<string, GroupFixture | null> = {}
let predictionDelay = 0
let requestCount = 0
/**
 * Every `directory:witness` pairing that was RENDERED, not only the ones that
 * survived to the end of a test. A stale witness beside a new directory's
 * number is a painting bug, and `waitFor` cannot see a commit that a later
 * effect repairs.
 */
let painted: string[] = []

function modelEntry(slug: string, displayName: string, matrix: boolean) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'sif',
    prediction_count: 2238,
    available_variants: ['sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.5, verify: 0.7 },
    supports_reviewer_dirs: matrix,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const sleep = (ms: number) =>
  ms > 0 ? new Promise((r) => setTimeout(r, ms)) : Promise.resolve()

function installFetch(): void {
  requestCount = 0
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const parsed = new URL(url, 'http://localhost')

    if (url.includes('/api/models')) {
      return jsonResponse([
        modelEntry(MODEL_A, 'LaTa (T5)', true),
        modelEntry(MODEL_NO_MATRIX, 'KaLM-mini', false),
        modelEntry(MODEL_B, 'mT5-base', true),
      ])
    }

    if (url.includes('/predictions')) {
      requestCount += 1
      const model = parsed.searchParams.get('model') ?? ''
      const variant = parsed.searchParams.get('variant') ?? ''
      const queryId = Number(parsed.pathname.split('/')[3])
      await sleep(predictionDelay)
      const group = groups[`${model}:${queryId}`]
      return jsonResponse({
        file_id: queryId,
        filename: `query-${queryId}.txt`,
        model,
        variant,
        predictions: [
          modelCard(1, 0.44),
          ...(group ? [groupPayload(group)] : []),
        ],
        seeded_dirs: [],
      })
    }

    return jsonResponse({})
  })
  vi.stubGlobal('fetch', fetchMock)
}

/**
 * What CenterArea will do once #156 publishes the shared prediction state:
 * take the authoritative current result, scope the member choice to its
 * identity, and render the bar beside the witness it selected.
 *
 * `generation` stands in for that state's request generation. The rest is the
 * real hook over the real fetch path, so the payload these tests assert on is
 * the one the backend serialises.
 */
function EvidenceHarness({ generation }: { generation: number }) {
  const { activeQueryId, activeModel, activeVariant, activePredictionRank } =
    useApp()
  const predictions = usePredictions(activeQueryId, activeModel, activeVariant)

  const current = useMemo(() => {
    const data = predictions.data
    if (!data?.predictions) return null
    if (data.model !== activeModel) return null
    if (data.variant !== activeVariant) return null
    if (data.file_id !== activeQueryId) return null
    return (
      data.predictions.find((p) => p.rank === activePredictionRank) ?? null
    )
  }, [
    predictions.data,
    activeModel,
    activeVariant,
    activeQueryId,
    activePredictionRank,
  ])

  const candidate = current as MemberEvidenceCandidate | null

  const identityKey = memberEvidenceKey({
    queryId: activeQueryId,
    model: activeModel,
    variant: activeVariant,
    generation,
    dirName: candidate?.dir_name ?? null,
    source: candidate?.source,
  })
  const selection = useSelectedMember(identityKey)
  const evidence = resolveMemberEvidence(candidate, selection.filename)

  painted.push(
    `${candidate?.dir_name ?? '-'}:${evidence?.selected?.filename ?? '-'}`,
  )

  const attribution = evidence?.selected
    ? attributionAppliesToWitness({
        // No token map is loaded in this harness; the reviewer-witness rule
        // is decided before any artifact is consulted.
        candidatePath: null,
        dirName: candidate?.dir_name ?? null,
        filename: evidence.selected.filename,
        source: candidate?.source,
      })
    : null

  return (
    <>
      <MemberEvidenceBar
        evidence={evidence}
        onSelectWitness={selection.select}
        attribution={attribution}
      />
      <div data-testid="witness-name">{evidence?.selected?.filename ?? ''}</div>
      <div data-testid="witness-text">{evidence?.selected?.text ?? ''}</div>
      <div data-testid="candidate-dir">{candidate?.dir_name ?? ''}</div>
    </>
  )
}

function Harness() {
  const { setActiveQueryId, setActiveModel, setActivePredictionRank } = useApp()
  const [generation, setGeneration] = useState(0)

  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL_A)
    setActivePredictionRank(REVIEWER_RANK)
  }, [setActiveQueryId, setActiveModel, setActivePredictionRank])

  return (
    <>
      <ModelSelector />
      <button type="button" onClick={() => setActiveQueryId(OTHER_QUERY_ID)}>
        Go to other query
      </button>
      <button
        type="button"
        onClick={() => {
          // A directory write invalidates every query's candidates, which is
          // exactly the refresh that can put a different directory behind the
          // same rank.
          setGeneration((g) => g + 1)
          window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
        }}
      >
        Refresh predictions
      </button>
      <button
        type="button"
        onClick={() => {
          // The harder half of the same case: the server's answer changes
          // while nothing on the client does. The only thing that moves is
          // the directory inside the payload, and it moves in the very render
          // that carries the new data.
          window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
        }}
      >
        Refresh from server
      </button>
      <EvidenceHarness generation={generation} />
    </>
  )
}

function renderHarness() {
  return render(
    <AppProvider>
      <Harness />
    </AppProvider>,
  )
}

const witnessName = () => screen.getByTestId('witness-name').textContent
const witnessText = () => screen.getByTestId('witness-text').textContent
const attributionText = () =>
  screen.getByTestId('member-evidence-attribution').textContent ?? ''
const selector = () => screen.getByTestId('member-selector') as HTMLSelectElement
const optionLabels = () =>
  Array.from(selector().options).map((o) => o.textContent ?? '')

async function chooseWitness(filename: string) {
  await userEvent.selectOptions(selector(), filename)
}

beforeEach(() => {
  predictionDelay = 0
  painted = []
  groups = {
    [`${MODEL_A}:${QUERY_ID}`]: defaultGroup(),
    [`${MODEL_B}:${QUERY_ID}`]: {
      dirName: 'reviewer-dir-2',
      label: 'Second reading',
      members: [
        { filename: 'query-5.txt', text: 'text of query-5.txt' },
        { filename: 'query-6.txt', text: 'text of query-6.txt' },
      ],
      supporting: { query_id: 6, filename: 'query-6.txt', score: 0.61 },
      score: 0.61,
    },
    [`${MODEL_A}:${OTHER_QUERY_ID}`]: {
      dirName: 'reviewer-dir-3',
      label: 'Third reading',
      members: [
        { filename: 'query-7.txt', text: 'text of query-7.txt' },
        { filename: 'query-9.txt', text: 'text of query-9.txt' },
      ],
      supporting: { query_id: 9, filename: 'query-9.txt', score: 0.55 },
      score: 0.55,
    },
  }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('member evidence over the real prediction fetch (issue #163)', () => {
  it('opens the witness that produced the number, not the seed', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(witnessText()).toBe('text of query-0.txt')

    // The number stays the group's, to the digit the API sent.
    expect(screen.getByText('0.800')).toBeTruthy()
    expect(screen.getByText('Group maximum')).toBeTruthy()
    // The seed-only value is not this candidate's score and never appears.
    expect(screen.queryByText(SEED_ONLY.toFixed(3))).toBeNull()
    expect(attributionText()).toContain('The witness shown below produced it.')
  })

  it('keeps the group maximum when the reviewer opens another member', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))

    await chooseWitness('query-2.txt')

    expect(witnessName()).toBe('query-2.txt')
    expect(witnessText()).toBe('text of query-2.txt')
    // Unchanged aggregate, and no invented per-member number beside it.
    expect(screen.getByText('0.800')).toBeTruthy()
    expect(screen.queryByText(SEED_ONLY.toFixed(3))).toBeNull()
    expect(attributionText()).toContain(
      'Produced by query-0.txt, not by the witness shown below.',
    )
  })

  it('puts the supporting witness one click away', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')
    expect(screen.getByTestId('show-supporting-witness')).toBeTruthy()

    await userEvent.click(screen.getByTestId('show-supporting-witness'))

    expect(witnessName()).toBe('query-0.txt')
    // The escape hatch disappears once it would be a no-op.
    expect(screen.queryByTestId('show-supporting-witness')).toBeNull()
  })

  it('keeps the API member order and marks the supporting one in words', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(optionLabels()).toEqual([
      'query-2.txt',
      'query-0.txt (produced the score)',
    ])
  })

  it('drops the chosen member when the model changes', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')

    await userEvent.selectOptions(
      screen.getByLabelText('Select model'),
      MODEL_B,
    )

    await waitFor(() => expect(witnessName()).toBe('query-6.txt'))
    expect(screen.getByTestId('candidate-dir').textContent).toBe(
      'reviewer-dir-2',
    )
    expect(optionLabels()).toEqual([
      'query-5.txt',
      'query-6.txt (produced the score)',
    ])
    expect(screen.getByText('0.610')).toBeTruthy()

    // Coming back re-defaults to that model's supporting witness rather than
    // restoring a choice made against a different group.
    await userEvent.selectOptions(
      screen.getByLabelText('Select model'),
      MODEL_A,
    )
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
  })

  it('drops the chosen member when the query changes', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')

    await userEvent.click(screen.getByText('Go to other query'))

    await waitFor(() => expect(witnessName()).toBe('query-9.txt'))
    expect(witnessText()).toBe('text of query-9.txt')
  })

  it('masks the previous witness on the first render after a switch', async () => {
    // The defect this guards is a paint, not an end state: an effect-based
    // reset would still show the previous group's witness for one commit.
    predictionDelay = 40
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')

    await userEvent.selectOptions(
      screen.getByLabelText('Select model'),
      MODEL_B,
    )

    // Synchronously after the switch, while model B is still in flight.
    expect(screen.queryByTestId('member-evidence')).toBeNull()
    expect(witnessName()).toBe('')
    expect(witnessText()).toBe('')

    await waitFor(() => expect(witnessName()).toBe('query-6.txt'))
  })

  it('does not carry a member into a different directory at the same rank', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')
    expect(witnessName()).toBe('query-2.txt')

    // Same query, model and variant, same rank 11: only the directory behind
    // that rank changed, which a rank-scoped selection would not notice.
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      dirName: 'reviewer-dir-9',
      label: 'Re-scored group',
      members: [
        { filename: 'query-2.txt', text: 'text of query-2.txt' },
        { filename: 'query-4.txt', text: 'text of query-4.txt' },
      ],
      supporting: { query_id: 4, filename: 'query-4.txt', score: 0.72 },
      score: 0.72,
    }
    await userEvent.click(screen.getByText('Refresh predictions'))

    await waitFor(() =>
      expect(screen.getByTestId('candidate-dir').textContent).toBe(
        'reviewer-dir-9',
      ),
    )
    // query-2.txt exists in the new group too, so a filename-only rule would
    // silently keep it. The identity key is what makes this a fresh choice.
    expect(witnessName()).toBe('query-4.txt')
    expect(screen.getByText('0.720')).toBeTruthy()
  })

  it('never paints one directory’s witness under another directory’s number', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')

    // The replacement group contains a member with the SAME filename, so a
    // filename-only selection rule keeps it silently.
    //
    // Honest note on what this pins today: `usePredictions` blanks its data
    // for the duration of a refetch, so the intermediate commit is already
    // witness-less and this assertion passes even with the reset moved into
    // an effect. The render-time reset is pinned directly by
    // ./useSelectedMember.test.tsx. This case guards the pairing end to end
    // and stays honest if #156's shared state keeps previous data on screen
    // during a refresh, which is exactly when the flash becomes reachable.
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      dirName: 'reviewer-dir-9',
      label: 'Re-scored group',
      members: [
        { filename: 'query-2.txt', text: 'text of query-2.txt' },
        { filename: 'query-4.txt', text: 'text of query-4.txt' },
      ],
      supporting: { query_id: 4, filename: 'query-4.txt', score: 0.72 },
      score: 0.72,
    }
    await userEvent.click(screen.getByText('Refresh from server'))

    await waitFor(() =>
      expect(screen.getByTestId('candidate-dir').textContent).toBe(
        'reviewer-dir-9',
      ),
    )
    expect(witnessName()).toBe('query-4.txt')
    expect(painted).toContain('reviewer-dir-1:query-2.txt')
    expect(painted).not.toContain('reviewer-dir-9:query-2.txt')
  })

  it('re-defaults to the supporting witness on a new request generation', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    await chooseWitness('query-2.txt')
    const before = requestCount

    await userEvent.click(screen.getByText('Refresh predictions'))

    await waitFor(() => expect(requestCount).toBeGreaterThan(before))
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(screen.getByText('0.800')).toBeTruthy()
  })

  it('shows no member evidence for a model that serves no reviewer groups', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))

    // KaLM-mini has no q-q matrix, so rank 11 has no candidate behind it.
    await userEvent.selectOptions(
      screen.getByLabelText('Select model'),
      MODEL_NO_MATRIX,
    )

    await waitFor(() => expect(witnessName()).toBe(''))
    expect(screen.queryByTestId('member-evidence')).toBeNull()
    expect(screen.getByTestId('candidate-dir').textContent).toBe('')
  })

  it('uses plain text, and no selector, for a single-member group', async () => {
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      dirName: 'reviewer-dir-1',
      label: 'Unattested homily',
      members: [{ filename: 'query-2.txt', text: 'text of query-2.txt' }],
      supporting: { query_id: 2, filename: 'query-2.txt', score: SEED_ONLY },
      score: SEED_ONLY,
    }
    renderHarness()

    await waitFor(() => expect(witnessName()).toBe('query-2.txt'))
    expect(screen.queryByTestId('member-selector')).toBeNull()
    expect(screen.queryByTestId('member-evidence-single')).toBeNull()
    expect(screen.getByText('0.200')).toBeTruthy()
    expect(attributionText()).toBe(
      'Similarity to the witness shown below, the only member this response lists.',
    )
  })

  it('refuses to nominate a witness when the response names none', async () => {
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      ...defaultGroup(),
      supporting: null,
    }
    renderHarness()

    // Something has to be rendered, but nothing may call it the source.
    await waitFor(() => expect(witnessName()).toBe('query-2.txt'))
    expect(attributionText()).toContain(
      'This response does not identify which witness produced it.',
    )
    expect(optionLabels()).toEqual(['query-2.txt', 'query-0.txt'])
    expect(screen.queryByTestId('show-supporting-witness')).toBeNull()
  })

  it('explains a supporting witness whose text is missing', async () => {
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      ...defaultGroup(),
      members: [
        { filename: 'query-2.txt', text: 'text of query-2.txt' },
        { filename: 'query-0.txt', text: '' },
      ],
    }
    renderHarness()

    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(witnessText()).toBe('')
    expect(attributionText()).toContain('its text is not available')
    expect(optionLabels()[1]).toBe(
      'query-0.txt (produced the score, no text available)',
    )
    // Still the winner: the readable seed does not inherit the number.
    expect(screen.getByText('0.800')).toBeTruthy()
  })

  it('explains a member the response cannot open at all', async () => {
    groups[`${MODEL_A}:${QUERY_ID}`] = {
      ...defaultGroup(),
      closedMembers: ['query-13.txt'],
    }
    renderHarness()

    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(screen.getByTestId('member-evidence-unopenable').textContent).toBe(
      '1 of 3 member witnesses cannot be opened here.',
    )
    expect(optionLabels()).toHaveLength(2)
    // The count next to the selector has to agree with that sentence.
    expect(screen.getByText('2 of 3 members')).toBeTruthy()
  })

  it('counts members plainly when every one of them opens', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    // Not "2 of 2 members": there is nothing to qualify.
    expect(screen.getByText('2 members')).toBeTruthy()
    expect(screen.queryByTestId('member-evidence-unopenable')).toBeNull()
  })

  it('never presents a reviewer witness as having model attribution', async () => {
    renderHarness()
    await waitFor(() => expect(witnessName()).toBe('query-0.txt'))
    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('No model attribution exists for a reviewer-group witness')
  })
})

// ---------------------------------------------------------------------------
// Cases that are about the artifact rather than the fetch
// ---------------------------------------------------------------------------

function labelledCandidate(files: { filename: string; text: string }[]) {
  return {
    dir_name: 'CANT.328.12',
    score: 0.91,
    dir_files: files.map((f) => f.filename),
    candidate_files: files,
    source: 'model' as const,
    supporting_member: null,
  }
}

function renderBar(props: {
  candidate: MemberEvidenceCandidate
  selected?: string | null
  candidatePath?: string | null
  lexicalHighlighting?: boolean
}) {
  const evidence = resolveMemberEvidence(props.candidate, props.selected ?? null)!
  const attribution = attributionAppliesToWitness({
    candidatePath: props.candidatePath,
    dirName: props.candidate.dir_name,
    filename: evidence.selected?.filename ?? null,
    source: props.candidate.source,
  })
  return render(
    <MemberEvidenceBar
      evidence={evidence}
      onSelectWitness={() => {}}
      attribution={attribution}
      lexicalHighlighting={props.lexicalHighlighting}
    />,
  )
}

describe('member evidence and model attribution', () => {
  it('stays out of the way for an ordinary single-file labelled candidate', () => {
    renderBar({
      candidate: labelledCandidate([{ filename: 'a.txt', text: 'labelled' }]),
      candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
    })
    expect(screen.queryByTestId('member-evidence')).toBeNull()
  })

  it('does not call a labelled directory score an individual file score', () => {
    renderBar({
      candidate: labelledCandidate([
        { filename: 'a.txt', text: 'first' },
        { filename: 'b.txt', text: 'second' },
      ]),
      candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
    })
    expect(screen.getByText('Retrieval score')).toBeTruthy()
    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      'Score for the whole directory from the retrieval run, not for one file in it.',
    )
    expect(screen.queryByTestId('member-evidence-model-attribution')).toBeNull()
  })

  it('withholds attribution built from a different file in the directory', () => {
    renderBar({
      candidate: labelledCandidate([
        { filename: 'a.txt', text: 'first' },
        { filename: 'b.txt', text: 'second' },
      ]),
      selected: 'b.txt',
      candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
    })
    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('describes a different witness')
  })

  it('labels word overlap as lexical rather than evidence for the score', () => {
    renderBar({
      candidate: labelledCandidate([
        { filename: 'a.txt', text: 'first' },
        { filename: 'b.txt', text: 'second' },
      ]),
      selected: 'b.txt',
      candidatePath: 'data/canon_labelled/CANT.328.12/a.txt',
      lexicalHighlighting: true,
    })
    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('word overlap, not evidence for the similarity score')
  })

  it('leaves the panel to explain a pair with no artifact at all', () => {
    renderBar({
      candidate: labelledCandidate([
        { filename: 'a.txt', text: 'first' },
        { filename: 'b.txt', text: 'second' },
      ]),
      candidatePath: null,
    })
    expect(screen.queryByTestId('member-evidence-model-attribution')).toBeNull()
  })

  it('gives the selector an accessible name and marks the current witness', () => {
    renderBar({
      candidate: {
        dir_name: 'reviewer-dir-1',
        score: GROUP_MAX,
        dir_files: ['query-2.txt', 'query-0.txt'],
        candidate_files: [
          { filename: 'query-2.txt', text: 'seed' },
          { filename: 'query-0.txt', text: 'support' },
        ],
        source: 'reviewer',
        supporting_member: {
          query_id: 0,
          filename: 'query-0.txt',
          score: GROUP_MAX,
        },
      },
    })
    const select = screen.getByLabelText('Viewing witness') as HTMLSelectElement
    expect(select.value).toBe('query-0.txt')
    expect(screen.getByRole('region', { name: 'Member evidence' })).toBeTruthy()
  })

  it('opens the second of two identically named witnesses, and claims nothing about which won', async () => {
    const candidate: MemberEvidenceCandidate = {
      dir_name: 'reviewer-dir-1',
      score: GROUP_MAX,
      dir_files: ['same.txt', 'same.txt'],
      candidate_files: [
        { filename: 'same.txt', text: 'first text' },
        { filename: 'same.txt', text: 'second text' },
      ],
      source: 'reviewer',
      supporting_member: {
        query_id: 4,
        filename: 'same.txt',
        score: GROUP_MAX,
      },
    }

    function DuplicateHarness() {
      const [choice, setChoice] = useState<string | null>(null)
      const evidence = resolveMemberEvidence(candidate, choice)!
      return (
        <>
          <MemberEvidenceBar evidence={evidence} onSelectWitness={setChoice} />
          <p data-testid="open-text">{evidence.selected?.text}</p>
        </>
      )
    }

    render(<DuplicateHarness />)
    expect(
      screen.getByTestId('member-evidence-attribution').textContent,
    ).toContain('more than one witness under that name')
    // Nothing to return to: no entry is provably the source.
    expect(screen.queryByTestId('show-supporting-witness')).toBeNull()

    await userEvent.selectOptions(
      screen.getByLabelText('Viewing witness'),
      screen.getByRole('option', { name: 'same.txt (entry 2 under this name)' }),
    )
    expect(screen.getByTestId('open-text').textContent).toBe('second text')
  })
})
