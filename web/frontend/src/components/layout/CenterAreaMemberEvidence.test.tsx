import { useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import {
  PredictionProvider,
  usePredictionState,
} from '../../contexts/PredictionContext'
import { TokenProvider } from '../../contexts/TokenContext'
import ModelSelector from '../predictions/ModelSelector'
import CenterArea from './CenterArea'

/**
 * The member selector, mounted (issue #163).
 *
 * The view model and its copy are unit-tested next door; these drive the real
 * component tree -- AppProvider, the shared PredictionProvider, the real
 * response validator, TokenProvider, DocumentPanel -- because every claim #163
 * makes is about what a reviewer sees in the candidate panel, and the risky
 * half of the change is which state follows the selection and which does not.
 *
 * What is deliberately NOT here: assessment eligibility and the feedback copy
 * that quotes it. Those are one lane over, exercised against the real panel in
 * `components/feedback/AssessmentEvidence.test.tsx`, where the pills and
 * Submit are actually mounted. Two tests below pin the boundary rather than
 * crossing it: viewing a member moves no rank and posts nothing.
 */

/** The deployment's preferred model, which ModelSelector opens on. */
const MODEL_A = 'google_mt5-base'
const MODEL_B = 'bowphs_LaTa'
const QUERY_ID = 7

const MODEL_DIR_A = 'CANT.328.12'
const MODEL_DIR_B = 'CSAR.347.17'
const REVIEWER_DIR = 'reviewer-dir-3'
const REVIEWER_RANK = 11

/**
 * Query words that appear in no candidate, so the only thing that can shade a
 * token is the attribution artifact. Word overlap would shade the same spans
 * for a different reason and make "the highlights moved with the witness"
 * unfalsifiable.
 */
const QUERY_TOKENS = ['ALPHA', 'BETA']

type Witness = { filename: string; text: string }

/** Files per directory, in the API's member order. */
let files: Record<string, Witness[]> = {}
/** Support metadata for the reviewer candidate; `undefined` omits the field. */
let reviewerSupport: unknown
/** Directories the artifact index knows about. */
let artifactDirs = new Set<string>()
/** Member the artifact was built from, per directory. */
let artifactWitness: Record<string, string> = {}
let requested: string[] = []

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function modelEntry(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'mean',
    prediction_count: 2238,
    available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.5, verify: 0.7 },
    supports_reviewer_dirs: true,
  }
}

function modelDirFor(model: string): string {
  return model === MODEL_A ? MODEL_DIR_A : MODEL_DIR_B
}

function predictionFor(model: string) {
  const modelDir = modelDirFor(model)
  const reviewer: Record<string, unknown> = {
    rank: REVIEWER_RANK,
    dir_name: REVIEWER_DIR,
    label: 'Ninth-century glosses',
    score: 0.8,
    dir_files: files[REVIEWER_DIR].map((f) => f.filename),
    preview_text: 'preview',
    candidate_files: files[REVIEWER_DIR],
    source: 'reviewer',
  }
  if (reviewerSupport !== undefined) {
    reviewer.supporting_member = reviewerSupport
  }
  return [
    {
      rank: 1,
      dir_name: modelDir,
      score: 0.9,
      dir_files: files[modelDir].map((f) => f.filename),
      preview_text: 'preview',
      candidate_files: files[modelDir],
      source: 'model',
    },
    reviewer,
  ]
}

function installFetch(): void {
  requested = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      requested.push(url)
      const parsed = new URL(url, 'http://localhost')
      const params = parsed.searchParams

      if (url.includes('/api/models')) {
        // Sorted by display name, as the real endpoint is -- so the first
        // entry is not the one the app opens on.
        return jsonResponse([
          modelEntry(MODEL_B, 'LaTa (T5)'),
          modelEntry(MODEL_A, 'mT5-base'),
        ])
      }

      if (url.includes('/token_map')) {
        const dir = params.get('candidate_dir') ?? ''
        if (!artifactDirs.has(dir)) {
          return jsonResponse({ error: { message: 'no artifact' } }, 404)
        }
        return jsonResponse({
          example_id: `ex-${dir}`,
          model: params.get('model') ?? '',
          layer: 4,
          D: 10,
          bucket: 'demo',
          query_path: `data/canon_unlabelled/query-${QUERY_ID}.txt`,
          // The wire shape the backend builds from the IG examples CSV. It
          // names ONE witness, which is the whole reason the panel has to ask
          // whether the artifact is about the member on screen.
          candidate_path: `data/canon_labelled/${dir}/${artifactWitness[dir]}`,
          query_tokens: [],
          candidate_tokens: [],
          similarity_matrix: [
            [0, 0],
            [0, 0],
          ],
          ig_weighted_matrix: null,
          top_matches: {},
          query_ig_baseline: [],
          query_ig_abtt: [],
          candidate_ig_baseline: [],
          candidate_ig_abtt: [],
          auto_highlights: null,
          available_variants: ['sif_abtt'],
          available_methods: ['ig'],
          // Lights query token 0 and candidate token 0.
          pair_matrices: {
            ig: {
              sif_abtt: [
                [1, 0],
                [0, 0],
              ],
            },
          },
        })
      }

      if (url.includes('/predictions')) {
        const model = params.get('model') ?? ''
        return jsonResponse({
          file_id: QUERY_ID,
          filename: `query-${QUERY_ID}.txt`,
          model,
          variant: params.get('variant') ?? '',
          status: 'ok',
          predictions: predictionFor(model),
          seeded_dirs: [],
        })
      }

      if (/\/api\/query\/\d+$/.test(parsed.pathname)) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: `query-${QUERY_ID}.txt`,
          text: QUERY_TOKENS.join(' '),
          tokens: QUERY_TOKENS.map((text, index) => ({
            text,
            index,
            category: 'content',
          })),
          char_count: 20,
          token_count: QUERY_TOKENS.length,
        })
      }

      return jsonResponse({})
    }),
  )
}

/**
 * The ranking the shared provider is exposing, read through the real hook.
 *
 * #163 must not touch the science: the same ranks, the same scores and the
 * same member order have to survive every selection the reviewer makes.
 */
function RankingProbe() {
  const predictions = usePredictionState()
  return (
    <div
      data-testid="ranking"
      data-generation={predictions.generation}
      data-phase={predictions.phase}
    >
      {(predictions.predictions ?? [])
        .map(
          (p) =>
            `${p.rank}/${p.dir_name}/${p.score}/${(p.candidate_files ?? [])
              .map((f) => f.filename)
              .join('+')}`,
        )
        .join(' ')}
    </div>
  )
}

function Harness() {
  const { setActiveQueryId, setActivePredictionRank, activePredictionRank } =
    useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
  }, [setActiveQueryId])
  return (
    <>
      <ModelSelector />
      <button
        type="button"
        data-testid="open-reviewer"
        onClick={() => setActivePredictionRank(REVIEWER_RANK)}
      >
        open reviewer candidate
      </button>
      <button
        type="button"
        data-testid="open-model"
        onClick={() => setActivePredictionRank(1)}
      >
        open model candidate
      </button>
      <span data-testid="active-rank">{activePredictionRank}</span>
      <RankingProbe />
      <CenterArea />
    </>
  )
}

function renderApp() {
  return render(
    <AppProvider>
      <TokenProvider>
        <PredictionProvider>
          <Harness />
        </PredictionProvider>
      </TokenProvider>
    </AppProvider>,
  )
}

async function selectModel(slug: string) {
  await userEvent.selectOptions(screen.getByLabelText('Select model'), slug)
}

/** The filename in the candidate panel's header. */
function shownWitness(): string | null {
  const header = screen.getByTestId('document-provenance-candidate').parentElement
  return header?.querySelector('span')?.textContent ?? null
}

function memberOptions(): string[] {
  return Array.from(
    screen.getByTestId('member-selector').querySelectorAll('option'),
  ).map((o) => o.textContent ?? '')
}

async function chooseWitness(key: string) {
  await userEvent.selectOptions(screen.getByTestId('member-selector'), key)
}

/** Query tokens currently carrying attribution shading. */
function shadedQueryWords(): string[] {
  return Array.from(document.querySelectorAll('[data-side="query"]'))
    .filter((el) => (el as HTMLElement).style.backgroundColor !== '')
    .map((el) => el.textContent ?? '')
}

const settle = () => act(() => new Promise((r) => setTimeout(r, 60)))

beforeEach(() => {
  files = {
    [MODEL_DIR_A]: [
      { filename: 'c1.txt', text: 'CANDIDATUM1 sequitur' },
      { filename: 'c2.txt', text: 'CANDIDATUM2 sequitur' },
    ],
    [MODEL_DIR_B]: [
      { filename: 'd1.txt', text: 'ALTERUM1 sequitur' },
      { filename: 'd2.txt', text: 'ALTERUM2 sequitur' },
    ],
    [REVIEWER_DIR]: [
      { filename: 'q-a.txt', text: 'MEMBRUMA sequitur' },
      { filename: 'q-b.txt', text: 'MEMBRUMB sequitur' },
    ],
  }
  reviewerSupport = { query_id: 41, filename: 'q-b.txt', score: 0.8 }
  artifactDirs = new Set([MODEL_DIR_A, MODEL_DIR_B])
  artifactWitness = { [MODEL_DIR_A]: 'c1.txt', [MODEL_DIR_B]: 'd1.txt' }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('which member of the group is on screen (issue #163)', () => {
  it('opens a reviewer group on the witness the response designates', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')

    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    // Not the first member. The default is the designated support, which is
    // the only member the response ties to the number at all.
    expect(shownWitness()).toBe('q-b.txt')
    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      "Highest similarity across this group's scorable member witnesses." +
        ' The witness shown below produced it.',
    )
  })

  it('keeps the group maximum explicit, and identical, under every choice', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    const bar = screen.getByTestId('member-evidence')
    expect(bar.textContent).toContain('Group maximum')
    expect(bar.textContent).toContain('0.800')

    await chooseWitness('q-a.txt')
    await screen.findByText('MEMBRUMA')

    // The number is the group's, so it does not follow the selection -- and
    // the label still says whose it is.
    expect(bar.textContent).toContain('Group maximum')
    expect(bar.textContent).toContain('0.800')
    // Neutral about the viewed member: the backend breaks equal maxima by the
    // smallest member id, so "did not produce it" would be a claim the wire
    // cannot support.
    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      "Highest similarity across this group's scorable member witnesses." +
        ' This response designates q-b.txt as the supporting witness.' +
        ' The individual similarity of the witness shown below is not supplied.',
    )
  })

  it('prints the number once, under the label that says what it covers', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    // The document header's generic "Similarity: 0.800" would be both a second
    // copy of the figure and a claim about the wrong thing: it reads as the
    // similarity of the witness below, which is exactly what is not supplied.
    expect(screen.queryByText(/^Similarity: /)).toBeNull()
    expect(screen.getAllByText('0.800')).toHaveLength(1)
  })

  it('offers the members in the API order, marking the one that produced it', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    expect(memberOptions()).toEqual([
      'q-a.txt',
      'q-b.txt (produced the score)',
    ])
    expect(screen.getByTestId('member-evidence').textContent).toContain(
      '2 members',
    )
  })

  it('returns to the supporting witness in one action', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    await chooseWitness('q-a.txt')
    await screen.findByText('MEMBRUMA')

    await userEvent.click(screen.getByTestId('show-supporting-witness'))
    await screen.findByText('MEMBRUMB')
    expect(shownWitness()).toBe('q-b.txt')
    expect(screen.queryByTestId('show-supporting-witness')).toBeNull()
  })

  it('mounts the selector for a model directory with several witnesses too', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')

    // A labelled directory's number is not attributed to any file in it, and
    // the strip says so rather than leaving the reviewer to assume the panel
    // is showing the file that earned it.
    expect(memberOptions()).toEqual(['c1.txt', 'c2.txt'])
    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      'Score for the whole directory from the retrieval run, not for one file in it.',
    )

    await chooseWitness('c2.txt')
    await screen.findByText('CANDIDATUM2')
    expect(shownWitness()).toBe('c2.txt')
  })

  it('says so when the designated member cannot be opened', async () => {
    reviewerSupport = { query_id: 41, filename: 'q-z.txt', score: 0.8 }
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMA')

    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      "Highest similarity across this group's scorable member witnesses." +
        ' Produced by q-z.txt, which cannot be opened here.',
    )
    expect(screen.getByTestId('member-evidence-unopenable').textContent).toBe(
      '1 of 3 member witnesses cannot be opened here.',
    )
    // No member is dressed as the source, and the selector still opens on the
    // response's first member rather than nominating one.
    expect(shownWitness()).toBe('q-a.txt')
    expect(screen.queryByTestId('show-supporting-witness')).toBeNull()
  })

  it('reports an unnamed winner as unnamed rather than picking one', async () => {
    reviewerSupport = { query_id: 41, score: 0.8 }
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMA')

    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      "Highest similarity across this group's scorable member witnesses." +
        ' Produced by member query 41, which this response does not name.',
    )
  })

  it('does not invent support when the response carries none', async () => {
    reviewerSupport = undefined
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMA')

    expect(screen.getByTestId('member-evidence-attribution').textContent).toBe(
      "Highest similarity across this group's scorable member witnesses." +
        ' This response does not identify which witness produced it.',
    )
  })
})

describe('the evidence on screen follows the witness on screen (issue #163)', () => {
  it('stops painting an artifact once another member is displayed', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await waitFor(() => expect(shadedQueryWords()).toEqual(['ALPHA']))

    await chooseWitness('c2.txt')

    // Same commit. The artifact was built from c1.txt; over c2.txt its cells
    // would name words that witness does not contain.
    expect(shadedQueryWords()).toEqual([])
    await settle()
    expect(shadedQueryWords()).toEqual([])
    expect(screen.getByTestId('member-evidence-model-attribution').textContent).toBe(
      'The available attribution describes a different witness, so it is not applied here.' +
        ' Any highlighting below is word overlap, not evidence for the similarity score.',
    )
  })

  it('paints it again when the artifact’s own witness is chosen back', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await chooseWitness('c2.txt')
    await screen.findByText('CANDIDATUM2')

    await chooseWitness('c1.txt')
    await screen.findByText('CANDIDATUM1')
    await waitFor(() => expect(shadedQueryWords()).toEqual(['ALPHA']))
    expect(screen.queryByTestId('member-evidence-model-attribution')).toBeNull()
  })

  it('never offers model attribution for a reviewer-group witness', async () => {
    // The reviewer directory has no artifact at all -- no run pairs a query
    // with another query -- so the strip has to say what the marks below are
    // rather than leaving word overlap to read as model evidence.
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')
    await settle()

    expect(shadedQueryWords()).toEqual([])
    expect(screen.getByTestId('member-evidence-model-attribution').textContent).toBe(
      'No model attribution exists for a reviewer-group witness, so none is shown.' +
        ' Any highlighting below is word overlap, not evidence for the similarity score.',
    )
  })

  it('drops a hovered pairing in the commit that changes the witness', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await waitFor(() => expect(shadedQueryWords()).toEqual(['ALPHA']))

    const alpha = screen.getByLabelText('query token ALPHA, highlighted evidence')
    fireEvent.mouseEnter(alpha)
    await waitFor(() =>
      expect(
        screen
          .getByLabelText('query token ALPHA, highlighted evidence')
          .className.includes('ring-accent/60'),
      ).toBe(true),
    )

    await chooseWitness('c2.txt')

    // A hover match is "this query token matches candidate token 3", and
    // candidate token 3 is a different word now.
    expect(
      screen.getByLabelText('query token ALPHA').className.includes('ring-accent/60'),
    ).toBe(false)
    await settle()
    expect(
      screen.getByLabelText('query token ALPHA').className.includes('ring-accent/60'),
    ).toBe(false)
  })
})

describe('a member choice cannot outlive what it was made against (issue #163)', () => {
  it('is discarded when the model changes', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await chooseWitness('c2.txt')
    await screen.findByText('CANDIDATUM2')

    await selectModel(MODEL_B)
    await screen.findByText('ALTERUM1')
    expect(shownWitness()).toBe('d1.txt')
  })

  it('is discarded when the candidate directory changes', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')
    await chooseWitness('q-a.txt')
    await screen.findByText('MEMBRUMA')

    fireEvent.click(screen.getByTestId('open-model'))
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    // Back on the designated support, not on the choice made for a directory
    // the reviewer has since left.
    expect(shownWitness()).toBe('q-b.txt')
  })

  it('is discarded when the ranking is refetched under the same key', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await chooseWitness('c2.txt')
    await screen.findByText('CANDIDATUM2')

    const before = Number(
      screen.getByTestId('ranking').getAttribute('data-generation'),
    )
    // Same key, new generation: what the model selector does on a re-pick.
    await selectModel(MODEL_B)
    await screen.findByText('ALTERUM1')
    await selectModel(MODEL_A)
    await screen.findByText('CANDIDATUM1')

    expect(
      Number(screen.getByTestId('ranking').getAttribute('data-generation')),
    ).toBeGreaterThan(before)
    expect(shownWitness()).toBe('c1.txt')
  })
})

describe('what the selector must not disturb (issue #163)', () => {
  it('changes no rank, no score and no member order', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    const before = screen.getByTestId('ranking').textContent
    const generation = screen
      .getByTestId('ranking')
      .getAttribute('data-generation')

    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')
    await chooseWitness('q-a.txt')
    await screen.findByText('MEMBRUMA')
    await settle()

    expect(screen.getByTestId('ranking').textContent).toBe(before)
    expect(screen.getByTestId('ranking').getAttribute('data-generation')).toBe(
      generation,
    )
    // Still rank 11: viewing another member is not moving to another candidate,
    // which is what the assessment panel and the prediction list read.
    expect(screen.getByTestId('active-rank').textContent).toBe(
      String(REVIEWER_RANK),
    )
  })

  it('names the displayed witness when it is the blank one', async () => {
    files[MODEL_DIR_A] = [
      { filename: 'c1.txt', text: 'CANDIDATUM1 sequitur' },
      { filename: 'c2.txt', text: '' },
    ]
    renderApp()
    await screen.findByText('CANDIDATUM1')
    expect(screen.queryByTestId('candidate-evidence-note')).toBeNull()

    await chooseWitness('c2.txt')
    await screen.findByTestId('candidate-evidence-note')

    // The blank pane is about c2.txt, and the reassurance is about the rest of
    // the directory -- which now means every position except the displayed
    // one, not every position except the first.
    expect(screen.getByTestId('candidate-evidence-note').textContent).toBe(
      'The file shown here (c2.txt) has no readable text in this deployment,' +
        ' so there is nothing to compare word by word.' +
        ' Other files in this directory do carry text.',
    )
  })

  it('refetches nothing: a member is already in the response', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    await settle()
    const before = requested.length

    await chooseWitness('c2.txt')
    await screen.findByText('CANDIDATUM2')
    await settle()

    // The token map is keyed by DIRECTORY, so asking again for another member
    // of the same directory would return the same artifact and claim it fresh.
    expect(requested.slice(before)).toEqual([])
  })
})

describe('support metadata reaches the panel through the validator (issue #163)', () => {
  it('carries a real response’s supporting member into the mounted bar', async () => {
    renderApp()
    await screen.findByText('CANDIDATUM1')
    fireEvent.click(screen.getByTestId('open-reviewer'))
    await screen.findByText('MEMBRUMB')

    // Proof the metadata survived normalisation rather than being rebuilt in
    // the view: the option marker comes from `supporting_member.filename`.
    expect(memberOptions()[1]).toBe('q-b.txt (produced the score)')
  })

  it('fails the whole ranking when the support object is unreadable', async () => {
    // A stringified id is the shape a half-upgraded backend would send.
    reviewerSupport = { query_id: '41', filename: 'q-b.txt', score: 0.8 }
    renderApp()

    await waitFor(() =>
      expect(screen.getByTestId('ranking').getAttribute('data-phase')).toBe(
        'error',
      ),
    )
    // Not a silently unattributed group: that wording is the app's honest
    // answer for "the server named nobody", and a build mismatch is not that.
    expect(screen.queryByTestId('member-evidence')).toBeNull()
    expect(screen.getByTestId('candidate-evidence-note').textContent).toContain(
      'The ranking for this document did not load',
    )
  })
})
