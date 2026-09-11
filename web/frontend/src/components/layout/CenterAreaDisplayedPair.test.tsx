import { Profiler, useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import { TokenProvider, useTokens } from '../../contexts/TokenContext'
import AttributionMethodSelector from '../common/AttributionMethodSelector'
import CenterArea from './CenterArea'

/**
 * One displayed pair, agreed on by everything that reads it (issue #163).
 *
 * `CenterAreaWitnessScope` asks what the view believes about a member change
 * that happens in place. This suite asks the harder question: what the view
 * believes while a candidate is still ON SCREEN but no longer current, and
 * what a control outside the panel believes in the frame the words change.
 *
 * The crossfade between two ranked candidates is deliberate and stays. What it
 * creates is a window in which the panel a reviewer can see, touch and take a
 * measurement from is not the panel the rest of the app is reasoning about.
 * Three consumers resolved evidence through it:
 *
 *   1. connection geometry, read from whichever element happened to be
 *      registered as `candidate:N` -- the outgoing manuscript's word -- and
 *      then cached, so the coordinates outlived the panel they were measured
 *      from;
 *   2. the member selector, whose stable callback labelled every event with
 *      the CURRENT group, so a filename the two groups share moved the
 *      incoming panel to a witness nobody chose;
 *   3. attribution applicability, published in a passive effect, so the first
 *      committed frame of a new witness carried the previous witness's method
 *      controls.
 *
 * Fixture choices that carry the weight, all of them the reviewer's:
 *
 *   - the query word ALPHA appears in EVERY witness of BOTH directories, so a
 *     stale link is a real claim and not an obviously empty one;
 *   - `getBoundingClientRect` returns real, DIFFERENT geometry per directory,
 *     so "the old coordinates survived" is observable. jsdom's zeros make
 *     every connection invisible and hide the whole class;
 *   - both directories contain a.txt and b.txt in the shared-siglum case,
 *     which is what makes a mislabelled selection pass a membership check;
 *   - `AttributionMethodSelector` is mounted as the sidebar sibling it really
 *     is, and every commit is recorded through a real `Profiler`, because the
 *     final DOM is correct and only an intermediate frame is not.
 */

const MODEL = 'google_mt5-base'
const FIRST_DIR = 'CANT.328.12'
const SECOND_DIR = 'CSAR.347.17'
const QUERY_ID = 7

/** Serve an artifact at all, and if so for which witness of the first group. */
let artifact: boolean
let artifactWitness: string
/** Give the second directory the first's filenames (sigla repeat across groups). */
let sharedFilenames: boolean
/** Move the second directory's words, so retained coordinates are visible. */
let secondDirMoved: boolean
/**
 * Designate b.txt as the first group's support.
 *
 * b.txt rather than a.txt on purpose: it is the SECOND witness of both groups,
 * so a shortcut press that leaked across the boundary lands on "quartus" and
 * not on the incoming group's own default. Designating the first file would
 * make the leak and the correct answer identical and the test vacuous.
 */
let designateSupport: boolean
let tokens: ReturnType<typeof useTokens>

interface Frame {
  witness: string
  candidateWords: string[]
  methodControls: boolean
  paths: string[]
  /** Candidate tokens the panel is painting as evidence for the query. */
  highlighted: string[]
  matches: number[]
}

let frames: Frame[]

function secondFiles(): { filename: string; text: string }[] {
  return [
    { filename: sharedFilenames ? 'a.txt' : 'c.txt', text: 'tertius ALPHA' },
    { filename: sharedFilenames ? 'b.txt' : 'd.txt', text: 'quartus ALPHA' },
  ]
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function candidateTokenEls(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('[data-side="candidate"]'))
}

function observe(): void {
  frames.push({
    witness:
      document.querySelector<HTMLSelectElement>('[data-testid="member-selector"]')
        ?.value ?? '',
    candidateWords: candidateTokenEls().map((el) => el.textContent ?? ''),
    methodControls: !!document.querySelector('[aria-label="Attribution method"]'),
    paths: Array.from(document.querySelectorAll('svg path[d]')).map(
      (el) => el.getAttribute('d') ?? '',
    ),
    highlighted: candidateTokenEls()
      .filter((el) => (el.getAttribute('aria-label') ?? '').includes('highlighted'))
      .map((el) => el.textContent ?? ''),
    matches: tokens?.hoveredMatches.map((match) => match.candidateIdx) ?? [],
  })
}

function Harness() {
  const { setActiveQueryId, setActiveModel, setActivePredictionRank } = useApp()
  tokens = useTokens()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return (
    <button type="button" onClick={() => setActivePredictionRank(2)}>
      Open second directory
    </button>
  )
}

function renderApp() {
  return render(
    <AppProvider>
      <TokenProvider>
        {/* The query header's directory badge reads the durable
            saved-directory record (issue #161), which has no fallback
            store: mounting the provider is how App composes them. */}
        <SavedDirectoryProvider accountKey="displayed-pair">
          <PredictionProvider>
            <Profiler id="center" onRender={observe}>
              <Harness />
              {/* A sidebar sibling, outside every scope the panel can apply. */}
              <AttributionMethodSelector />
              <CenterArea />
            </Profiler>
          </PredictionProvider>
        </SavedDirectoryProvider>
      </TokenProvider>
    </AppProvider>,
  )
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')

      if (url.pathname.endsWith('/predictions')) {
        const first = [
          { filename: 'a.txt', text: 'ALPHA primus' },
          { filename: 'b.txt', text: 'secundus ALPHA' },
        ]
        return jsonResponse({
          file_id: QUERY_ID,
          filename: `query-${QUERY_ID}.txt`,
          model: url.searchParams.get('model') ?? '',
          variant: 'sif_abtt',
          status: 'ok',
          seeded_dirs: [],
          predictions: [
            {
              rank: 1,
              dir_name: FIRST_DIR,
              score: 0.9,
              source: 'model',
              dir_files: first.map((f) => f.filename),
              candidate_files: first,
              preview_text: first[0].text,
              supporting_member: designateSupport
                ? { query_id: 11, filename: 'b.txt', score: 0.9 }
                : null,
            },
            {
              rank: 2,
              dir_name: SECOND_DIR,
              score: 0.7,
              source: 'model',
              dir_files: secondFiles().map((f) => f.filename),
              candidate_files: secondFiles(),
              preview_text: secondFiles()[0].text,
              supporting_member: null,
            },
          ],
        })
      }

      if (url.pathname.endsWith('/token_map')) {
        if (!artifact) {
          return jsonResponse({ error: { message: 'no artifact' } }, 404)
        }
        const dir = url.searchParams.get('candidate_dir')
        const method = url.searchParams.get('method') ?? 'ig'
        return jsonResponse({
          example_id: dir === FIRST_DIR ? 123 : 124,
          model: MODEL,
          layer: 4,
          D: 10,
          bucket: 'unlabelled_bulk',
          query_path: `data/canon_unlabelled/query-${QUERY_ID}.txt`,
          candidate_path: `data/canon_labelled/${dir}/${
            dir === FIRST_DIR ? artifactWitness : secondFiles()[0].filename
          }`,
          query_tokens: [{ idx: 0, text: 'ALPHA', is_content: true }],
          candidate_tokens: [],
          similarity_matrix: [[1, 0]],
          ig_weighted_matrix: null,
          top_matches: {},
          query_ig_baseline: [],
          query_ig_abtt: [],
          candidate_ig_baseline: [],
          candidate_ig_abtt: [],
          auto_highlights: null,
          available_methods: ['ig'],
          available_variants: ['sif_abtt'],
          pair_matrices: { [method]: { sif_abtt: [[1, 0]] } },
        })
      }

      if (url.pathname === `/api/query/${QUERY_ID}`) {
        return jsonResponse({
          file_id: QUERY_ID,
          filename: `query-${QUERY_ID}.txt`,
          text: 'ALPHA',
          tokens: [{ text: 'ALPHA', index: 0, category: 'content' }],
          char_count: 5,
          token_count: 1,
        })
      }

      return jsonResponse({})
    }),
  )
}

beforeEach(() => {
  artifact = false
  artifactWitness = 'a.txt'
  sharedFilenames = false
  secondDirMoved = false
  designateSupport = false
  frames = []
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
    function (this: HTMLElement) {
      const side = this.getAttribute('data-side')
      const index = Number(this.getAttribute('data-token-idx') ?? 0)
      if (side === 'query') return new DOMRect(10 + index * 60, 20, 50, 20)
      if (side === 'candidate') {
        // The second directory's panel sits somewhere else, so a path drawn
        // for the first one is recognisable after the second has entered.
        const isSecond = this.closest('[data-tour="candidate-panel"]')
          ?.textContent?.includes('tertius')
        return new DOMRect(
          (secondDirMoved && isSecond ? 400 : 250) + index * 60,
          20,
          50,
          20,
        )
      }
      return new DOMRect(0, 0, 600, 300)
    },
  )
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function queryToken(): HTMLElement {
  return screen.getByLabelText('query token ALPHA, highlighted evidence')
}

function renderedPaths(): string[] {
  return Array.from(document.querySelectorAll('svg path[d]')).map(
    (el) => el.getAttribute('d') ?? '',
  )
}

/** Long enough for framer-motion's 0.2s candidate crossfade to finish. */
async function settleCrossfade(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 300))
  })
}

describe('evidence resolves through the pair that rendered it', () => {
  it('does not join a fresh hover to the words of the candidate leaving', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()
    frames = []

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    // The outgoing panel is still on screen: that is the crossfade working.
    expect(screen.getByText('primus')).toBeTruthy()

    fireEvent.mouseEnter(queryToken())

    // The hover itself is correct -- it reports the incoming pair, where
    // ALPHA is the second word.
    expect(tokens.hoveredMatches.map((m) => m.candidateIdx)).toEqual([1])

    // ...and nothing drew it against the manuscript still on screen, whose
    // second word is "primus" and matches nothing in the current evidence.
    const stale = frames.filter(
      (frame) =>
        frame.candidateWords.includes('primus') &&
        (frame.paths.length > 0 || frame.highlighted.includes('primus')),
    )
    expect(stale).toEqual([])

    await waitFor(() => expect(screen.getByText('tertius')).toBeTruthy())
  })

  it('does not let coordinates measured for the outgoing panel become the incoming one’s', async () => {
    secondDirMoved = true
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    fireEvent.mouseEnter(queryToken())
    await screen.findByText('tertius')
    await settleCrossfade()

    const target = document.querySelector<HTMLElement>(
      '[data-side="candidate"][data-token-idx="1"]',
    )
    expect(target?.getBoundingClientRect().left).toBe(460)

    // The hover survives the transition, so there IS a line; it has to have
    // been re-measured rather than kept from the panel that has gone.
    const paths = renderedPaths()
    expect(paths.length).toBeGreaterThan(0)
    expect(paths.every((d) => d.endsWith('460 30'))).toBe(true)
  })

  it('still draws evidence for a settled candidate, crossfade and all', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    await screen.findByText('tertius')
    await settleCrossfade()

    fireEvent.mouseEnter(queryToken())
    await waitFor(() => expect(renderedPaths().length).toBeGreaterThan(0))
    expect(screen.getByText('ALPHA', { selector: '[data-side="candidate"]' })).toBeTruthy()
  })
})

describe('a control acts for the group it was rendered for', () => {
  it('refuses a selection made in the group that is animating away', async () => {
    sharedFilenames = true
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    // The first group's strip is still mounted and still interactive, and the
    // second group has a b.txt of its own.
    expect(screen.getByText('primus')).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Viewing witness'), {
      target: { value: 'b.txt' },
    })
    await settleCrossfade()

    const selector = screen.getByLabelText('Viewing witness') as HTMLSelectElement
    expect(selector.value).toBe('a.txt')
    expect(screen.getByText('tertius')).toBeTruthy()
    expect(screen.queryByText('quartus')).toBeNull()
  })

  it('refuses the same selection from the keyboard shortcut, not only the select', async () => {
    sharedFilenames = true
    designateSupport = true
    renderApp()
    // Opens on the designated support, b.txt.
    await screen.findByText('secundus')
    await settleCrossfade()

    // Leave the designated support, so the first group offers the shortcut
    // back to it.
    fireEvent.change(screen.getByLabelText('Viewing witness'), {
      target: { value: 'a.txt' },
    })
    const shortcut = await screen.findByTestId('show-supporting-witness')

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    expect(screen.getByText('primus')).toBeTruthy()

    // Enter on a focused button is a click: the accessible route has to be
    // refused for the same reason the pointer one is. The shortcut names
    // b.txt, which the incoming group also has, and whose text is "quartus".
    shortcut.focus()
    await userEvent.keyboard('{Enter}')
    await settleCrossfade()

    expect(screen.getByText('tertius')).toBeTruthy()
    expect(screen.queryByText('quartus')).toBeNull()
    expect(
      (screen.getByLabelText('Viewing witness') as HTMLSelectElement).value,
    ).toBe('a.txt')
  })

  it('takes an ordinary selection in the group on screen, by pointer and by key', async () => {
    designateSupport = true
    renderApp()
    // Opens on the designated support, b.txt.
    await screen.findByText('secundus')
    await settleCrossfade()

    await userEvent.selectOptions(
      screen.getByLabelText('Viewing witness'),
      'a.txt',
    )
    expect(screen.getByText('primus')).toBeTruthy()

    const shortcut = await screen.findByTestId('show-supporting-witness')
    shortcut.focus()
    await userEvent.keyboard('{Enter}')
    await waitFor(() => expect(screen.getByText('secundus')).toBeTruthy())
    expect(
      (screen.getByLabelText('Viewing witness') as HTMLSelectElement).value,
    ).toBe('b.txt')
  })
})

describe('applicability reaches the sidebar in the frame the words change', () => {
  it('commits no frame pairing a new witness with the previous witness’s controls', async () => {
    artifact = true
    artifactWitness = 'a.txt'
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    frames = []

    fireEvent.change(screen.getByLabelText('Viewing witness'), {
      target: { value: 'b.txt' },
    })

    // Not an end-state assertion: the final DOM was always right, and only an
    // intermediate commit was not.
    const bFrames = frames.filter((frame) => frame.witness === 'b.txt')
    expect(bFrames.length).toBeGreaterThan(0)
    expect(bFrames.some((frame) => frame.methodControls)).toBe(false)
    expect(screen.queryByRole('radiogroup', { name: 'Attribution method' })).toBeNull()
  })

  it('commits no such frame when the whole candidate changes either', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    await settleCrossfade()
    frames = []

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    await screen.findByText('tertius')

    const secondDirFrames = frames.filter((frame) =>
      frame.candidateWords.includes('tertius'),
    )
    expect(secondDirFrames.length).toBeGreaterThan(0)
    // The artifact for the second directory names ITS first witness, so the
    // controls are legitimate again -- but only once it has arrived.
    await waitFor(() =>
      expect(
        screen.getByRole('radiogroup', { name: 'Attribution method' }),
      ).toBeTruthy(),
    )
  })

  it('keeps the controls for the witness the artifact does describe', async () => {
    artifact = true
    artifactWitness = 'a.txt'
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })

    // Away and back: the gate is about currentness, not about destroying the
    // artifact or the reviewer's method preference.
    fireEvent.change(screen.getByLabelText('Viewing witness'), {
      target: { value: 'b.txt' },
    })
    expect(screen.queryByRole('radiogroup', { name: 'Attribution method' })).toBeNull()

    fireEvent.change(screen.getByLabelText('Viewing witness'), {
      target: { value: 'a.txt' },
    })
    await waitFor(() =>
      expect(
        screen.getByRole('radiogroup', { name: 'Attribution method' }),
      ).toBeTruthy(),
    )
    expect(tokens.selectedMethod).toBe('ig')
  })
})
