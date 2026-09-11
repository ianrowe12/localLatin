import { useEffect, useLayoutEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { TokenProvider, useTokens } from '../../contexts/TokenContext'
import AttributionMethodSelector from '../common/AttributionMethodSelector'
import CenterArea from './CenterArea'

/**
 * Everything downstream of the member selector, mounted (issue #163).
 *
 * The other mounted suites ask which witness is on screen. This one asks what
 * the rest of the view still believes about it, and it exists because three
 * consumers kept the previous witness's evidence in ways a "which words are
 * rendered" assertion cannot see:
 *
 *   1. a memoised token span that compared appearance but not behaviour, so a
 *      live hover handler still held the old member's matches;
 *   2. an `AnimatePresence` that outlived the state it drew, so removed
 *      connection lines stayed on screen over the new manuscript;
 *   3. attribution metadata read from the directory's artifact before anyone
 *      asked whether that artifact describes the member on screen.
 *
 * Three fixture choices are load-bearing, because the default ones hide all
 * three faults:
 *
 *   - the query word appears in EVERY witness, with the same row maximum, so
 *     the spans' visible props are identical across the switch and only the
 *     handler's behaviour differs. A query word absent from the candidates
 *     changes the shading and refreshes the memo by accident.
 *   - `getBoundingClientRect` returns real geometry. jsdom's zeros make every
 *     connection invisible, so retained paths cannot be observed at all.
 *   - `AttributionMethodSelector` is mounted as the sidebar sibling it really
 *     is, OUTSIDE the panel's witness scope, which is why masking context
 *     below the panel cannot fix what it reads.
 */

const MODEL = 'google_mt5-base'
const DIR = 'CANT.328.12'
const QUERY_ID = 7

/** Both witnesses contain ALPHA, at different positions. */
const WITNESSES = {
  a: { filename: 'a.txt', text: 'ALPHA primus' },
  b: { filename: 'b.txt', text: 'secundus ALPHA' },
}

let files: { filename: string; text: string }[]
/** Serve an artifact for this directory. */
let artifact: boolean
/** Which witness the artifact was built from. */
let artifactWitness: string
/** Serve the OT method with no matrix cell for sif_abtt (a real shape). */
let sparseOtCell: boolean
let requests: string[]
let tokens: ReturnType<typeof useTokens>
/** What each committed frame actually showed. */
let commits: { witness: string; paths: number }[]

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function Probe() {
  tokens = useTokens()
  useLayoutEffect(() => {
    const selector = document.querySelector<HTMLSelectElement>(
      '[data-testid="member-selector"]',
    )
    commits.push({
      witness: selector?.value ?? '',
      paths: document.querySelectorAll('svg path[d]').length,
    })
  })
  return null
}

function Harness() {
  const { setActiveQueryId, setActiveModel } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return (
    <>
      {/* A sidebar sibling of the panel, exactly as LeftSidebar mounts it. */}
      <AttributionMethodSelector />
      <CenterArea />
      <Probe />
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

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost')
      requests.push(url.toString())

      if (url.pathname.endsWith('/predictions')) {
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
              dir_name: DIR,
              score: 0.9,
              source: 'model',
              dir_files: files.map((file) => file.filename),
              candidate_files: files,
              preview_text: files[0].text,
              supporting_member: null,
            },
          ],
        })
      }

      if (url.pathname.endsWith('/token_map')) {
        if (!artifact) {
          return jsonResponse({ error: { message: 'no artifact' } }, 404)
        }
        const method = url.searchParams.get('method') ?? 'ig'
        const witness = files.find((f) => f.filename === artifactWitness)
        return jsonResponse({
          example_id: 123,
          model: MODEL,
          layer: 4,
          D: 10,
          bucket: 'unlabelled_bulk',
          query_path: `data/canon_unlabelled/query-${QUERY_ID}.txt`,
          // Names ONE witness, which is what the applicability test reads.
          candidate_path: `data/canon_labelled/${DIR}/${artifactWitness}`,
          query_tokens: [{ idx: 0, text: 'ALPHA', is_content: true }],
          candidate_tokens: (witness?.text ?? '')
            .split(' ')
            .map((text, idx) => ({ idx, text, is_content: true })),
          similarity_matrix: [[1, 0]],
          ig_weighted_matrix: null,
          top_matches: {},
          query_ig_baseline: [],
          query_ig_abtt: [],
          candidate_ig_baseline: [],
          candidate_ig_abtt: [],
          auto_highlights: null,
          // The union across the artifact's variants, which is why a listed
          // method can still have no cell for the requested one
          // (web/services/token_map_svc.py).
          available_methods: ['ig', 'ot'],
          available_variants: ['sif_abtt'],
          pair_matrices:
            sparseOtCell && method === 'ot'
              ? {}
              : { [method]: { sif_abtt: [[1, 0]] } },
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
  files = [WITNESSES.a, WITNESSES.b]
  artifact = false
  artifactWitness = WITNESSES.a.filename
  sparseOtCell = false
  requests = []
  commits = []
  // Real geometry. Without it every connection is `visible: false` and the
  // retained-path assertions below can never fail, which is exactly why the
  // fault survived the existing suites.
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
    function (this: HTMLElement) {
      const side = this.getAttribute('data-side')
      const index = Number(this.getAttribute('data-token-idx') ?? 0)
      if (side === 'query') return new DOMRect(10 + index * 60, 20, 50, 20)
      if (side === 'candidate') return new DOMRect(250 + index * 60, 20, 50, 20)
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

function svgPaths(): { d: string | null; opacity: string | null }[] {
  return Array.from(document.querySelectorAll('svg path[d]')).map((path) => ({
    d: path.getAttribute('d'),
    opacity: path.getAttribute('opacity'),
  }))
}

function chooseWitness(filename: string): void {
  fireEvent.change(screen.getByLabelText('Viewing witness'), {
    target: { value: filename },
  })
}

/** Let the exit animation finish, so a retained path is a choice, not a frame. */
async function settleAnimations(): Promise<void> {
  await act(() => new Promise((resolve) => setTimeout(resolve, 400)))
}

describe('a live handler reports the witness on screen', () => {
  it('answers a fresh hover with the displayed member, not the previous one', async () => {
    renderApp()
    await screen.findByText('primus')

    // ALPHA is candidate index 0 in a.txt and index 1 in b.txt. Its row
    // maximum is 1 either way, so the span's shading -- and every other
    // visible prop -- is identical across the switch.
    chooseWitness(WITNESSES.b.filename)
    expect(screen.getByText('secundus')).toBeTruthy()

    fireEvent.mouseEnter(queryToken())
    expect(tokens.hoveredMatches.map((match) => match.candidateIdx)).toEqual([1])
  })

  it('answers keyboard focus the same way, having left an attributed witness', async () => {
    // Stricter: a.txt no longer contains ALPHA at all, so the only thing that
    // could name candidate index 0 is the artifact built from a.txt. Focus and
    // hover share the handler, and the keyboard path must not be the way a
    // different manuscript's model evidence returns.
    artifact = true
    files = [{ filename: 'a.txt', text: 'TERTIUS primus' }, WITNESSES.b]
    renderApp()
    await screen.findByText('primus')
    await screen.findByRole('radiogroup', { name: 'Attribution method' })

    chooseWitness(WITNESSES.b.filename)
    expect(screen.getByText('secundus')).toBeTruthy()
    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('different witness')

    fireEvent.focus(queryToken())
    expect(tokens.hoveredMatches.map((match) => match.candidateIdx)).toEqual([1])
  })
})

describe('no connection outlives the words it was drawn between', () => {
  // Hover is the reachable user path. The other two use the provider's own
  // writers, which is the contract those fields are under, rather than a claim
  // that a pin control is currently mounted anywhere.
  it.each([
    ['a hover', 'hover'],
    ['a manual pin', 'pin'],
    ['an auto-highlight', 'auto'],
  ])('removes %s line in the commit that changes the witness', async (_name, kind) => {
    renderApp()
    await screen.findByText('primus')

    if (kind === 'hover') {
      fireEvent.mouseEnter(queryToken())
    } else {
      act(() => {
        const matches = [{ candidateIdx: 0, score: 1, rank: 0 }]
        if (kind === 'pin') tokens.pinToken(0, matches)
        else tokens.applyAutoHighlights([{ queryIdx: 0, matches }])
      })
    }
    await waitFor(() => expect(svgPaths().length).toBeGreaterThan(0))
    await settleAnimations()

    commits = []
    chooseWitness(WITNESSES.b.filename)

    // The scoped fields really are empty...
    expect(tokens.hoveredMatches).toEqual([])
    expect(tokens.pinnedTokens.size).toBe(0)
    expect(tokens.autoHighlightedTokens.size).toBe(0)
    // ...and no path survived them into the new witness's words. An exiting
    // child is no longer in `activeConnections`, so asserting that list alone
    // would pass over a line still on screen.
    expect(svgPaths()).toEqual([])
    expect(commits.every((commit) => commit.paths === 0)).toBe(true)
    expect(commits[0]?.witness).toBe(WITNESSES.b.filename)
  })

  it('still animates a line away when only the hover ends', async () => {
    // The presence owner is discarded on a witness change, not on every
    // change: leaving a token is still an exit animation, which is what the
    // overlay is for.
    renderApp()
    await screen.findByText('primus')
    fireEvent.mouseEnter(queryToken())
    await waitFor(() => expect(svgPaths().length).toBeGreaterThan(0))

    fireEvent.mouseLeave(queryToken())
    expect(tokens.hoveredMatches).toEqual([])
    expect(svgPaths().length).toBeGreaterThan(0)

    await settleAnimations()
    expect(svgPaths()).toEqual([])
  })
})

describe('attribution metadata describes the displayed witness', () => {
  it('offers no method controls while the artifact is about another member', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })

    chooseWitness(WITNESSES.b.filename)

    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('different witness')
    // The selector is a sidebar sibling reading the outer provider, so this is
    // not something the panel's scope could have masked.
    expect(tokens.availableMethods).toEqual([])
    expect(
      screen.queryByRole('radiogroup', { name: 'Attribution method' }),
    ).toBeNull()
  })

  it('lets a sibling artifact suppress nothing about this witness', async () => {
    // `available_methods` is the union across the artifact's variants, so OT
    // can be listed with no cell for sif_abtt. That is a fact about a.txt's
    // pair. Applied to b.txt it replaced a perfectly good word-overlap
    // comparison with a notice about a pair nobody is looking at.
    artifact = true
    sparseOtCell = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    act(() => tokens.setSelectedMethod('ot'))
    await screen.findByText(/No OT attribution for this pair/)

    chooseWitness(WITNESSES.b.filename)

    expect(screen.getByText('secundus')).toBeTruthy()
    expect(screen.queryByText(/No OT attribution for this pair/)).toBeNull()
    // ALPHA overlaps b.txt word for word, and that is real evidence about the
    // text on screen, so it is shaded and said to be word overlap.
    expect(queryToken().style.backgroundColor).not.toBe('')
    expect(
      screen.getByTestId('member-evidence-model-attribution').textContent,
    ).toContain('word overlap')
  })

  it('keeps the missing-method notice when the artifact IS about this witness', async () => {
    // The #156 behaviour this must not weaken: for the pair actually on
    // screen, a method with no matrix is stated rather than quietly replaced
    // by a cosine grid that cannot answer the question just asked.
    artifact = true
    sparseOtCell = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    act(() => tokens.setSelectedMethod('ot'))

    expect(await screen.findByText(/No OT attribution for this pair/)).toBeTruthy()
    expect(screen.queryByTestId('member-evidence-model-attribution')).toBeNull()
  })

  it('keeps preferences and the cached artifact across a round trip', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    act(() => {
      tokens.setSelectedMethod('ot')
      tokens.setViewMode('heatmap')
    })
    await waitFor(() =>
      expect(requests.some((url) => url.includes('method=ot'))).toBe(true),
    )
    await act(() => Promise.resolve())
    const before = requests.length

    chooseWitness(WITNESSES.b.filename)
    expect(tokens.selectedMethod).toBe('ot')
    expect(tokens.viewMode).toBe('heatmap')

    chooseWitness(WITNESSES.a.filename)
    expect(tokens.selectedMethod).toBe('ot')
    expect(tokens.viewMode).toBe('heatmap')
    // Withholding an artifact from one member does not discard it: coming
    // back costs no request, and the attribution is live again.
    expect(requests.length).toBe(before)
    expect(screen.queryByTestId('member-evidence-model-attribution')).toBeNull()
    expect(queryToken()).toBeTruthy()
  })
})
