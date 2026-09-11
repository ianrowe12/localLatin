import { Profiler, useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider, usePredictionState } from '../../contexts/PredictionContext'
import { TokenProvider, useTokens } from '../../contexts/TokenContext'
import AttributionMethodSelector from '../common/AttributionMethodSelector'
import CenterArea from './CenterArea'

/**
 * Ownership of the marks, and of the right to advertise evidence (issue #163).
 *
 * `CenterAreaDisplayedPair` established that refs, control events and
 * applicability are keyed by the pair that rendered them. Two consequences of
 * that were left unfinished, and both are about the same mistake: believing a
 * value because of who is holding it rather than because it is still true.
 *
 *   1. A panel animating away kept its own props but went on READING the
 *      shared token state, which the scope above had already advanced to the
 *      incoming pair. Owning the ref callbacks cannot reach this: context
 *      re-renders the retained panel, and highlight, pin and dim props are all
 *      recomputed before `TokenSpan` ever compares anything. So a hover into
 *      the new manuscript repainted a word in the old one -- a claim about a
 *      document the reviewer had already left.
 *
 *   2. Applicability was published as two strings written together. Comparing
 *      them proves they were written together, not that they are current: after
 *      a shared refresh they still agree, about a request that no longer
 *      exists. The first committed frames of that refresh have no candidate
 *      words at all and still offered the previous witness's method controls.
 *
 * The fixture gives the query TWO words, so the incoming pair's match lands on
 * a DIFFERENT index from the outgoing panel's word. Without that, a leaked
 * hover and a correct one are indistinguishable.
 *
 *   query               ALPHA BETA
 *   CANT.328.12/a.txt   ALPHA primus     <- outgoing, index 1 is "primus"
 *   CSAR.347.17/a.txt   tertius BETA     <- incoming, index 1 is the match
 *
 * Style attributes are read straight off the DOM, because that is the level the
 * defect lives at: the candidate spans are not interactive and carry no
 * aria-label, so an accessible-name assertion cannot see them change colour.
 */

const MODEL = 'google_mt5-base'
const FIRST_DIR = 'CANT.328.12'
const SECOND_DIR = 'CSAR.347.17'
const GALLERY_DIR = 'PARI.1454.03'
const QUERY_ID = 7

let artifact: boolean
/** Hold the replacement ranking open, so a refresh's pending frames are real. */
let rankingPending: boolean
let tokens: ReturnType<typeof useTokens>
let phase: string
let frames: {
  phase: string
  candidateWords: string[]
  methodControls: boolean
}[]

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
    phase,
    candidateWords: candidateTokenEls().map((el) => el.textContent ?? ''),
    methodControls: !!document.querySelector('[aria-label="Attribution method"]'),
  })
}

function Harness() {
  const {
    setActiveQueryId,
    setActiveModel,
    setActivePredictionRank,
    setOverrideCandidateDir,
  } = useApp()
  const predictions = usePredictionState()
  phase = predictions.phase
  tokens = useTokens()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL)
  }, [setActiveQueryId, setActiveModel])
  return (
    <>
      <button type="button" onClick={() => setActivePredictionRank(2)}>
        Open second directory
      </button>
      <button type="button" onClick={predictions.refresh}>
        Refresh ranking
      </button>
      <button type="button" onClick={() => setOverrideCandidateDir(GALLERY_DIR)}>
        Open example
      </button>
    </>
  )
}

function renderApp() {
  return render(
    <AppProvider>
      <TokenProvider>
        <PredictionProvider>
          <Profiler id="ownership" onRender={observe}>
            <Harness />
            {/* The sidebar sibling, outside every scope the panel can apply. */}
            <AttributionMethodSelector />
            <CenterArea />
          </Profiler>
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

      if (url.pathname.endsWith('/predictions')) {
        // Never resolves: the reviewer is looking at the app between a refresh
        // and its replacement, which is the whole window under test.
        if (rankingPending) return new Promise<Response>(() => {})
        const first = [
          { filename: 'a.txt', text: 'ALPHA primus' },
          { filename: 'b.txt', text: 'secundus ALPHA' },
        ]
        const second = [
          { filename: 'a.txt', text: 'tertius BETA' },
          { filename: 'b.txt', text: 'quartus ALPHA' },
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
              supporting_member: null,
            },
            {
              rank: 2,
              dir_name: SECOND_DIR,
              score: 0.7,
              source: 'model',
              dir_files: second.map((f) => f.filename),
              candidate_files: second,
              preview_text: second[0].text,
              supporting_member: null,
            },
          ],
        })
      }

      const galleryFiles = url.pathname.match(/^\/api\/candidate_dir\/(.+)\/files$/)
      if (galleryFiles) {
        return jsonResponse([{ filename: 'a.txt', text: 'quintus ALPHA' }])
      }

      if (url.pathname.endsWith('/token_map')) {
        if (!artifact) return jsonResponse({ error: { message: 'no artifact' } }, 404)
        const dir = url.searchParams.get('candidate_dir')
        const method = url.searchParams.get('method') ?? 'ig'
        return jsonResponse({
          example_id: 123,
          model: MODEL,
          layer: 4,
          D: 10,
          bucket: 'unlabelled_bulk',
          query_path: `data/canon_unlabelled/query-${QUERY_ID}.txt`,
          candidate_path: `data/canon_labelled/${dir}/a.txt`,
          query_tokens: [],
          candidate_tokens: [],
          similarity_matrix: [[1, 0]],
          ig_weighted_matrix: null,
          top_matches: {},
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
          text: 'ALPHA BETA',
          tokens: [
            { text: 'ALPHA', index: 0, category: 'content' },
            { text: 'BETA', index: 1, category: 'content' },
          ],
          char_count: 10,
          token_count: 2,
        })
      }

      return jsonResponse({})
    }),
  )
}

beforeEach(() => {
  artifact = false
  rankingPending = false
  frames = []
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

/** Long enough for framer-motion's 0.2s candidate crossfade to finish. */
async function settleCrossfade(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 300))
  })
}

function candidateWord(text: string): HTMLElement {
  return screen.getByText(text, { selector: '[data-side="candidate"]' })
}

function hoverQuery(text: string): void {
  fireEvent.mouseEnter(screen.getByLabelText(`query token ${text}, highlighted evidence`))
}

describe('a retained panel reads only the marks that are about its own words', () => {
  it('does not repaint an exiting witness from a hover into the incoming one', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    // The outgoing manuscript is still on screen: that is the crossfade, and
    // it stays.
    const outgoing = candidateWord('ALPHA')
    const before = outgoing.style.backgroundColor
    expect(before).not.toBe('')

    hoverQuery('BETA')

    // The hover itself is right: it reports the INCOMING pair, where the match
    // for BETA is the second word.
    expect(tokens.hoveredMatches.map((m) => m.candidateIdx)).toEqual([1])
    // And the words it is not about were not touched by it.
    expect(outgoing.isConnected).toBe(true)
    expect(outgoing.style.backgroundColor).toBe(before)
  })

  it('does not give an exiting witness a pin aimed at the incoming one', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    // Index 1 of the outgoing panel. In the incoming panel index 1 is the
    // match for BETA, so a pin that leaks lands here and says so in orange.
    const outgoing = screen.getByText('primus')
    const before = outgoing.getAttribute('style')

    act(() => {
      tokens.pinToken(0, [{ candidateIdx: 1, score: 1, rank: 0 }])
    })

    expect(outgoing.isConnected).toBe(true)
    expect(outgoing.getAttribute('style')).toBe(before)
  })

  it('does not give an exiting witness an automatic highlight aimed at the incoming one', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()

    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    const outgoing = screen.getByText('primus')
    const before = outgoing.getAttribute('style')

    act(() => {
      tokens.applyAutoHighlights([
        { queryIdx: 0, matches: [{ candidateIdx: 1, score: 1, rank: 0 }] },
      ])
    })

    expect(outgoing.isConnected).toBe(true)
    expect(outgoing.getAttribute('style')).toBe(before)
  })

  it('still marks the witness that has entered', async () => {
    renderApp()
    await screen.findByText('primus')
    await settleCrossfade()
    fireEvent.click(screen.getByRole('button', { name: 'Open second directory' }))
    await screen.findByText('tertius')
    await settleCrossfade()

    // The over-fix guard. A gate that simply stopped panels reading shared
    // state would pass every test above and leave hover, pins and automatic
    // highlights doing nothing at all. All three are gated by the same test,
    // so all three are checked here.
    const entered = candidateWord('BETA')
    const base = entered.getAttribute('style')

    hoverQuery('BETA')
    expect(tokens.hoveredMatches.map((m) => m.candidateIdx)).toEqual([1])
    await waitFor(() =>
      expect(document.querySelectorAll('svg path[d]').length).toBeGreaterThan(0),
    )

    act(() => {
      tokens.pinToken(1, [{ candidateIdx: 1, score: 1, rank: 0 }])
    })
    const pinned = entered.getAttribute('style')
    expect(pinned).not.toBe(base)
    expect(pinned).toContain('box-shadow')

    act(() => {
      tokens.clearAllPins()
      tokens.applyAutoHighlights([
        { queryIdx: 1, matches: [{ candidateIdx: 1, score: 1, rank: 0 }] },
      ])
    })
    expect(entered.getAttribute('style')).not.toBe(base)
  })
})

describe('applicability is checked against the live request, not two stored strings', () => {
  it('offers no method controls in a committed frame of a refresh with no words', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })

    rankingPending = true
    frames = []
    fireEvent.click(screen.getByRole('button', { name: 'Refresh ranking' }))

    // Not an end-state assertion. Both stored strings still describe the pair
    // that has just been superseded, and they still agree with each other, so
    // only a live comparison can close this window -- and it has to close it in
    // the commit itself, not in an effect that produces a second one.
    const pending = frames.filter(
      (frame) => frame.phase === 'loading' && frame.candidateWords.length === 0,
    )
    expect(pending.length).toBeGreaterThan(0)
    expect(pending.some((frame) => frame.methodControls)).toBe(false)
  })

  it('offers them again once the refreshed ranking settles', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })

    // The over-fix guard for the same gate: refusing a stale publication must
    // not mean refusing to publish again, and a reviewer who refreshes must
    // get their evidence controls back.
    fireEvent.click(screen.getByRole('button', { name: 'Refresh ranking' }))
    await settleCrossfade()

    await waitFor(() =>
      expect(
        screen.getByRole('radiogroup', { name: 'Attribution method' }),
      ).toBeTruthy(),
    )
    expect(screen.getByText('primus')).toBeTruthy()
  })

  it('keeps an open example’s controls while the ranking underneath it refreshes', async () => {
    artifact = true
    renderApp()
    await screen.findByRole('radiogroup', { name: 'Attribution method' })
    fireEvent.click(screen.getByRole('button', { name: 'Open example' }))
    await screen.findByText('quintus')
    await settleCrossfade()
    expect(
      screen.getByRole('radiogroup', { name: 'Attribution method' }),
    ).toBeTruthy()

    // A gallery inspection carries its own candidate and its own files, so a
    // ranking refresh underneath it changes nothing it is showing. Binding its
    // applicability to the ranking's generation would blank controls that are
    // still describing exactly the pair on screen.
    rankingPending = true
    fireEvent.click(screen.getByRole('button', { name: 'Refresh ranking' }))
    await settleCrossfade()

    expect(screen.getByText('quintus')).toBeTruthy()
    expect(
      screen.getByRole('radiogroup', { name: 'Attribution method' }),
    ).toBeTruthy()
  })
})
