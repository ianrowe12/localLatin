import { useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { TokenProvider } from '../../contexts/TokenContext'
import { REVIEWER_DIRS_UPDATED_EVENT } from '../../api/reviewerDirs'
import ModelSelector from '../predictions/ModelSelector'
import CenterArea from './CenterArea'

/**
 * What is on the candidate panel RIGHT NOW (issue #156).
 *
 * The shared state already refuses to hand out old-key data, but these assert
 * against the mounted component at the exact commit after the change, not after
 * everything settles. That gap is where the two P1 defects lived: an exit
 * animation kept a superseded candidate legible for 200ms, and a gallery
 * override outlived the document it was opened from.
 *
 * Every change below is driven with `fireEvent` or `act`, which flush
 * synchronously. `userEvent` awaits, and awaiting is exactly what let the stale
 * evidence animate away before anybody looked.
 */

const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'google_mt5-base'
const QUERY_ID = 7
const OTHER_QUERY_ID = 8
const GALLERY_DIR = 'GALLERY.DIR'
const GALLERY_DIR_2 = 'OTHER.GALLERY.DIR'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const sleep = (ms: number) =>
  ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve()

function modelEntry(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'sif',
    prediction_count: 2238,
    available_variants: ['sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.5, verify: 0.7 },
    supports_reviewer_dirs: true,
  }
}

type CandidateFile = { filename: string; text: string }
type Candidate = { dir: string; files: CandidateFile[] }

/** Ranked candidates per model, rank 1 first. */
let candidates: Record<string, Candidate[]> = {}
let predictionDelay: Record<string, number> = {}
let requested: string[] = []
/** Files served for an on-demand gallery directory fetch, by directory. */
let galleryDirs: Record<string, CandidateFile[]> = {}
/** Directories whose files request fails, and how many times it still will. */
let galleryFailures: Record<string, number> = {}
/**
 * Token maps by query id. `highlight` serves a matrix that lights the first
 * candidate token; `pending` never resolves within the test; anything else 404s,
 * which is the ordinary "no artifact for this pair" answer.
 */
let tokenMaps: Record<number, 'highlight' | 'pending'> = {}

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
        return jsonResponse([
          modelEntry(MODEL_A, 'LaTa (T5)'),
          modelEntry(MODEL_B, 'mT5-base'),
        ])
      }
      if (url.includes('/token_map')) {
        const queryId = Number(parsed.pathname.split('/')[3])
        const mode = tokenMaps[queryId]
        if (mode === 'pending') {
          await sleep(10_000)
        }
        if (mode !== 'highlight') {
          return jsonResponse({ error: { message: 'no artifact' } }, 404)
        }
        // Two query tokens, two candidate tokens. Column max is what
        // DocumentPanel shades the candidate with, so this lights candidate
        // token 0 and nothing else.
        return jsonResponse({
          example_id: `ex-${queryId}`,
          query_file_id: queryId,
          candidate_dir: params.get('candidate_dir'),
          candidate_path: 'x/c.txt',
          query_tokens: [],
          candidate_tokens: [],
          similarity_matrix: [
            [1, 0],
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
          // No attribution in the artifact, so the payload's own cosine matrix
          // is the honest whole story and CenterArea uses it directly.
          available_methods: [],
          pair_matrices: {},
        })
      }
      if (url.includes('/candidate_dir/')) {
        const dir = decodeURIComponent(
          parsed.pathname.split('/candidate_dir/')[1].split('/')[0],
        )
        if ((galleryFailures[dir] ?? 0) > 0) {
          galleryFailures[dir] -= 1
          return jsonResponse(
            { error: { code: 'unavailable', message: 'directory files unavailable' } },
            503,
          )
        }
        return jsonResponse(galleryDirs[dir] ?? [])
      }
      if (url.includes('/predictions')) {
        const model = params.get('model') ?? ''
        const queryId = Number(parsed.pathname.split('/')[3])
        await sleep(predictionDelay[model] ?? 0)
        return jsonResponse({
          file_id: queryId,
          filename: `query-${queryId}.txt`,
          model,
          variant: 'sif_abtt',
          status: 'ok',
          predictions: (candidates[model] ?? []).map((candidate, i) => ({
            rank: i + 1,
            dir_name: candidate.dir,
            score: 0.9 - i * 0.1,
            dir_files: candidate.files.map((f) => f.filename),
            preview_text: 'preview',
            candidate_files: candidate.files,
            source: 'model',
          })),
          seeded_dirs: [],
        })
      }
      if (/\/api\/query\/\d+$/.test(parsed.pathname)) {
        const queryId = Number(parsed.pathname.split('/')[3])
        const words = [`QUERYWORD${queryId}`, 'uerba']
        return jsonResponse({
          file_id: queryId,
          filename: `query-${queryId}.txt`,
          text: words.join(' '),
          tokens: words.map((text, index) => ({
            text,
            index,
            category: 'content',
          })),
          char_count: 20,
          token_count: words.length,
        })
      }
      return jsonResponse({})
    }),
  )
}

/** The navigations a reviewer can actually make, as real controls. */
function Harness() {
  const {
    setActiveQueryId,
    setActiveModel,
    setActivePredictionRank,
    navigateToExample,
    navigateToQuery,
  } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
    setActiveModel(MODEL_A)
  }, [setActiveQueryId, setActiveModel])
  return (
    <>
      <ModelSelector />
      <button
        type="button"
        data-testid="open-gallery-pair"
        onClick={() => navigateToExample(MODEL_A, QUERY_ID, GALLERY_DIR)}
      >
        open gallery pair
      </button>
      <button
        type="button"
        data-testid="open-gallery-pair-other-query"
        onClick={() => navigateToExample(MODEL_A, OTHER_QUERY_ID, GALLERY_DIR)}
      >
        open the same directory for the next document
      </button>
      <button
        type="button"
        data-testid="open-second-gallery-pair"
        onClick={() => navigateToExample(MODEL_A, QUERY_ID, GALLERY_DIR_2)}
      >
        open the other example
      </button>
      <button
        type="button"
        data-testid="open-second-gallery-pair-other-query"
        onClick={() => navigateToExample(MODEL_A, OTHER_QUERY_ID, GALLERY_DIR_2)}
      >
        open the other example for the next document
      </button>
      <button
        type="button"
        data-testid="go-to-other-query"
        onClick={() => navigateToQuery(OTHER_QUERY_ID)}
      >
        next document
      </button>
      <button
        type="button"
        data-testid="choose-rank-2"
        onClick={() => setActivePredictionRank(2)}
      >
        second candidate
      </button>
      <CenterArea />
    </>
  )
}

function renderCenter() {
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

/** Token-map requests as `queryId:candidate_dir`. */
function tokenMapPairs(): string[] {
  return requested
    .filter((url) => url.includes('/token_map'))
    .map((url) => {
      const parsed = new URL(url, 'http://localhost')
      const queryId = parsed.pathname.split('/')[3]
      return `${queryId}:${parsed.searchParams.get('candidate_dir')}`
    })
}

/**
 * Candidate words currently carrying attribution shading.
 *
 * Candidate spans are non-interactive and so have no accessible name; the
 * shading is an inline background style (TokenSpan), which is exactly what a
 * reviewer sees. This is the only way to tell "the previous pair's highlights"
 * from "the same words, unhighlighted".
 */
function highlightedCandidateWords(): string[] {
  return Array.from(document.querySelectorAll('[data-side="candidate"]'))
    .filter((el) => (el as HTMLElement).style.backgroundColor !== '')
    .map((el) => el.textContent ?? '')
}

const settle = () => act(() => sleep(80))

beforeEach(() => {
  candidates = {
    [MODEL_A]: [{ dir: 'LATA.DIR', files: [{ filename: 'a.txt', text: 'LATAWORD' }] }],
    [MODEL_B]: [{ dir: 'MT5.DIR', files: [{ filename: 'b.txt', text: 'MT5WORD' }] }],
  }
  predictionDelay = {}
  galleryDirs = {
    [GALLERY_DIR]: [{ filename: 'g.txt', text: 'GALLERYWORD' }],
    [GALLERY_DIR_2]: [{ filename: 'h.txt', text: 'OTHERGALLERYWORD' }],
  }
  galleryFailures = {}
  tokenMaps = {}
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('stale evidence is removed, not animated out (issue #156)', () => {
  it('drops the previous model’s candidate in the same commit as the switch', async () => {
    // The reviewer switches model. Until the new ranking arrives there is no
    // candidate for this document, and "no candidate" has to look like no
    // candidate -- not like the previous model's answer fading under the new
    // model's name.
    predictionDelay[MODEL_B] = 120
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('Select model'), {
      target: { value: MODEL_B },
    })

    // Same commit. No awaiting, so the 0.2s exit animation has not run.
    expect(screen.queryByText('LATAWORD')).toBeNull()
    expect(screen.queryByText('LATA.DIR')).toBeNull()

    expect(await screen.findByText('MT5WORD')).toBeTruthy()
  })

  it('drops the current candidate the moment a refresh invalidates it', async () => {
    // Creating a reviewer directory broadcasts a refresh, which supersedes the
    // ranking on screen. The evidence belongs to the superseded generation.
    predictionDelay[MODEL_A] = 0
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    predictionDelay[MODEL_A] = 120
    act(() => {
      window.dispatchEvent(new CustomEvent(REVIEWER_DIRS_UPDATED_EVENT))
    })

    expect(screen.queryByText('LATAWORD')).toBeNull()
    await settle()
  })
})

describe('a gallery inspection cannot outlive its query (issue #156)', () => {
  it('is gone in the same commit as the navigation, and asks for no evidence about it', async () => {
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    expect(await screen.findByText('GALLERYWORD')).toBeTruthy()
    expect(screen.getByText(/Viewing example pair/)).toBeTruthy()

    fireEvent.click(screen.getByTestId('go-to-other-query'))

    // Same commit: the example belonged to the previous document.
    expect(screen.queryByText('GALLERYWORD')).toBeNull()
    expect(screen.queryByText(/Viewing example pair/)).toBeNull()

    await settle()
    // And no token-map request was ever made for the new document paired with
    // the old document's gallery directory.
    expect(tokenMapPairs()).not.toContain(`${OTHER_QUERY_ID}:${GALLERY_DIR}`)
    expect(
      tokenMapPairs().filter((pair) => pair.endsWith(GALLERY_DIR)),
    ).toEqual([`${QUERY_ID}:${GALLERY_DIR}`])
  })

  it('still lets the gallery open a pair for the document it names', async () => {
    // The guard is about identity, not about disabling off-list inspection.
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    expect(await screen.findByText('GALLERYWORD')).toBeTruthy()
    await settle()
    expect(tokenMapPairs()).toContain(`${QUERY_ID}:${GALLERY_DIR}`)
  })
})

describe('the blank-text notice describes the file on screen (issue #156)', () => {
  it('says which witness is blank when another one in the directory is not', async () => {
    // The panel always shows the first file. A whole-directory check called
    // this candidate readable, so the reviewer got a blank pane and no
    // explanation at all.
    candidates[MODEL_A] = [
      {
        dir: 'MIXED.DIR',
        files: [
          { filename: 'blank.txt', text: '   ' },
          { filename: 'readable.txt', text: 'READABLEWORD' },
        ],
      },
    ]
    renderCenter()

    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('blank.txt')
    expect(note.textContent).toContain('Other files in this directory do carry text')
    // And it does not pretend the other witness is what is being compared.
    expect(screen.queryByText('READABLEWORD')).toBeNull()
  })

  it('says the directory has no text when none of its witnesses does', async () => {
    candidates[MODEL_A] = [
      {
        dir: 'EMPTY.DIR',
        files: [
          { filename: 'blank.txt', text: '' },
          { filename: 'alsoblank.txt', text: '  ' },
        ],
      },
    ]
    renderCenter()

    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('This candidate directory has no readable text')
    expect(note.textContent).not.toContain('Other files')
  })

  it('says nothing when the witness on screen is readable', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()
    expect(screen.queryByTestId('candidate-evidence-note')).toBeNull()
  })

  it('explains a blank gallery witness too', async () => {
    // An inspection is exempt from the RANKING notices, not from an honest
    // account of the words it is showing.
    galleryDirs[GALLERY_DIR] = [{ filename: 'g.txt', text: '  ' }]
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('no readable text')
  })
})

describe('the animation boundary is the pair, not the panel (issue #156)', () => {
  it('drops the previous example when the next one is in the same directory', async () => {
    // Two gallery examples that share a candidate directory. Their evidence
    // identities differ only by query, both are non-null, and the directory's
    // files are already cached -- so nothing between them is ever null and
    // there is no loading render to fall back on. Keying the animated child
    // alone leaves the OUTGOING panel, with the first query's attribution,
    // mounted for the length of its exit.
    tokenMaps[QUERY_ID] = 'highlight'
    tokenMaps[OTHER_QUERY_ID] = 'pending'
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    await screen.findByText('GALLERYWORD')
    await waitFor(() =>
      expect(highlightedCandidateWords()).toEqual(['GALLERYWORD']),
    )

    fireEvent.click(screen.getByTestId('open-gallery-pair-other-query'))

    // Same commit. The words are the same words -- it is the same directory --
    // but they must no longer be dressed as evidence about the previous query.
    expect(highlightedCandidateWords()).toEqual([])
    await settle()
    expect(highlightedCandidateWords()).toEqual([])
  })

  it('does not resurrect an old example when the next directory is already cached', async () => {
    // A guard, not a reproduction: this path is additionally covered by the
    // files hook's own render-time key check, which reports the destination
    // directory as unloaded for one render. It is here so the boundary fix
    // above cannot quietly break the ordinary cached case.
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-second-gallery-pair'))
    await screen.findByText('OTHERGALLERYWORD')
    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    await screen.findByText('GALLERYWORD')

    fireEvent.click(screen.getByTestId('open-second-gallery-pair-other-query'))

    expect(screen.queryByText('GALLERYWORD')).toBeNull()
    expect(screen.getByText('OTHERGALLERYWORD')).toBeTruthy()
  })

  it('still crossfades between candidates of one settled ranking', async () => {
    // The boundary deliberately excludes the rank. Moving between two
    // candidates of the same request is the one transition where the outgoing
    // panel is wholly self-consistent -- its own filename, score and rank fade
    // out together -- so it keeps its animation.
    candidates[MODEL_A] = [
      { dir: 'FIRST.DIR', files: [{ filename: 'a.txt', text: 'FIRSTWORD' }] },
      { dir: 'SECOND.DIR', files: [{ filename: 'b.txt', text: 'SECONDWORD' }] },
    ]
    renderCenter()
    await screen.findByText('FIRSTWORD')

    fireEvent.click(screen.getByTestId('choose-rank-2'))
    // Retained on purpose: this is the exit animation, within one ranking.
    expect(screen.getByText('FIRSTWORD')).toBeTruthy()

    expect(await screen.findByText('SECONDWORD')).toBeTruthy()
    await waitFor(() => expect(screen.queryByText('FIRSTWORD')).toBeNull())
  })
})

describe('a failed example fetch is a failure, not an empty manuscript (issue #156)', () => {
  it('reports the load failure instead of claiming the directory has no text', async () => {
    // useCandidateDirFiles answers a failed request with data: null,
    // loading: false and an error. The first two alone look exactly like a
    // directory whose files carry no words, and the blank-text notice believed
    // them -- reporting a 503 as a fact about the manuscript.
    galleryFailures[GALLERY_DIR] = 1
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))

    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('did not load')
    expect(note.textContent).toContain('not an empty directory')
    expect(note.textContent).not.toContain('no readable text')
  })

  it('keeps the inspection open and retries when the example is reopened', async () => {
    galleryFailures[GALLERY_DIR] = 1
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    await screen.findByTestId('candidate-evidence-note')
    // The reviewer is still inside the example, with the way out in reach.
    expect(screen.getByText(/Viewing example pair/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Exit' }))
    fireEvent.click(screen.getByTestId('open-gallery-pair'))

    expect(await screen.findByText('GALLERYWORD')).toBeTruthy()
    expect(screen.queryByTestId('candidate-evidence-note')).toBeNull()
  })

  it('still explains a directory whose files loaded and carry no words', async () => {
    // The distinction only means something if the blank case survives it.
    galleryDirs[GALLERY_DIR] = [{ filename: 'g.txt', text: '' }]
    renderCenter()
    await screen.findByText('LATAWORD')

    fireEvent.click(screen.getByTestId('open-gallery-pair'))
    const note = await screen.findByTestId('candidate-evidence-note')
    expect(note.textContent).toContain('no readable text')
    expect(note.textContent).not.toContain('did not load')
  })
})
