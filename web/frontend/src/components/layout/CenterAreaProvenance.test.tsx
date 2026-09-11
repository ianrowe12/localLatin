import { useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import { TokenProvider } from '../../contexts/TokenContext'
import ModelSelector from '../predictions/ModelSelector'
import CenterArea from './CenterArea'

/**
 * What the mounted application says each panel is holding (issue #162).
 *
 * The leaf tests prove the caption function and the panel render every
 * provenance case when one is handed to them. These prove the real application
 * hands over the right one, which is the half that was deferred while issue
 * #156 owned the current-candidate identity.
 *
 * The claim under test is a factual one about a manuscript: a labelled-corpus
 * witness and a colleague's provisional grouping of originally unlabeled
 * documents are different kinds of evidence, and the panel used to call both of
 * them "Predicted Source". Getting that wrong on a stale render is the same
 * error as getting it wrong outright, so every navigation below is driven with
 * `fireEvent`/`act`, which flush synchronously, and the assertion is made at
 * the commit after the change rather than after everything settles.
 */

const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'google_mt5-base'
const QUERY_ID = 7
const OTHER_QUERY_ID = 8
const LABELLED_GALLERY_DIR = 'GALLERY.DIR'
const REVIEWER_GALLERY_DIR = 'reviewer-dir-9'

const QUERY_CAPTION = 'Query witness (unlabeled)'
const LABELED_CAPTION = 'Labeled-reference witness'
const REVIEWER_CAPTION = 'Reviewer group member (originally unlabeled)'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const sleep = (ms: number) =>
  ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve()

function modelEntry(
  slug: string,
  displayName: string,
  supportsReviewerDirs = true,
) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'sif',
    prediction_count: 2238,
    available_variants: ['sif_abtt'],
    default_variant: 'sif_abtt',
    confidence_bands: { no_match: 0.5, verify: 0.7 },
    supports_reviewer_dirs: supportsReviewerDirs,
  }
}

type CandidateFile = { filename: string; text: string }
type Candidate = {
  rank: number
  dir: string
  /** The server's own answer, exactly as `web/models.py` spells it. */
  source: 'model' | 'reviewer'
  label?: string
  files: CandidateFile[]
}

let candidates: Record<string, Candidate[]> = {}
let predictionDelay: Record<string, number> = {}
let galleryDirs: Record<string, CandidateFile[]> = {}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const parsed = new URL(url, 'http://localhost')
      const params = parsed.searchParams

      if (url.includes('/api/models')) {
        return jsonResponse([
          modelEntry(MODEL_A, 'LaTa (T5)'),
          // No q-q matrix for this one, so it serves no reviewer directories.
          modelEntry(MODEL_B, 'mT5-base', false),
        ])
      }
      if (url.includes('/token_map')) {
        return jsonResponse({ error: { message: 'no artifact' } }, 404)
      }
      if (url.includes('/candidate_dir/')) {
        const dir = decodeURIComponent(
          parsed.pathname.split('/candidate_dir/')[1].split('/')[0],
        )
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
          predictions: (candidates[model] ?? []).map((candidate) => ({
            rank: candidate.rank,
            dir_name: candidate.dir,
            score: 0.9,
            dir_files: candidate.files.map((f) => f.filename),
            preview_text: 'preview',
            candidate_files: candidate.files,
            source: candidate.source,
            label: candidate.label,
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
        data-testid="open-labelled-example"
        onClick={() => navigateToExample(MODEL_A, QUERY_ID, LABELLED_GALLERY_DIR)}
      >
        open a labelled example
      </button>
      <button
        type="button"
        data-testid="open-reviewer-example"
        onClick={() => navigateToExample(MODEL_A, QUERY_ID, REVIEWER_GALLERY_DIR)}
      >
        open a reviewer-directory example
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
        data-testid="choose-reviewer-rank"
        onClick={() => setActivePredictionRank(11)}
      >
        the reviewer directory
      </button>
      <button
        type="button"
        data-testid="choose-rank-1"
        onClick={() => setActivePredictionRank(1)}
      >
        the top candidate
      </button>
      <CenterArea />
    </>
  )
}

function renderCenter() {
  return render(
    <AppProvider>
      <TokenProvider>
        {/* The query header's directory badge reads the durable
            saved-directory record (issue #161), which has no fallback
            store: mounting the provider is how App composes them. */}
        <SavedDirectoryProvider accountKey="center-provenance">
          <PredictionProvider>
            <Harness />
          </PredictionProvider>
        </SavedDirectoryProvider>
      </TokenProvider>
    </AppProvider>,
  )
}

const candidateCaption = () =>
  screen.queryByTestId('document-provenance-candidate')?.textContent ?? null

const queryCaption = () =>
  screen.queryByTestId('document-provenance-query')?.textContent ?? null

beforeEach(() => {
  candidates = {
    [MODEL_A]: [
      {
        rank: 1,
        dir: 'LATA.DIR',
        source: 'model',
        files: [{ filename: 'a.txt', text: 'LATAWORD' }],
      },
      // Reviewer directories are anchored at MAX_MODEL_RANK + 1, so rank 11 is
      // what the backend really serves rather than "the next free slot".
      {
        rank: 11,
        dir: 'reviewer-dir-3',
        source: 'reviewer',
        label: 'Abigail: possible Ansegisus excerpt',
        files: [{ filename: 'r.txt', text: 'REVIEWERWORD' }],
      },
    ],
    [MODEL_B]: [
      {
        rank: 1,
        dir: 'MT5.DIR',
        source: 'model',
        files: [{ filename: 'b.txt', text: 'MT5WORD' }],
      },
    ],
  }
  predictionDelay = {}
  galleryDirs = {
    [LABELLED_GALLERY_DIR]: [{ filename: 'g.txt', text: 'GALLERYWORD' }],
    [REVIEWER_GALLERY_DIR]: [{ filename: 'h.txt', text: 'REVIEWERGALLERYWORD' }],
  }
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the mounted panels say what they are holding (issue #162)', () => {
  it('captions the left panel as the unlabeled query witness', async () => {
    renderCenter()
    expect(await screen.findByText(`QUERYWORD${QUERY_ID}`)).toBeTruthy()

    expect(queryCaption()).toBe(QUERY_CAPTION)
    // The query is never a candidate, so it carries no rank and no source
    // language of any kind.
    expect(queryCaption()).not.toContain('Rank')
    expect(screen.queryByText('Predicted Source')).toBeNull()
  })

  it('captions a model candidate as a labelled reference, with its rank', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    expect(candidateCaption()).toBe(`${LABELED_CAPTION} · Rank 1`)
  })

  it('captions a reviewer-directory candidate as an originally unlabeled member', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    fireEvent.click(screen.getByTestId('choose-reviewer-rank'))
    expect(await screen.findByText('REVIEWERWORD')).toBeTruthy()

    // The anchored rank, not a position in the list.
    expect(candidateCaption()).toBe(`${REVIEWER_CAPTION} · Rank 11`)
    // A provisional grouping is never described as a corpus label.
    expect(candidateCaption()).not.toContain(LABELED_CAPTION)
  })

  it('moves the caption with the selected candidate, both ways', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    fireEvent.click(screen.getByTestId('choose-reviewer-rank'))
    expect(await screen.findByText('REVIEWERWORD')).toBeTruthy()
    expect(candidateCaption()).toContain(REVIEWER_CAPTION)

    fireEvent.click(screen.getByTestId('choose-rank-1'))
    expect(await screen.findByText('LATAWORD')).toBeTruthy()
    await waitFor(() =>
      expect(candidateCaption()).toBe(`${LABELED_CAPTION} · Rank 1`),
    )
  })
})

describe('a gallery example is captioned from its own identity (issue #162)', () => {
  it('lets the override outrank the selected prediction', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    // A labelled example from the gallery, with the reviewer directory at rank
    // 11 selected underneath it afterwards. The keyboard rank shortcuts move
    // the selection without releasing an open example (PredictionList), so this
    // pairing is reachable: the example is what is on screen, so the example is
    // what the caption is about.
    fireEvent.click(screen.getByTestId('open-labelled-example'))
    expect(await screen.findByText('GALLERYWORD')).toBeTruthy()

    fireEvent.click(screen.getByTestId('choose-reviewer-rank'))
    await act(() => sleep(80))

    expect(screen.queryByText('REVIEWERWORD')).toBeNull()
    expect(candidateCaption()).toBe(LABELED_CAPTION)
    expect(candidateCaption()).not.toContain(REVIEWER_CAPTION)
    // An override carries no rank, so the caption must not borrow rank 11 from
    // the selection it is covering.
    expect(candidateCaption()).not.toContain('Rank')
  })

  it('reads a reviewer-directory example off the backend’s own id rule', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    // No prediction stands behind a gallery example, so the only evidence is
    // the directory id. `reviewer-dir-` is the same partition the backend
    // resolves the candidate-files route by.
    fireEvent.click(screen.getByTestId('open-reviewer-example'))
    expect(await screen.findByText('REVIEWERGALLERYWORD')).toBeTruthy()

    expect(candidateCaption()).toBe(REVIEWER_CAPTION)
  })

  it('gives the panel back to the ranked candidate on exit', async () => {
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()

    // A reviewer-directory example, opened over a labelled ranked candidate.
    fireEvent.click(screen.getByTestId('open-reviewer-example'))
    expect(await screen.findByText('REVIEWERGALLERYWORD')).toBeTruthy()
    expect(candidateCaption()).toBe(REVIEWER_CAPTION)

    fireEvent.click(screen.getByText('Exit'))

    // Opening an example resets the selection to rank 1 (AppContext), so the
    // panel comes back to the top ranked candidate -- a labelled reference,
    // carrying its rank again. If the override still captioned the panel this
    // would read "Reviewer group member" with no rank.
    expect(await screen.findByText('LATAWORD')).toBeTruthy()
    await waitFor(() =>
      expect(candidateCaption()).toBe(`${LABELED_CAPTION} · Rank 1`),
    )
  })
})

describe('a superseded caption is never shown as current (issue #162)', () => {
  it('drops the previous model’s provenance in the same commit as the switch', async () => {
    // MODEL_A's selected answer is a reviewer grouping. MODEL_B has no q-q
    // matrix in this fixture, so it serves no reviewer directories at all --
    // the documented `supports_reviewer_dirs: false` case. Leaving the first
    // caption up would describe the new model's panel with the old model's
    // kind, and then keep describing a candidate that does not exist here.
    predictionDelay[MODEL_B] = 120
    renderCenter()
    expect(await screen.findByText('LATAWORD')).toBeTruthy()
    fireEvent.click(screen.getByTestId('choose-reviewer-rank'))
    expect(await screen.findByText('REVIEWERWORD')).toBeTruthy()
    expect(candidateCaption()).toContain(REVIEWER_CAPTION)

    fireEvent.change(screen.getByLabelText('Select model'), {
      target: { value: MODEL_B },
    })

    // Same commit: no awaiting, so no exit animation has had a chance to run.
    expect(candidateCaption()).toBeNull()

    await act(() => sleep(200))
    // The ranking arrived and has nothing at rank 11, so there is still no
    // candidate and still nothing to caption.
    expect(candidateCaption()).toBeNull()

    fireEvent.click(screen.getByTestId('choose-rank-1'))
    expect(await screen.findByText('MT5WORD')).toBeTruthy()
    expect(candidateCaption()).toBe(`${LABELED_CAPTION} · Rank 1`)
  })

  it('drops the previous document’s provenance in the same commit as the navigation', async () => {
    predictionDelay[MODEL_A] = 120
    renderCenter()
    await act(() => sleep(200))
    expect(await screen.findByText('LATAWORD')).toBeTruthy()
    fireEvent.click(screen.getByTestId('choose-reviewer-rank'))
    expect(await screen.findByText('REVIEWERWORD')).toBeTruthy()

    fireEvent.click(screen.getByTestId('go-to-other-query'))

    expect(candidateCaption()).toBeNull()
    // The left panel is still a query witness, because that is the one fact
    // navigation cannot change.
    await waitFor(() =>
      expect(screen.queryByText(`QUERYWORD${OTHER_QUERY_ID}`)).toBeTruthy(),
    )
    expect(queryCaption()).toBe(QUERY_CAPTION)
  })

  it('does not caption a candidate the reviewer cannot see', async () => {
    // No candidates at all: the panel has nothing in it, and a caption would be
    // a claim about a document that is not on screen.
    candidates = { ...candidates, [MODEL_A]: [] }
    renderCenter()
    expect(await screen.findByText(`QUERYWORD${QUERY_ID}`)).toBeTruthy()
    await act(() => sleep(80))

    expect(candidateCaption()).toBeNull()
    expect(queryCaption()).toBe(QUERY_CAPTION)
  })
})
