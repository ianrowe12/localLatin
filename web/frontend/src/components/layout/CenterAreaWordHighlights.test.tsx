import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { PredictionProvider } from '../../contexts/PredictionContext'
import { SavedDirectoryProvider } from '../../contexts/SavedDirectoryContext'
import { TokenProvider } from '../../contexts/TokenContext'
import ModelSelector from '../predictions/ModelSelector'
import CenterArea from './CenterArea'

/**
 * Word-level highlighting in the reviewer's panels (issue #211).
 *
 * Prof. Firey's report was that the highlighting "disregards the distinctive
 * words". Measured, the attribution is not the problem: the display is. The
 * model reads `Episcopus` as `Epi` + `##scop` + `##us`, the grid is indexed by
 * piece, and the panels render the file's words -- so the highlight landed on
 * whichever word happened to share an index with the piece.
 *
 * The fixture below is that exact shape. The piece grid's hot row is piece 1
 * (`##scop`); the word grid's hot row is word 0 (`Episcopus`). Displayed token
 * 1 is `aut`. So "which word is outlined" separates the two readings cleanly.
 */

const MODEL = 'google_mt5-base'
const QUERY_ID = 7
const DIR = 'Can.apost.7'
const VARIANT = 'sif_abtt'

// What the query panel renders: the file's own words.
const QUERY_WORDS = ['Episcopus', 'aut', 'presbiter']
const CANDIDATE_TEXT = 'sacerdotes ministri'

// What the model read.
const QUERY_PIECES = ['Epi', '##scop', '##us', 'aut', 'pres', '##biter']
const CANDIDATE_PIECES = ['sacer', '##dotes', 'ministri']

const HOT_PIECE = QUERY_PIECES.indexOf('##scop') // 1
const HOT_WORD = 0 // Episcopus

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** A grid that is hot in exactly one row. */
function oneHotRows(rows: number, cols: number, hot: number): number[][] {
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, () => (i === hot ? 1 : 0)),
  )
}

let served: URL[] = []

function installFetch(): void {
  served = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const parsed = new URL(url, 'http://localhost')
    served.push(parsed)

    if (url.includes('/api/models')) {
      return jsonResponse([
        {
          slug: MODEL,
          display_name: 'mT5-base',
          layer: 6,
          pooling: 'mean',
          prediction_count: 2238,
          available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
          default_variant: 'sif_abtt',
        },
      ])
    }

    if (url.includes('/token_map')) {
      return jsonResponse({
        example_id: 121,
        model: MODEL,
        layer: 6,
        D: 10,
        bucket: 'unlabelled_bulk',
        query_path: 'q.txt',
        candidate_path: `${DIR}/c.txt`,
        query_tokens: QUERY_PIECES.map((text, idx) => ({
          idx,
          text,
          is_content: true,
        })),
        candidate_tokens: CANDIDATE_PIECES.map((text, idx) => ({
          idx,
          text,
          is_content: true,
        })),
        similarity_matrix: oneHotRows(
          QUERY_PIECES.length,
          CANDIDATE_PIECES.length,
          HOT_PIECE,
        ),
        ig_weighted_matrix: null,
        top_matches: {},
        query_ig_baseline: [],
        query_ig_abtt: [],
        candidate_ig_baseline: [],
        candidate_ig_abtt: [],
        auto_highlights: null,
        available_methods: ['ig'],
        available_variants: [VARIANT],
        variant_requested: VARIANT,
        variant_served: VARIANT,
        attribution_source: 'ig',
        pair_matrices: {
          ig: {
            [VARIANT]: oneHotRows(
              QUERY_PIECES.length,
              CANDIDATE_PIECES.length,
              HOT_PIECE,
            ),
          },
        },
        // The display half of the fix: words, the pieces behind them, and the
        // same grid aggregated to word x word.
        query_words: [
          { idx: 0, text: 'Episcopus', piece_indices: [0, 1, 2], score: 1, score_pos: 1, score_neg: 0, is_content: true },
          { idx: 1, text: 'aut', piece_indices: [3], score: 0, score_pos: 0, score_neg: 0, is_content: true },
          { idx: 2, text: 'presbiter', piece_indices: [4, 5], score: 0, score_pos: 0, score_neg: 0, is_content: true },
        ],
        candidate_words: [
          { idx: 0, text: 'sacerdotes', piece_indices: [0, 1], score: 1, score_pos: 1, score_neg: 0, is_content: true },
          { idx: 1, text: 'ministri', piece_indices: [2], score: 0, score_pos: 0, score_neg: 0, is_content: true },
        ],
        word_segmentation: 'text',
        word_aggregation: 'sum',
        word_similarity_matrix: oneHotRows(3, 2, HOT_WORD),
        word_pair_matrices: { ig: { [VARIANT]: oneHotRows(3, 2, HOT_WORD) } },
      })
    }

    if (url.includes('/predictions')) {
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
        model: parsed.searchParams.get('model') ?? '',
        variant: parsed.searchParams.get('variant') ?? '',
        predictions: [
          {
            rank: 1,
            dir_name: DIR,
            score: 0.9,
            dir_files: ['c.txt'],
            preview_text: CANDIDATE_TEXT,
            candidate_files: [{ filename: 'c.txt', text: CANDIDATE_TEXT }],
          },
        ],
      })
    }

    if (/\/api\/query\/\d+$/.test(parsed.pathname)) {
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
        text: QUERY_WORDS.join(' '),
        tokens: QUERY_WORDS.map((text, index) => ({
          text,
          index,
          category: 'content',
        })),
        char_count: 30,
        token_count: QUERY_WORDS.length,
      })
    }

    return jsonResponse({})
  })
  vi.stubGlobal('fetch', fetchMock)
}

function Harness() {
  const { setActiveQueryId } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
  }, [setActiveQueryId])
  return (
    <>
      {/* The picker is what resolves the served model list, exactly as the
          app composes it; without it nothing asks for a ranking. */}
      <ModelSelector />
      <CenterArea />
    </>
  )
}

function renderApp() {
  return render(
    <AppProvider>
      <TokenProvider>
        <SavedDirectoryProvider accountKey="center-word-highlights">
          <PredictionProvider>
            <Harness />
          </PredictionProvider>
        </SavedDirectoryProvider>
      </TokenProvider>
    </AppProvider>,
  )
}

/** Every query-side span that is currently carrying a highlight. */
function highlightedQueryWords(): string[] {
  return screen
    .queryAllByLabelText(/^query token .*, highlighted evidence$/)
    .map((el) => el.textContent?.trim() ?? '')
}

const togglePieces = () => screen.getByLabelText('Show pieces')

beforeEach(() => {
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('word-level highlighting (issue #211)', () => {
  it('outlines the whole word a fragmented highlight belongs to', async () => {
    renderApp()

    await waitFor(() => expect(highlightedQueryWords()).toEqual(['Episcopus']))

    // One span, not three: the reviewer never sees `Epi` or `##scop`.
    expect(screen.getByText('Episcopus')).toBeTruthy()
    expect(screen.queryByText('##scop')).toBeNull()
    // And not the word that merely shares an index with the hot piece.
    expect(highlightedQueryWords()).not.toContain('aut')
  })

  it('shows the model pieces when the toggle is on, and words again when it is off', async () => {
    renderApp()
    await waitFor(() => expect(highlightedQueryWords()).toEqual(['Episcopus']))

    await userEvent.click(togglePieces())

    await waitFor(() => expect(screen.getByText('##scop')).toBeTruthy())
    expect(screen.getByText('Epi')).toBeTruthy()
    // The piece grid's own hot row, now over the pieces it actually indexes.
    expect(highlightedQueryWords()).toEqual(['##scop'])
    expect(screen.queryByText('Episcopus')).toBeNull()

    await userEvent.click(togglePieces())

    await waitFor(() => expect(highlightedQueryWords()).toEqual(['Episcopus']))
    expect(screen.queryByText('##scop')).toBeNull()
  })

  it('starts in the word view and says which unit is highlighted', async () => {
    renderApp()
    await waitFor(() =>
      expect((togglePieces() as HTMLInputElement).checked).toBe(false),
    )

    expect(screen.getByText('Highlighting whole words')).toBeTruthy()

    await userEvent.click(togglePieces())

    expect((togglePieces() as HTMLInputElement).checked).toBe(true)
    expect(screen.getByText(/Highlighting the model.s subword pieces/)).toBeTruthy()
  })

  it('keeps the toggle reachable from the keyboard', async () => {
    renderApp()
    await waitFor(() => expect(highlightedQueryWords()).toEqual(['Episcopus']))

    togglePieces().focus()
    expect(document.activeElement).toBe(togglePieces())
    await userEvent.keyboard(' ')

    await waitFor(() =>
      expect((togglePieces() as HTMLInputElement).checked).toBe(true),
    )
  })

  it('asks for the attribution of the variant on screen', async () => {
    renderApp()
    await waitFor(() => expect(highlightedQueryWords()).toEqual(['Episcopus']))

    const tokenMapRequests = served.filter((u) => u.pathname.includes('token_map'))
    expect(tokenMapRequests.length).toBeGreaterThan(0)
    for (const request of tokenMapRequests) {
      expect(request.searchParams.get('variant')).toBe(VARIANT)
    }
  })
})
