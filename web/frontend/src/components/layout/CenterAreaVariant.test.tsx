import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { TokenProvider } from '../../contexts/TokenContext'
import VariantSelector from '../predictions/VariantSelector'
import CenterArea from './CenterArea'

const MODEL = 'bowphs_LaTa'
const QUERY_ID = 7

const RAW_DIR = 'CANT.328.12'
const CORRECTED_DIR = 'CSAR.347.17'
const DIRS = [RAW_DIR, CORRECTED_DIR]
const VARIANTS = ['baseline', 'abtt', 'sif', 'sif_abtt']

// One query token per (candidate_dir, attribution variant) pairing, so the
// highlighted token names exactly which matrix is on screen.
const QUERY_TOKENS = [
  ...DIRS.flatMap((d) => VARIANTS.map((v) => `${d}__${v}`)),
  // The variant-independent cosine matrix the payload always carries. If one
  // of these lights up, the view fell back off the attribution matrix.
  ...DIRS.map((d) => `${d}__cosine`),
]
function hotIndex(dir: string, variant: string): number {
  return QUERY_TOKENS.indexOf(`${dir}__${variant}`)
}
function cosineMatrix(dir: string): number[][] {
  const hot = QUERY_TOKENS.indexOf(`${dir}__cosine`)
  return QUERY_TOKENS.map((_, qi) => (qi === hot ? [1, 1] : [0, 0]))
}

function dirForVariant(variant: string): string {
  return variant === 'raw' ? RAW_DIR : CORRECTED_DIR
}

let requested: string[] = []
/** `dir:variant` pairings the artifact index does not know about (404). */
let missingArtifacts = new Set<string>()
/** Variants the artifact loads for, but holds no pair_matrices for. */
let variantsWithoutMatrices = new Set<string>()
/** Model an artifact from before attribution existed (no available_methods). */
let preAttribution = false
let tokenMapDelay = new Map<string, number>()
let predictionDelay = 0

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const sleep = (ms: number) =>
  ms > 0 ? new Promise((r) => setTimeout(r, ms)) : Promise.resolve()

function installFetch(): void {
  requested = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    requested.push(url)
    const parsed = new URL(url, 'http://localhost')
    const params = parsed.searchParams

    if (url.includes('/api/models')) {
      return jsonResponse([
        {
          slug: MODEL,
          display_name: 'LaTa (T5)',
          layer: 4,
          pooling: 'mean',
          prediction_count: 2238,
          available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
          default_variant: 'sif_abtt',
        },
      ])
    }

    if (url.includes('/token_map')) {
      const dir = params.get('candidate_dir') ?? ''
      const variant = params.get('variant') ?? ''
      const pair = `${dir}:${variant}`
      await sleep(tokenMapDelay.get(pair) ?? 0)
      if (missingArtifacts.has(pair)) {
        return jsonResponse(
          { error: { message: `Example not found: ${QUERY_ID}/${dir}` } },
          404,
        )
      }
      const hot = hotIndex(dir, variant)
      return jsonResponse({
        example_id: dir === RAW_DIR ? 121 : 122,
        model: MODEL,
        layer: 4,
        D: 10,
        bucket: 'demo',
        query_path: 'q.txt',
        candidate_path: `${dir}/c.txt`,
        query_tokens: [],
        candidate_tokens: [],
        similarity_matrix: cosineMatrix(dir),
        ig_weighted_matrix: null,
        top_matches: {},
        query_ig_baseline: [],
        query_ig_abtt: [],
        candidate_ig_baseline: [],
        candidate_ig_abtt: [],
        auto_highlights: null,
        available_variants: VARIANTS,
        // The real wire shape. `load_token_map` only creates the method key
        // when it actually adds a matrix, so "artifact carries attribution but
        // not this variant" serialises as `{}` -- identical to "artifact has
        // no attribution at all". `available_methods` is what separates them.
        available_methods: preAttribution ? [] : ['ig'],
        pair_matrices:
          preAttribution || variantsWithoutMatrices.has(variant)
            ? {}
            : {
                ig: {
                  [variant]: QUERY_TOKENS.map((_, qi) =>
                    qi === hot ? [1, 1] : [0, 0],
                  ),
                },
              },
      })
    }

    if (url.includes('/predictions')) {
      const variant = params.get('variant') ?? ''
      await sleep(predictionDelay)
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
        model: MODEL,
        variant,
        predictions: [
          {
            rank: 1,
            dir_name: dirForVariant(variant),
            score: 0.9,
            dir_files: ['c.txt'],
            preview_text: 'preview',
            candidate_files: [{ filename: 'c.txt', text: 'cand one' }],
          },
        ],
      })
    }

    if (/\/api\/query\/\d+$/.test(parsed.pathname)) {
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
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
  })
  vi.stubGlobal('fetch', fetchMock)
}

function tokenMapPairs(): string[] {
  return requested
    .filter((url) => url.includes('/token_map'))
    .map((url) => {
      const p = new URL(url, 'http://localhost').searchParams
      return `${p.get('candidate_dir')}:${p.get('variant')}`
    })
}

/** The (candidate_dir, variant) pairing whose matrix is currently rendered. */
function renderedPairing(): string | null {
  for (const text of QUERY_TOKENS) {
    if (screen.queryByLabelText(`query token ${text}, highlighted evidence`)) {
      return text.replace('__', ':')
    }
  }
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
      <VariantSelector />
      <CenterArea />
    </>
  )
}

function renderApp() {
  return render(
    <AppProvider>
      <TokenProvider>
        <Harness />
      </TokenProvider>
    </AppProvider>,
  )
}

async function click(name: RegExp) {
  await userEvent.click(screen.getByRole('radio', { name }))
}

beforeEach(() => {
  missingArtifacts = new Set()
  variantsWithoutMatrices = new Set()
  preAttribution = false
  tokenMapDelay = new Map()
  predictionDelay = 0
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const settle = () => new Promise((r) => setTimeout(r, 80))

/**
 * A note on what these pin, since the names oversell some of them.
 *
 * Three are true regression tests, each failing on the pre-fix code and each
 * isolating one defect: `never requests a (candidate, variant) pairing...`
 * (the prediction-lag guard), the two `useTokenMap` tests in
 * ../../api/tokenMap.test.ts (stale data across a key change), and
 * `says so instead of falling back...` (the cosine fall-through).
 *
 * The rest -- including `follows the reported sequence`, named after the bug
 * report -- pass on the pre-fix code too, because the mock always returns the
 * requested variant's matrix and so cannot exhibit the "frozen" symptom. They
 * are guards against re-breaking the happy path, not proof of the fix.
 */
describe('CenterArea variant switching (issue #73)', () => {
  it('follows the reported sequence sif_abtt -> raw -> abtt', async () => {
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )

    await click(/Raw/)
    await waitFor(() => expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`))

    await click(/^ABTT/)
    await waitFor(() => expect(renderedPairing()).toBe(`${CORRECTED_DIR}:abtt`))

    // Every switch changed what is on screen, and nothing that was on screen
    // ever belonged to a variant other than the selected one.
    expect(renderedPairing()).not.toBe(`${RAW_DIR}:baseline`)
  })

  it('never requests a (candidate, variant) pairing the reviewer did not select', async () => {
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    await click(/Raw/)
    await waitFor(() => expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`))
    await click(/^ABTT/)
    await waitFor(() => expect(renderedPairing()).toBe(`${CORRECTED_DIR}:abtt`))
    await click(/SIF\+ABTT/)
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )

    // Raw ranks a different directory first than the corrected variants do
    // (exactly the shape of the four demo queries), so a request that pairs
    // RAW_DIR with a corrected variant -- or CORRECTED_DIR with baseline --
    // can only have come from predictions that lag the selector.
    const legal = new Set([
      `${RAW_DIR}:baseline`,
      `${CORRECTED_DIR}:abtt`,
      `${CORRECTED_DIR}:sif`,
      `${CORRECTED_DIR}:sif_abtt`,
    ])
    expect(tokenMapPairs().filter((p) => !legal.has(p))).toEqual([])
  })

  it('re-selecting a variant after Raw still repaints (predictions cached)', async () => {
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    await click(/Raw/)
    await waitFor(() => expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`))
    await click(/SIF\+ABTT/)
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    // Second lap: both predictions and token maps now come from their caches.
    await click(/Raw/)
    await waitFor(() => expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`))
    await click(/^SIF —/)
    await waitFor(() => expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif`))
  })

  it('drops a superseded token map when its fetch lands after a newer one', async () => {
    predictionDelay = 5
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    // Raw's own map is slow, so anything else in flight resolves first.
    tokenMapDelay.set(`${RAW_DIR}:baseline`, 60)
    await click(/Raw/)
    await new Promise((r) => setTimeout(r, 200))
    expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`)
  })

  it('explains the 404, rather than blanking, when the pair has no artifact', async () => {
    // This is the reachable gap state on current main: #90's resolver filters
    // on `variants_available` and returns None when no artifact carries the
    // requested variant, so the reviewer gets a 404 -- not a payload missing a
    // matrix. Before, that rendered an unexplained empty evidence panel.
    missingArtifacts.add(`${CORRECTED_DIR}:abtt`)
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    await click(/Raw/)
    await waitFor(() => expect(renderedPairing()).toBe(`${RAW_DIR}:baseline`))
    await click(/^ABTT/)
    await settle()
    expect(renderedPairing()).toBe(null)
    expect(screen.getByRole('status').textContent).toMatch(
      /No attribution available for this pair under the ABTT variant/,
    )
  })

  it('keeps the cosine map, and stays quiet, for a pre-attribution artifact', async () => {
    // No `available_methods` means the artifact predates attribution, and its
    // cosine grid is the whole story rather than a stand-in for a missing
    // variant. Note it serialises `pair_matrices: {}` exactly like the
    // variant-missing case above, which is why that is not the test.
    preAttribution = true
    renderApp()
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:cosine`),
    )
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('says so instead of falling back to the variant-independent cosine map', async () => {
    // The artifact loads, but carries no matrix for the selected variant.
    // Falling through to `similarity_matrix` would render highlights that are
    // identical for every variant: the "nothing changes" symptom.
    function OverrideHarness() {
      const { navigateToExample } = useApp()
      useEffect(() => {
        navigateToExample(MODEL, QUERY_ID, CORRECTED_DIR)
      }, [navigateToExample])
      return (
        <>
          <VariantSelector />
          <CenterArea />
        </>
      )
    }
    variantsWithoutMatrices.add('abtt')
    render(
      <AppProvider>
        <TokenProvider>
          <OverrideHarness />
        </TokenProvider>
      </AppProvider>,
    )
    await waitFor(() =>
      expect(renderedPairing()).toBe(`${CORRECTED_DIR}:sif_abtt`),
    )
    await click(/^ABTT/)
    await settle()
    expect(renderedPairing()).toBe(null)
    expect(screen.getByRole('status').textContent).toMatch(
      /No IG attribution for the ABTT variant/,
    )
  })
})
