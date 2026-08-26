import { useEffect } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider, useApp } from '../../contexts/AppContext'
import { TokenProvider } from '../../contexts/TokenContext'
import AttributionMethodSelector from '../common/AttributionMethodSelector'
import ModelSelector from '../predictions/ModelSelector'
import CenterArea from './CenterArea'

// Formerly CenterAreaVariant.test.tsx (issue #73). Issue #94 removed the
// post-processing picker, so the switch these tests drive is now the one a
// PI-admin can still make -- the attribution *method* -- plus moving between
// candidates by changing model. Every behavioural assertion from #73 survives
// that translation: no stale matrix across a key change, no pairing the
// reviewer never selected, an explicit notice instead of a blank panel or a
// cosine fallback. Only the variant radios are gone.

const MODEL_A = 'bowphs_LaTa'
const MODEL_B = 'google_mt5-base'
const QUERY_ID = 7

// Each model ranks a different directory first, which is what makes a
// mismatched (candidate, model) pairing detectable at all.
const DIR_A = 'CANT.328.12'
const DIR_B = 'CSAR.347.17'
const DIRS = [DIR_A, DIR_B]
const METHODS = ['ig', 'bertscore']
/** The one pipeline the app serves, in the attribution vocabulary. */
const VARIANT = 'sif_abtt'

// One query token per (candidate_dir, method) pairing, so the highlighted
// token names exactly which matrix is on screen.
const QUERY_TOKENS = [
  ...DIRS.flatMap((d) => METHODS.map((m) => `${d}__${m}`)),
  // The method-independent cosine matrix the payload always carries. If one
  // of these lights up, the view fell back off the attribution matrix.
  ...DIRS.map((d) => `${d}__cosine`),
]
function hotIndex(dir: string, method: string): number {
  return QUERY_TOKENS.indexOf(`${dir}__${method}`)
}
function cosineMatrix(dir: string): number[][] {
  const hot = QUERY_TOKENS.indexOf(`${dir}__cosine`)
  return QUERY_TOKENS.map((_, qi) => (qi === hot ? [1, 1] : [0, 0]))
}

function dirForModel(model: string): string {
  return model === MODEL_A ? DIR_A : DIR_B
}

let requested: string[] = []
/** `dir:method` pairings the artifact index does not know about (404). */
let missingArtifacts = new Set<string>()
/** Methods the artifact loads for, but holds no pair_matrices for. */
let methodsWithoutMatrices = new Set<string>()
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

function modelEntry(slug: string, displayName: string) {
  return {
    slug,
    display_name: displayName,
    layer: 4,
    pooling: 'mean',
    prediction_count: 2238,
    available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
    default_variant: 'sif_abtt',
  }
}

function installFetch(): void {
  requested = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    requested.push(url)
    const parsed = new URL(url, 'http://localhost')
    const params = parsed.searchParams

    if (url.includes('/api/models')) {
      // Sorted by display name, as the real endpoint is.
      return jsonResponse([
        modelEntry(MODEL_A, 'LaTa (T5)'),
        modelEntry(MODEL_B, 'mT5-base'),
      ])
    }

    if (url.includes('/token_map')) {
      const dir = params.get('candidate_dir') ?? ''
      const method = params.get('method') ?? ''
      const pair = `${dir}:${method}`
      await sleep(tokenMapDelay.get(pair) ?? 0)
      if (missingArtifacts.has(pair)) {
        return jsonResponse(
          { error: { message: `Example not found: ${QUERY_ID}/${dir}` } },
          404,
        )
      }
      const hot = hotIndex(dir, method)
      return jsonResponse({
        example_id: dir === DIR_A ? 121 : 122,
        model: params.get('model') ?? '',
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
        available_variants: [VARIANT],
        // The real wire shape. `load_token_map` only creates the method key
        // when it actually adds a matrix, so "artifact carries attribution but
        // not this method" serialises as `{}` -- identical to "artifact has no
        // attribution at all". `available_methods` is what separates them.
        available_methods: preAttribution ? [] : METHODS,
        pair_matrices:
          preAttribution || methodsWithoutMatrices.has(method)
            ? {}
            : {
                [method]: {
                  [VARIANT]: QUERY_TOKENS.map((_, qi) =>
                    qi === hot ? [1, 1] : [0, 0],
                  ),
                },
              },
      })
    }

    if (url.includes('/predictions')) {
      const model = params.get('model') ?? ''
      const variant = params.get('variant') ?? ''
      await sleep(predictionDelay)
      return jsonResponse({
        file_id: QUERY_ID,
        filename: 'query-7.txt',
        model,
        variant,
        predictions: [
          {
            rank: 1,
            dir_name: dirForModel(model),
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

/** `candidate_dir:model:method` for every token-map request made so far. */
function tokenMapKeys(): string[] {
  return requested
    .filter((url) => url.includes('/token_map'))
    .map((url) => {
      const p = new URL(url, 'http://localhost').searchParams
      return `${p.get('candidate_dir')}:${p.get('model')}:${p.get('method')}`
    })
}

/** Every variant a token-map request asked for. */
function tokenMapVariants(): string[] {
  return requested
    .filter((url) => url.includes('/token_map'))
    .map((url) => new URL(url, 'http://localhost').searchParams.get('variant') ?? '')
}

/** The (candidate_dir, method) pairing whose matrix is currently rendered. */
function renderedPairing(): string | null {
  for (const text of QUERY_TOKENS) {
    if (screen.queryByLabelText(`query token ${text}, highlighted evidence`)) {
      return text.replace('__', ':')
    }
  }
  return null
}

function Harness() {
  const { setActiveQueryId } = useApp()
  useEffect(() => {
    setActiveQueryId(QUERY_ID)
  }, [setActiveQueryId])
  return (
    <>
      {/* The two controls that still change what evidence is on screen. */}
      <ModelSelector />
      <AttributionMethodSelector />
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

/** Switch model through the real picker. */
async function selectModel(slug: string) {
  await userEvent.selectOptions(screen.getByLabelText('Select model'), slug)
}

// The picker's accessible name is the method's tooltip title, not its short
// label (AttributionMethodSelector's METHODS table).
const METHOD_TITLE: Record<string, RegExp> = {
  ig: /Integrated gradients/,
  bertscore: /Token similarity score/,
}

/** Switch attribution method through the real PI-admin picker. */
async function selectMethod(method: 'ig' | 'bertscore') {
  await userEvent.click(screen.getByRole('radio', { name: METHOD_TITLE[method] }))
}

beforeEach(() => {
  missingArtifacts = new Set()
  methodsWithoutMatrices = new Set()
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
 * Two are true regression tests, each isolating one defect: `never requests a
 * (candidate, model) pairing...` (the prediction-lag guard, which fails
 * without the key check in CenterArea) and `says so instead of falling back...`
 * (the cosine fall-through). The stale-data-across-a-key-change defect is
 * pinned by the two unit tests in ../../api/tokenMap.test.ts.
 *
 * The rest pass on the pre-fix code too, because the mock always returns the
 * requested key's matrix and so cannot exhibit the "frozen" symptom. They are
 * guards against re-breaking the happy path, not proof of the fix.
 */
describe('CenterArea evidence switching (issues #73, #94)', () => {
  it('follows a model switch mT5-base -> LaTa -> mT5-base', async () => {
    renderApp()
    // The reviewer lands on mT5-base (issue #94), which ranks DIR_B first.
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))

    await selectModel(MODEL_A)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:ig`))

    await selectModel(MODEL_B)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
  })

  it('follows a method switch IG -> BERTScore on the same candidate', async () => {
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))

    await selectMethod('bertscore')
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:bertscore`))

    await selectMethod('ig')
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
  })

  it('never requests a (candidate, model) pairing the reviewer did not select', async () => {
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
    await selectModel(MODEL_A)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:ig`))
    await selectMethod('bertscore')
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:bertscore`))
    await selectModel(MODEL_B)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:bertscore`))

    // The two models rank different directories first, so a request pairing
    // DIR_A with MODEL_B -- or DIR_B with MODEL_A -- can only have come from
    // predictions that lag the selector by a commit.
    const legal = new Set([
      `${DIR_A}:${MODEL_A}:ig`,
      `${DIR_A}:${MODEL_A}:bertscore`,
      `${DIR_B}:${MODEL_B}:ig`,
      `${DIR_B}:${MODEL_B}:bertscore`,
    ])
    expect(tokenMapKeys().filter((key) => !legal.has(key))).toEqual([])
    // And with one pipeline, every request speaks of that pipeline only.
    expect(new Set(tokenMapVariants())).toEqual(new Set([VARIANT]))
  })

  it('re-selecting a model still repaints (predictions cached)', async () => {
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
    await selectModel(MODEL_A)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:ig`))
    await selectModel(MODEL_B)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
    // Second lap: both predictions and token maps now come from their caches.
    await selectModel(MODEL_A)
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:ig`))
    await selectMethod('bertscore')
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_A}:bertscore`))
  })

  it('drops a superseded token map when its fetch lands after a newer one', async () => {
    predictionDelay = 5
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))
    // LaTa's own map is slow, so anything else in flight resolves first.
    tokenMapDelay.set(`${DIR_A}:ig`, 60)
    await selectModel(MODEL_A)
    await new Promise((r) => setTimeout(r, 200))
    expect(renderedPairing()).toBe(`${DIR_A}:ig`)
  })

  it('explains the 404, rather than blanking, when the pair has no artifact', async () => {
    // The reachable gap state: #90's resolver returns None when no artifact
    // covers the pair, so the reviewer gets a 404 rather than a payload
    // missing a matrix. Before #92 that rendered an unexplained empty panel.
    missingArtifacts.add(`${DIR_A}:ig`)
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))

    await selectModel(MODEL_A)
    await settle()

    expect(renderedPairing()).toBe(null)
    expect(screen.getByRole('status').textContent).toMatch(
      /No attribution available for this pair/,
    )
  })

  it('keeps the cosine map, and stays quiet, for a pre-attribution artifact', async () => {
    // No `available_methods` means the artifact predates attribution, and its
    // cosine grid is the whole story rather than a stand-in for a missing
    // matrix. Note it serialises `pair_matrices: {}` exactly like the
    // method-missing case below, which is why that is not the test.
    preAttribution = true
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:cosine`))
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('says so instead of falling back to the method-independent cosine map', async () => {
    // The artifact loads and advertises the method, but carries no matrix for
    // it. Falling through to `similarity_matrix` would render highlights that
    // are identical for every method: the "nothing changes" symptom of #73.
    methodsWithoutMatrices.add('bertscore')
    renderApp()
    await waitFor(() => expect(renderedPairing()).toBe(`${DIR_B}:ig`))

    await selectMethod('bertscore')
    await settle()

    expect(renderedPairing()).toBe(null)
    expect(screen.getByRole('status').textContent).toMatch(
      /No BERTScore attribution for this pair/,
    )
  })
})
