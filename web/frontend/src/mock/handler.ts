import type { QueryListItem, QueryListResponse, QueryDetail } from '../api/queries'
import type { StatsResponse, ModelInfo } from '../api/models'
import type { TokenMapResponse } from '../api/tokenMap'
import { MOCK_QUERIES, MOCK_QUERY_DETAILS } from './queries'
import { MOCK_PREDICTIONS } from './predictions'
import { MOCK_TOKEN_MAPS } from './tokenMaps'

// ---------------------------------------------------------------------------
// Synthetic query generation (expand 3 real queries to 2238 items)
// ---------------------------------------------------------------------------

const FILENAME_PREFIXES = [
  'BAV1341',
  'Hat42',
  'Par12048',
  'Vat5845',
  'Mon6245',
  'Bamb131',
  'Sang671',
  'Koeln213',
  'Laon201',
  'Wuerzb146',
]

const FOLIO_SUFFIXES = ['r', 'v']

function generateSyntheticQueries(): QueryListItem[] {
  const items: QueryListItem[] = [...MOCK_QUERIES]

  // Seeded random for deterministic filenames
  let seed = 7919
  function rand(): number {
    seed = (seed * 16807 + 0) % 2147483647
    return seed / 2147483647
  }

  const PREVIEWS = [
    'Si quis episcopus aut presbiter contra institutionem domini aliquid aliud in sacrifi',
    'De his qui ad clerum promoveri debent, ut nullus neophitus ordinetur episcopus.',
    'Placuit ut quotienscumque concilium congregandum est, episcopi qui neque aegritudine',
    'Si quis laicus uxorem suam dimiserit et alteram duxerit, vel eam quae ab alio dimis',
    'Ut nullus episcoporum alterius parrochianum iudicare praesumat nisi rogatus.',
    'De clericis qui ab uno loco ad alterum transeunt sine litteris commendaticiis.',
    'Si quis presbiter ordinatus deprehenderit se non rite ordinatum, denuo ordinetur.',
    'Omnes qui fideles sunt debent abstinere se a spectaculis et a ludis gentilium.',
    'Si quis clericus inventus fuerit ieiunans die dominica vel sabbato praeter unum sab',
    'Lectores et cantores et ostiarii et exorcistae possint ingredi nuptias celebrantes.',
  ]

  for (let fileId = 4; fileId <= 2238; fileId++) {
    const prefix = FILENAME_PREFIXES[Math.floor(rand() * FILENAME_PREFIXES.length)]
    const folio = Math.floor(rand() * 200) + 1
    const side = FOLIO_SUFFIXES[Math.floor(rand() * 2)]
    const fragment = Math.floor(rand() * 5) + 1
    const filename = `${prefix}.${folio}${side}.${fragment}.txt`
    const isReviewed = rand() < 0.11 // ~247 reviewed out of 2238
    const preview = PREVIEWS[Math.floor(rand() * PREVIEWS.length)]

    items.push({
      file_id: fileId,
      filename,
      text_preview: preview,
      review_status: isReviewed ? 'reviewed' : 'unreviewed',
      review_count: isReviewed ? Math.floor(rand() * 3) + 1 : 0,
    })
  }

  return items
}

const ALL_QUERIES = generateSyntheticQueries()

// ---------------------------------------------------------------------------
// Mock response helper
// ---------------------------------------------------------------------------

function mockResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

// ---------------------------------------------------------------------------
// Handler functions
// ---------------------------------------------------------------------------

function handleQueryList(url: string): QueryListResponse {
  const params = new URL(url, 'http://localhost').searchParams
  const status = params.get('status') || ''
  const search = (params.get('search') || '').toLowerCase()
  const page = parseInt(params.get('page') || '1', 10)
  const pageSize = parseInt(params.get('page_size') || '20', 10)

  let filtered = ALL_QUERIES
  if (status) {
    filtered = filtered.filter((q) => q.review_status === status)
  }
  if (search) {
    filtered = filtered.filter(
      (q) =>
        q.filename.toLowerCase().includes(search) ||
        q.text_preview.toLowerCase().includes(search),
    )
  }

  const total = filtered.length
  const start = (page - 1) * pageSize
  const items = filtered.slice(start, start + pageSize)

  return {
    items,
    total,
    page,
    page_size: pageSize,
    has_more: start + pageSize < total,
  }
}

function handleQueryDetail(id: number): QueryDetail | { error: { message: string } } {
  const detail = MOCK_QUERY_DETAILS.get(id)
  if (detail) return detail

  // Return a synthetic detail for generated queries
  const item = ALL_QUERIES.find((q) => q.file_id === id)
  if (!item) return { error: { message: `Query ${id} not found` } }

  const text = item.text_preview + ' ... [fragment continues]'
  return {
    file_id: item.file_id,
    filename: item.filename,
    text,
    tokens: text.split(/\s+/).map((t, i) => ({
      text: t,
      index: i,
      category: /^[.,;:!?()]+$/.test(t) ? 'punctuation' : 'content',
    })),
    char_count: text.length,
    token_count: text.split(/\s+/).length,
  }
}

function handlePredictions(
  queryId: number,
  _model: string,
): ReturnType<typeof MOCK_PREDICTIONS.get> | { error: { message: string } } {
  const pred = MOCK_PREDICTIONS.get(queryId)
  if (pred) return pred

  // Return a minimal prediction set for synthetic queries
  const item = ALL_QUERIES.find((q) => q.file_id === queryId)
  if (!item) return { error: { message: `Query ${queryId} not found` } }

  return {
    file_id: queryId,
    filename: item.filename,
    model: 'bowphs_LaTa',
    predictions: [
      {
        rank: 1,
        dir_name: 'Can.apost.42',
        score: 0.55,
        dir_files: ['Can.apost.42_a.txt'],
        preview_text: 'XLII. Si quis episcopus aut presbiter aut diaconus alea',
        candidate_files: [
          {
            filename: 'Can.apost.42_a.txt',
            text: 'XLII. Si quis episcopus aut presbiter aut diaconus aleae id est tabulae ludo se voluptuosius dederit, aut desinat aut certe damnetur.',
          },
        ],
      },
      {
        rank: 2,
        dir_name: 'Nic.325.c.5',
        score: 0.42,
        dir_files: ['Nic.325.c.5_a.txt'],
        preview_text: 'De his qui communione privantur, sive ex clero sive ex',
        candidate_files: [
          {
            filename: 'Nic.325.c.5_a.txt',
            text: 'De his qui communione privantur, sive ex clero sive ex laico ordine, ab episcopis per unamquamque provinciam sententia regularis obtineat.',
          },
        ],
      },
    ],
  }
}

function handleTokenMap(
  url: string,
): TokenMapResponse | { error: { message: string } } {
  // URL pattern: /api/query/{queryId}/token-map/{candidateId}
  const match = url.match(/\/api\/query\/(\d+)\/token-map\/([^?]+)/)
  if (!match) return { error: { message: 'Invalid token map URL' } }

  const queryId = match[1]
  const candidateId = decodeURIComponent(match[2])
  const key = `${queryId}-${candidateId}`

  const data = MOCK_TOKEN_MAPS.get(key)
  if (data) return data

  return { error: { message: `Token map not found for ${key}` } }
}

function handleStats(): StatsResponse {
  return {
    total_queries: 2238,
    reviewed_count: 247,
    unreviewed_count: 1991,
    feedback_count: 312,
    reviews_by_model: { 'bowphs/LaTa': 247 },
    reviews_by_reviewer: { scholar: 247 },
    rank_distribution: { '1': 89, '2': 67, '3': 45, '4': 28, '5': 18 },
    recent_reviews: [
      { file_id: 101, filename: 'BnF lat. 4886, f.12r', timestamp: new Date(Date.now() - 3600_000).toISOString(), model_slug: 'bowphs/LaTa' },
      { file_id: 203, filename: 'BnF lat. 5132, f.45v', timestamp: new Date(Date.now() - 7200_000).toISOString(), model_slug: 'bowphs/LaTa' },
      { file_id: 87, filename: 'BnF lat. 2819, f.3r', timestamp: new Date(Date.now() - 86400_000).toISOString(), model_slug: 'bowphs/LaTa' },
      { file_id: 512, filename: 'BnF lat. 7230, f.91r', timestamp: new Date(Date.now() - 172800_000).toISOString(), model_slug: 'bowphs/LaTa' },
      { file_id: 44, filename: 'BnF lat. 1118, f.22v', timestamp: new Date(Date.now() - 604800_000).toISOString(), model_slug: 'bowphs/LaTa' },
    ],
    next_unreviewed_ids: [248, 249, 250, 251, 252],
  }
}

function handleModels(): ModelInfo[] {
  return [
    {
      slug: 'bowphs_LaTa',
      display_name: 'LaTa (T5)',
      layer: 4,
      pooling: 'mean',
      prediction_count: 2238,
    },
  ]
}

// ---------------------------------------------------------------------------
// Fetch interceptor
// ---------------------------------------------------------------------------

const originalFetch = window.fetch

export function installMockHandler(): void {
  window.fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url

    // Only intercept /api/* calls
    if (!url.includes('/api/')) {
      return originalFetch(input, init)
    }

    // Add 200-400ms artificial delay
    await new Promise((r) => setTimeout(r, 200 + Math.random() * 200))

    // Route to mock handlers
    // Query list
    if (url.match(/\/api\/queries\?/) || url.match(/\/api\/queries$/)) {
      return mockResponse(handleQueryList(url))
    }

    // Predictions: /api/query/{id}/predictions?model={model}
    const predMatch = url.match(/\/api\/query\/(\d+)\/predictions\?model=(.+)/)
    if (predMatch) {
      const id = parseInt(predMatch[1], 10)
      const model = decodeURIComponent(predMatch[2])
      const result = handlePredictions(id, model)
      if (result && 'error' in result) {
        return mockResponse(result, 404)
      }
      return mockResponse(result)
    }

    // Token map: /api/query/{id}/token-map/{candidateId}
    if (url.match(/\/api\/query\/\d+\/token-map\//)) {
      const result = handleTokenMap(url)
      if ('error' in result) {
        return mockResponse(result, 404)
      }
      return mockResponse(result)
    }

    // Query detail: /api/query/{id}
    const detailMatch = url.match(/\/api\/query\/(\d+)$/)
    if (detailMatch) {
      const id = parseInt(detailMatch[1], 10)
      const result = handleQueryDetail(id)
      if ('error' in result) {
        return mockResponse(result, 404)
      }
      return mockResponse(result)
    }

    // Feedback
    if (url.includes('/api/feedback') && init?.method === 'POST') {
      return mockResponse({ success: true })
    }

    // Stats
    if (url.includes('/api/stats')) {
      return mockResponse(handleStats())
    }

    // Models
    if (url.includes('/api/models')) {
      return mockResponse(handleModels())
    }

    return new Response(JSON.stringify({ error: { message: 'Not found' } }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}
