import type {
  NextQueryResponse,
  QueryDetail,
  QueryListItem,
  QueryListResponse,
} from '../api/queries'
import type { StatsResponse, ModelInfo } from '../api/models'
import {
  DEFAULT_VARIANT,
  PREDICTION_VARIANTS,
  type PredictionVariant,
} from '../api/variants'
import type { AuthUser } from '../api/auth'
import { FALLBACK_BANDS } from '../utils/confidenceBands'
import type { ReviewerDir } from '../api/reviewerDirs'
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
let mockUser: AuthUser | null = null
let pendingAccounts: AuthUser[] = []
const approvedAccounts: AuthUser[] = [
  {
    id: 1,
    username: 'pi',
    display_name: 'PI Scholar',
    role: 'pi_admin',
    approval_status: 'approved',
    must_change_password: false,
  },
  {
    id: 2,
    username: 'reviewer',
    display_name: 'Reviewer',
    role: 'reviewer',
    approval_status: 'approved',
    must_change_password: false,
  },
]

// ---------------------------------------------------------------------------
// Mock response helper
// ---------------------------------------------------------------------------

function mockResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

async function handleAuth(url: string, init?: RequestInit): Promise<Response | null> {
  if (url.includes('/api/auth/me')) {
    return mockUser
      ? mockResponse(mockUser)
      : mockResponse({ error: { message: 'Not authenticated' } }, 401)
  }

  if (url.includes('/api/auth/change_password')) {
    if (mockUser) mockUser = { ...mockUser, must_change_password: false }
    return mockUser
      ? mockResponse(mockUser)
      : mockResponse({ error: { message: 'Not authenticated' } }, 401)
  }

  if (url.includes('/api/auth/signout')) {
    mockUser = null
    return mockResponse({ success: true })
  }

  if (url.includes('/api/auth/accounts')) {
    if (url.includes('/reset_password')) {
      const accountId = Number(url.match(/accounts\/(\d+)\/reset_password/)?.[1])
      const account = approvedAccounts.find((item) => item.id === accountId)
      return mockResponse({
        account: {
          ...(account ?? approvedAccounts[0]),
          must_change_password: true,
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          last_login_at: null,
          approved_at: new Date().toISOString(),
          approved_by_account_id: 1,
          rejected_at: null,
          approval_note: '',
        },
        temporary_password: 'temporary-password-456',
      })
    }
    if (url.includes('/approve')) {
      const accountId = Number(url.match(/accounts\/(\d+)\/approve/)?.[1])
      const account = pendingAccounts.find((item) => item.id === accountId)
      pendingAccounts = pendingAccounts.filter((item) => item.id !== accountId)
      return mockResponse({
        ...account,
        approval_status: 'approved',
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_login_at: null,
        approved_at: new Date().toISOString(),
        approved_by_account_id: 1,
        rejected_at: null,
        approval_note: '',
      })
    }
    if (url.includes('/reject')) {
      const accountId = Number(url.match(/accounts\/(\d+)\/reject/)?.[1])
      const account = pendingAccounts.find((item) => item.id === accountId)
      pendingAccounts = pendingAccounts.filter((item) => item.id !== accountId)
      return mockResponse({
        ...account,
        approval_status: 'rejected',
        is_active: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_login_at: null,
        approved_at: null,
        approved_by_account_id: null,
        rejected_at: new Date().toISOString(),
        approval_note: '',
      })
    }
    if (init?.method === 'POST') {
      const body = init?.body ? JSON.parse(String(init.body)) : {}
      const account: AuthUser = {
        id: pendingAccounts.length + 10,
        username: body.username ?? 'new-reviewer',
        display_name: body.display_name ?? body.username ?? 'New Reviewer',
        role: body.role ?? 'reviewer',
        approval_status: 'approved',
        must_change_password: false,
      }
      return mockResponse(
        {
          account: {
            ...account,
            is_active: true,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            last_login_at: null,
            approved_at: new Date().toISOString(),
            approved_by_account_id: 1,
            rejected_at: null,
            approval_note: '',
          },
          temporary_password: body.password ? null : 'temporary-password-123',
        },
        201,
      )
    }
    const listed = url.includes('status=approved')
      ? approvedAccounts
      : pendingAccounts
    return mockResponse(
      listed.map((account) => ({
        ...account,
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_login_at: null,
        approved_at: null,
        approved_by_account_id: null,
        rejected_at: null,
        approval_note: '',
      })),
    )
  }

  if (url.includes('/api/auth/register')) {
    const body = init?.body ? JSON.parse(String(init.body)) : {}
    const account: AuthUser = {
      id: pendingAccounts.length + 2,
      username: body.username ?? 'reviewer',
      display_name: body.display_name ?? body.username ?? 'Reviewer',
      role: 'reviewer',
      approval_status: 'pending',
      must_change_password: false,
    }
    pendingAccounts = [account, ...pendingAccounts]
    return mockResponse(
      {
        status: 'pending_approval',
        message: 'Account request submitted. A PI/admin must approve it before sign-in.',
        account,
      },
      201,
    )
  }

  if (url.includes('/api/auth/signin')) {
    const body = init?.body ? JSON.parse(String(init.body)) : {}
    mockUser = {
      id: body.username === 'pi' ? 1 : 2,
      username: body.username ?? 'reviewer',
      display_name: body.username === 'pi' ? 'PI Scholar' : 'Reviewer',
      role: body.username === 'pi' ? 'pi_admin' : 'reviewer',
      approval_status: 'approved',
      must_change_password: false,
    }
    return mockResponse(mockUser)
  }

  return null
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

function handleNextQuery(url: string): NextQueryResponse {
  const params = new URL(url, 'http://localhost').searchParams
  const afterParam = params.get('after')
  const after = afterParam === null ? null : parseInt(afterParam, 10)
  const actionable = ALL_QUERIES
    .filter((q) => q.review_status === 'unreviewed')
    .map((q) => q.file_id)

  if (after !== null && Number.isFinite(after)) {
    const nextAfter = actionable.find((fileId) => fileId > after)
    if (nextAfter !== undefined) return { file_id: nextAfter }
  }

  return { file_id: actionable[0] ?? null }
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
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): ReturnType<typeof MOCK_PREDICTIONS.get> | { error: { message: string } } {
  const pred = MOCK_PREDICTIONS.get(queryId)
  // Echo back the requested model and variant, as the real API does. The
  // canned entries carry one hardcoded pair, and CenterArea drops a response
  // whose key does not match the current selection (issue #73), so an
  // unechoed model would blank the evidence panel under dev:mock.
  if (pred) return { ...pred, model: model || pred.model, variant }

  // Return a minimal prediction set for synthetic queries
  const item = ALL_QUERIES.find((q) => q.file_id === queryId)
  if (!item) return { error: { message: `Query ${queryId} not found` } }

  return {
    file_id: queryId,
    filename: item.filename,
    model: model || 'bowphs_LaTa',
    variant,
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
  // URL pattern: /api/query/{queryId}/token_map?candidate_dir=...&method=&variant=
  const match = url.match(/\/api\/query\/(\d+)\/token_map\?(.*)$/)
  if (!match) return { error: { message: 'Invalid token map URL' } }

  const queryId = match[1]
  const params = new URLSearchParams(match[2])
  const candidateId = params.get('candidate_dir') ?? ''
  const key = `${queryId}-${candidateId}`

  const data = MOCK_TOKEN_MAPS.get(key)
  if (data) return data

  return { error: { message: `Token map not found for ${key}` } }
}

function handleStats(): StatsResponse {
  return {
    total_queries: 2238,
    reviewed_count: 247,
    skipped_count: 12,
    unreviewed_count: 1979,
    unresolved_count: 0,
    feedback_count: 312,
    reviews_by_model: { 'bowphs/LaTa': 247 },
    reviews_by_reviewer: { scholar: 247 },
    rank_distribution: { '1': 89, '2': 67, '3': 45, '4': 28, '5': 18, skipped: 12 },
    outcome_distribution: { matched_rank: 247, skipped: 12 },
    recent_reviews: [
      { file_id: 101, filename: 'BnF lat. 4886, f.12r', timestamp: new Date(Date.now() - 3600_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'matched_rank', reviewer: 'scholar', correct_rank: 1 },
      { file_id: 203, filename: 'BnF lat. 5132, f.45v', timestamp: new Date(Date.now() - 7200_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'skipped', reviewer: 'scholar', correct_rank: null },
      { file_id: 87, filename: 'BnF lat. 2819, f.3r', timestamp: new Date(Date.now() - 86400_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'none_of_top_k', reviewer: 'scholar', correct_rank: 0 },
      { file_id: 512, filename: 'BnF lat. 7230, f.91r', timestamp: new Date(Date.now() - 172800_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'matched_rank', reviewer: 'scholar', correct_rank: 2 },
      { file_id: 44, filename: 'BnF lat. 1118, f.22v', timestamp: new Date(Date.now() - 604800_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'matched_rank', reviewer: 'scholar', correct_rank: 1 },
    ],
    needs_attention: [
      { file_id: 203, filename: 'BnF lat. 5132, f.45v', timestamp: new Date(Date.now() - 7200_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'skipped', reviewer: 'scholar', notes: 'Needs reference work' },
      { file_id: 87, filename: 'BnF lat. 2819, f.3r', timestamp: new Date(Date.now() - 86400_000).toISOString(), model_slug: 'bowphs/LaTa', outcome: 'none_of_top_k', reviewer: 'scholar', notes: 'No candidate is acceptable' },
    ],
    next_unreviewed_ids: [248, 249, 250, 251, 252],
  }
}

/**
 * Notes are shared across reviewers (issue #96), so mock mode has to be able to
 * show a note that belongs to somebody else. One seeded query carries one;
 * every other query is unreviewed, as before.
 *
 * Shaped like what the server actually returns to a reviewer who has not
 * answered this query yet: Abigail's note and attribution, and no decision --
 * the API merges the shared note with the caller's own answer, and there is no
 * own answer here.
 */
const MOCK_NOTED_QUERY_ID = 101

function handleLatestFeedback(url: string): unknown {
  const queryId = Number(new URL(url, 'http://mock').searchParams.get('query_id'))
  if (queryId !== MOCK_NOTED_QUERY_ID) return null
  return {
    id: 9001,
    query_id: MOCK_NOTED_QUERY_ID,
    timestamp: new Date(Date.now() - 3600_000).toISOString(),
    model_slug: 'bowphs_LaTa',
    variant: DEFAULT_VARIANT,
    outcome: 'legacy_unresolved',
    correct_rank: null,
    correct_dir: null,
    selected_ranks: null,
    notes: 'Rank 2 matches the chapter number; rank 1 is a later gloss.',
    reviewer: 'Abigail Scholar',
    reviewer_account_id: 2,
    reviewer_username: 'abigail',
    schema_version: 2,
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
      available_variants: PREDICTION_VARIANTS,
      default_variant: DEFAULT_VARIANT,
      confidence_bands: FALLBACK_BANDS,
      supports_reviewer_dirs: true,
    },
  ]
}

// ---------------------------------------------------------------------------
// Reviewer directories
// ---------------------------------------------------------------------------

// Mock mode has no q-q matrix, so a created directory is remembered and echoed
// back on the seed query. That exercises the creation flow and the badge; the
// merge into *other* queries' candidate lists needs real similarity data and is
// covered by the backend tests.
const mockReviewerDirs: ReviewerDir[] = []

function handleCreateReviewerDir(init?: RequestInit): ReviewerDir {
  const body = JSON.parse(String(init?.body ?? '{}'))
  const dir: ReviewerDir = {
    dir_id: `reviewer-dir-mock${mockReviewerDirs.length + 1}`,
    label: body.label || `New directory from query ${body.query_file_id}`,
    status: 'awaiting_match',
    seed_query_id: body.query_file_id,
    member_query_ids: [body.query_file_id],
    created_at: new Date().toISOString(),
    created_by: 'scholar',
    model_slug: body.model_slug ?? 'bowphs_LaTa',
    variant: body.variant ?? DEFAULT_VARIANT,
    best_match_score: 0.31,
    has_potential_match: false,
  }
  mockReviewerDirs.push(dir)
  return dir
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
    const authResponse = await handleAuth(url, init)
    if (authResponse) return authResponse

    // Next actionable query
    if (url.match(/\/api\/queries\/next/)) {
      return mockResponse(handleNextQuery(url))
    }

    // Query list
    if (url.match(/\/api\/queries\?/) || url.match(/\/api\/queries$/)) {
      return mockResponse(handleQueryList(url))
    }

    // Predictions: /api/query/{id}/predictions?model={model}&variant={variant}
    const predMatch = url.match(/\/api\/query\/(\d+)\/predictions\?(.+)/)
    if (predMatch) {
      const id = parseInt(predMatch[1], 10)
      const predParams = new URLSearchParams(predMatch[2])
      const model = predParams.get('model') ?? ''
      const variant = (predParams.get('variant') ?? DEFAULT_VARIANT) as PredictionVariant
      const result = handlePredictions(id, model, variant)
      if (result && 'error' in result) {
        return mockResponse(result, 404)
      }
      return mockResponse(result)
    }

    // Token map: /api/query/{id}/token_map?candidate_dir=...
    if (url.match(/\/api\/query\/\d+\/token_map\?/)) {
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
    if (url.includes('/api/feedback/latest')) {
      return mockResponse(handleLatestFeedback(url))
    }
    if (url.includes('/api/feedback') && init?.method === 'POST') {
      return mockResponse({ success: true })
    }

    // Stats
    if (url.includes('/api/stats')) {
      return mockResponse(handleStats())
    }

    // Reviewer directories
    if (url.includes('/api/reviewer_dirs')) {
      if (init?.method === 'POST') {
        return mockResponse(handleCreateReviewerDir(init), 201)
      }
      return mockResponse(mockReviewerDirs)
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
