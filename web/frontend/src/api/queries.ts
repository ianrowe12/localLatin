import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, toApiErrorInfo, type ApiErrorInfo } from './client'
import { FEEDBACK_UPDATED_EVENT } from './feedback'
import { REVIEWER_DIRS_UPDATED_EVENT, type ReviewerDir } from './reviewerDirs'
import { DEFAULT_VARIANT, type PredictionVariant } from './variants'

// ---------------------------------------------------------------------------
// Types matching backend models.py
// ---------------------------------------------------------------------------

export interface QueryListItem {
  file_id: number
  filename: string
  text_preview: string
  review_status: 'unreviewed' | 'reviewed' | 'skipped'
  review_count: number
}

export interface QueryListResponse {
  items: QueryListItem[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export interface NextQueryResponse {
  file_id: number | null
}

export interface TokenInfo {
  text: string
  index: number
  category: string
}

export interface QueryDetail {
  file_id: number
  filename: string
  text: string
  tokens: TokenInfo[]
  char_count: number
  token_count: number
}

export interface CandidateFile {
  filename: string
  text: string
}

/**
 * Where a candidate came from. `model` is a labelled directory the retrieval
 * run ranked; `reviewer` is a directory a reviewer created, scored live from
 * the query-query matrix. Reviewer candidates are appended after the model's
 * ten, so a model candidate's rank still means what it always did.
 */
export type CandidateSource = 'model' | 'reviewer'

export interface Prediction {
  rank: number
  dir_name: string
  score: number
  dir_files: string[]
  preview_text: string
  candidate_files: CandidateFile[] | null
  source?: CandidateSource
  // Reviewer candidates only.
  label?: string | null
  created_by?: string | null
  seed_query_id?: number | null
}

export interface PredictionResponse {
  file_id: number
  filename: string
  model: string
  variant: PredictionVariant
  predictions: Prediction[]
  /**
   * The retrieval run's own status for this (model, variant, query) row, copied
   * verbatim from the predictions CSV: `ok`, `excluded_blank_source`,
   * `excluded_zero_norm`, or absent on artifacts written before the column
   * existed.
   *
   * An excluded query keeps its CSV row with every rank cell blank, so
   * `predictions: []` alone cannot say whether a ranking is missing on purpose
   * or missing for an unknown reason. Absent/null means unknown -- never an
   * exclusion, and never a reason to reinterpret a ranking that is present.
   */
  status?: string | null
  // Reviewer directories seeded by this query. Drives the badge on the
  // document itself rather than on any candidate card.
  seeded_dirs?: ReviewerDir[]
}

// ---------------------------------------------------------------------------
// Hook state shape
// ---------------------------------------------------------------------------

interface HookState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

// ---------------------------------------------------------------------------
// useQueryList
// ---------------------------------------------------------------------------

export function useQueryList(
  status: string,
  search: string,
  page: number,
  pageSize = 50,
): HookState<QueryListResponse> {
  const [state, setState] = useState<HookState<QueryListResponse>>({
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, QueryListResponse>())
  const listKeyRef = useRef<string | null>(null)
  const [refreshVersion, setRefreshVersion] = useState(0)

  useEffect(() => {
    const refresh = () => {
      cache.current.clear()
      setRefreshVersion((version) => version + 1)
    }
    window.addEventListener(FEEDBACK_UPDATED_EVENT, refresh)
    return () => window.removeEventListener(FEEDBACK_UPDATED_EVENT, refresh)
  }, [])

  useEffect(() => {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (search) params.set('search', search)
    params.set('page', String(page))
    params.set('page_size', String(pageSize))
    const key = params.toString()
    const listKey = `${status}|${search}|${pageSize}`
    const sameList = listKeyRef.current === listKey
    listKeyRef.current = listKey

    const applyPage = (data: QueryListResponse) => {
      setState((prev) => {
        if (page > 1 && sameList && prev.data) {
          const seen = new Set(prev.data.items.map((item) => item.file_id))
          const mergedItems = [
            ...prev.data.items,
            ...data.items.filter((item) => !seen.has(item.file_id)),
          ]
          return {
            data: { ...data, items: mergedItems },
            loading: false,
            error: null,
          }
        }
        return { data, loading: false, error: null }
      })
    }

    const cached = cache.current.get(key)
    if (cached) {
      applyPage(cached)
      return
    }

    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))

    apiFetch<QueryListResponse>(`/api/queries?${key}`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(key, data)
        applyPage(data)
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState((prev) => ({ ...prev, loading: false, error: err.message }))
      })

    return () => {
      cancelled = true
    }
  }, [status, search, page, pageSize, refreshVersion])

  return state
}

export async function fetchNextQuery(after?: number): Promise<NextQueryResponse> {
  const params = new URLSearchParams()
  if (after !== undefined) params.set('after', String(after))
  const query = params.toString()
  return apiFetch<NextQueryResponse>(`/api/queries/next${query ? `?${query}` : ''}`)
}

// ---------------------------------------------------------------------------
// useQueryDetail
// ---------------------------------------------------------------------------

export function useQueryDetail(id: number | null): HookState<QueryDetail> {
  const [state, setState] = useState<HookState<QueryDetail> & { id: number | null }>({
    id: null,
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<number, QueryDetail>())

  useEffect(() => {
    if (id === null) {
      setState({ id: null, data: null, loading: false, error: null })
      return
    }

    const cached = cache.current.get(id)
    if (cached) {
      setState({ id, data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState({ id, data: null, loading: true, error: null })

    apiFetch<QueryDetail>(`/api/query/${id}`)
      .then((data) => {
        if (cancelled) return
        // The response echoes the file it is for. A body for another document
        // is not this document's text, whatever the URL asked for.
        if (data?.file_id !== id) {
          setState({
            id,
            data: null,
            loading: false,
            error: 'The server sent text for a different document.',
          })
          return
        }
        cache.current.set(id, data)
        setState({ id, data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState({ id, data: null, loading: false, error: err.message })
      })

    return () => {
      cancelled = true
    }
  }, [id])

  // Render-time key guard, not an effect (issue #156). `id` changes during
  // render; the effect above only catches up one commit later. Without this,
  // the paint in between shows the PREVIOUS document's text -- and the ranking
  // beside it is already the new document's, so a reviewer can judge one
  // fragment's candidates against another fragment's words.
  if (state.id !== id) {
    return { data: null, loading: id !== null, error: null }
  }
  return { data: state.data, loading: state.loading, error: state.error }
}

// ---------------------------------------------------------------------------
// useCandidateDirFiles
// ---------------------------------------------------------------------------

export function useCandidateDirFiles(
  candidateDir: string | null,
): HookState<CandidateFile[]> {
  const [state, setState] = useState<
    HookState<CandidateFile[]> & { dir: string | null }
  >({
    dir: null,
    data: null,
    loading: false,
    error: null,
  })
  const cache = useRef(new Map<string, CandidateFile[]>())

  useEffect(() => {
    if (candidateDir === null) {
      setState({ dir: null, data: null, loading: false, error: null })
      return
    }

    const cached = cache.current.get(candidateDir)
    if (cached) {
      setState({ dir: candidateDir, data: cached, loading: false, error: null })
      return
    }

    let cancelled = false
    setState({ dir: candidateDir, data: null, loading: true, error: null })

    const encoded = encodeURIComponent(candidateDir)
    apiFetch<CandidateFile[]>(`/api/candidate_dir/${encoded}/files`)
      .then((data) => {
        if (cancelled) return
        cache.current.set(candidateDir, data)
        setState({ dir: candidateDir, data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (cancelled) return
        setState({ dir: candidateDir, data: null, loading: false, error: err.message })
      })

    return () => {
      cancelled = true
    }
  }, [candidateDir])

  // Same render-time guard as useQueryDetail: the response body carries no
  // directory name of its own, so the key is the only thing that binds these
  // files to the directory whose name is on the panel above them.
  if (state.dir !== candidateDir) {
    return { data: null, loading: candidateDir !== null, error: null }
  }
  return { data: state.data, loading: state.loading, error: state.error }
}

// ---------------------------------------------------------------------------
// usePredictions
// ---------------------------------------------------------------------------

/**
 * What the app currently knows about the ranking for one query/model/variant.
 *
 * Six states, because the five failures below used to reach the reviewer as one
 * sentence -- "No predictions available" -- and each of them calls for a
 * different next action (issue #156).
 *
 * - `idle`: nothing is selected yet. Not a failure.
 * - `loading`: a request for THIS key is outstanding. Includes the first render
 *   after the key changed and every refresh, because old-key data is not
 *   current evidence.
 * - `error`: the request failed, or the response was not the shape this app can
 *   read. `error` says which.
 * - `empty`: a valid response with no model candidates and no exclusion
 *   recorded. Unexplained, and it must stay unexplained.
 * - `excluded`: a valid response with no model candidates and an explicit
 *   `status` from the retrieval run. Non-evaluable for a stated reason.
 * - `ready`: a validated response with at least one model candidate.
 *
 * Reviewer-created directories deliberately cannot produce `ready`: they are
 * scored live from the q-q matrix and say nothing about whether the model
 * produced a ranking. A colleague's new directory is not model evidence.
 */
export type PredictionPhase =
  | 'idle'
  | 'loading'
  | 'error'
  | 'empty'
  | 'excluded'
  | 'ready'

/** Which guard in the retrieval run dropped this query, when one did. */
export type ExclusionReason = 'blank_source' | 'zero_norm' | 'other'

export interface PredictionRequestKey {
  queryId: number
  model: string
  variant: PredictionVariant
}

export function samePredictionKey(
  a: PredictionRequestKey | null,
  b: PredictionRequestKey | null,
): boolean {
  if (a === null || b === null) return a === b
  return a.queryId === b.queryId && a.model === b.model && a.variant === b.variant
}

export function predictionKeyString(key: PredictionRequestKey): string {
  return `${key.queryId}:${key.model}:${key.variant}`
}

/**
 * `excluded_*` is the retrieval run's vocabulary, and only it can put a query in
 * one of these states. An unknown `excluded_…` value is still an exclusion but
 * is not given one of the two known explanations; anything else -- including
 * `ok`, an empty cell and a missing field -- is not an exclusion at all.
 */
export function classifyExclusion(
  status: string | null | undefined,
): ExclusionReason | null {
  if (typeof status !== 'string') return null
  const value = status.trim()
  if (value === 'excluded_blank_source') return 'blank_source'
  if (value === 'excluded_zero_norm') return 'zero_norm'
  if (value.startsWith('excluded')) return 'other'
  return null
}

/** The model's own candidates. `source` defaults to `model` on legacy rows. */
export function modelCandidates(predictions: Prediction[]): Prediction[] {
  return predictions.filter(
    (prediction) => (prediction.source ?? 'model') === 'model',
  )
}

/** Reviewer-created directories merged in by the API. */
export function reviewerCandidates(predictions: Prediction[]): Prediction[] {
  return predictions.filter((prediction) => prediction.source === 'reviewer')
}

/**
 * Whether a candidate carries text a reviewer can actually read.
 *
 * `web/routers/predictions.py` fills `candidate_files` with `texts.get(fname, "")`,
 * so a candidate can arrive with filenames and no words behind them. Exposed
 * rather than papered over: the honest answer is "this candidate's evidence is
 * missing", never a manufactured excerpt.
 */
export function candidateHasText(prediction: Prediction): boolean {
  return (prediction.candidate_files ?? []).some(
    (file) => typeof file?.text === 'string' && file.text.trim().length > 0,
  )
}

/**
 * Whether ONE witness carries readable words.
 *
 * Deliberately separate from `candidateHasText`, which answers the
 * whole-directory question. The review view shows a single file, so "does this
 * directory have text somewhere" is the wrong test for explaining a blank pane:
 * a blank first witness beside a readable second one produced an unexplained
 * empty panel. Callers that need the directory-wide meaning still have it.
 */
export function fileHasText(file: CandidateFile | null | undefined): boolean {
  return typeof file?.text === 'string' && file.text.trim().length > 0
}

export type PredictionValidation =
  | { ok: true; value: PredictionResponse }
  | { ok: false; reason: string }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function validateCandidateFiles(value: unknown): CandidateFile[] | null | false {
  if (value === null || value === undefined) return null
  if (!Array.isArray(value)) return false
  const files: CandidateFile[] = []
  for (const entry of value) {
    if (!isRecord(entry)) return false
    if (typeof entry.filename !== 'string') return false
    if (typeof entry.text !== 'string') return false
    files.push({ filename: entry.filename, text: entry.text })
  }
  return files
}

/**
 * One seeded reviewer directory, checked rather than cast.
 *
 * `seeded_dirs` used to be waved through as `ReviewerDir[]` once the container
 * was known to be an array. Its entries are not inert data: `AwaitingMatchBadge`
 * reads `.status` on every one of them the moment the ranking arrives, so a
 * single null entry took down the review view with a TypeError -- the loudest
 * possible version of the silent failure this issue is about.
 *
 * `status` is checked against the closed set the badge understands. An
 * unrecognised status is treated as malformed rather than filtered out: the
 * badge is the ONLY thing that reports what became of a directory the reviewer
 * created, and quietly rendering no badge for a state this build cannot name
 * would be the same unexplained absence in a new place. Both sides of the wire
 * ship together (web/ is a subtree), so a third status means the frontend has
 * to be taught what it means.
 *
 * Returns the normalised directory, or a reason string naming the entry.
 */
function validateReviewerDir(value: unknown, index: number): ReviewerDir | string {
  const at = `seeded_dirs[${index}]`
  if (!isRecord(value)) return `${at} is not an object`
  if (typeof value.dir_id !== 'string' || value.dir_id.length === 0) {
    return `${at}.dir_id is missing`
  }
  if (typeof value.label !== 'string') return `${at}.label is not a string`
  if (value.status !== 'awaiting_match' && value.status !== 'matched') {
    return `${at}.status is not a known status`
  }
  if (typeof value.seed_query_id !== 'number' || !Number.isInteger(value.seed_query_id)) {
    return `${at}.seed_query_id is not an integer`
  }
  const members = value.member_query_ids
  if (members !== undefined && !Array.isArray(members)) {
    return `${at}.member_query_ids is malformed`
  }
  if (
    Array.isArray(members) &&
    members.some((id) => typeof id !== 'number' || !Number.isInteger(id))
  ) {
    return `${at}.member_query_ids is malformed`
  }
  // The badge renders this number with .toFixed(2); a string would throw there
  // and a NaN would print "NaN" beside a reviewer's own directory.
  const score = value.best_match_score
  if (score !== undefined && score !== null) {
    if (typeof score !== 'number' || !Number.isFinite(score)) {
      return `${at}.best_match_score is not a number`
    }
  }
  const potential = value.has_potential_match
  if (potential !== undefined && typeof potential !== 'boolean') {
    return `${at}.has_potential_match is not a boolean`
  }
  for (const field of ['created_at', 'created_by', 'model_slug'] as const) {
    if (value[field] !== undefined && typeof value[field] !== 'string') {
      return `${at}.${field} is not a string`
    }
  }
  if (value.variant !== undefined && value.variant !== null && typeof value.variant !== 'string') {
    return `${at}.variant is not a string`
  }
  return {
    dir_id: value.dir_id,
    label: value.label,
    status: value.status,
    seed_query_id: value.seed_query_id,
    member_query_ids: (members as number[] | undefined) ?? [],
    created_at: typeof value.created_at === 'string' ? value.created_at : '',
    created_by: typeof value.created_by === 'string' ? value.created_by : '',
    model_slug: typeof value.model_slug === 'string' ? value.model_slug : '',
    variant: (value.variant as ReviewerDir['variant'] | undefined) ?? null,
    best_match_score: typeof score === 'number' ? score : null,
    has_potential_match: potential === true,
  }
}

/**
 * Runtime validation of a prediction response (issue #156).
 *
 * `apiFetch<PredictionResponse>` is a cast, not a check: TypeScript generics
 * disappear at runtime, so a 200 carrying the wrong document's id, a duplicated
 * rank or a NaN score used to flow straight into the ranked list and into a
 * feedback payload. Anything that fails here is a malformed response, not a
 * ranking with a few odd entries: silently dropping a candidate would renumber
 * what the reviewer sees against what the server resolves a rank to.
 *
 * Sparse ranks are legal and preserved. Reviewer directories are anchored at
 * rank 11 whatever the model returned, so [1, 11] is a real ranking, and
 * requiring 1..n or exactly ten would reject it.
 */
export function validatePredictionResponse(
  payload: unknown,
  key: PredictionRequestKey,
): PredictionValidation {
  if (!isRecord(payload)) return { ok: false, reason: 'response is not an object' }
  if (payload.file_id !== key.queryId) return { ok: false, reason: 'file_id mismatch' }
  if (payload.model !== key.model) return { ok: false, reason: 'model mismatch' }
  if (payload.variant !== key.variant) return { ok: false, reason: 'variant mismatch' }
  if (typeof payload.filename !== 'string') {
    return { ok: false, reason: 'filename missing' }
  }
  const status = payload.status
  if (status !== undefined && status !== null && typeof status !== 'string') {
    return { ok: false, reason: 'status is not a string' }
  }
  if (!Array.isArray(payload.predictions)) {
    return { ok: false, reason: 'predictions is not an array' }
  }
  if (payload.seeded_dirs !== undefined && !Array.isArray(payload.seeded_dirs)) {
    return { ok: false, reason: 'seeded_dirs is not an array' }
  }

  const seededDirs: ReviewerDir[] = []
  const rawSeeded = (payload.seeded_dirs as unknown[] | undefined) ?? []
  for (let i = 0; i < rawSeeded.length; i++) {
    const dir = validateReviewerDir(rawSeeded[i], i)
    if (typeof dir === 'string') return { ok: false, reason: dir }
    seededDirs.push(dir)
  }

  const seen = new Set<number>()
  const predictions: Prediction[] = []
  for (const entry of payload.predictions) {
    if (!isRecord(entry)) return { ok: false, reason: 'candidate is not an object' }
    const rank = entry.rank
    if (typeof rank !== 'number' || !Number.isInteger(rank) || rank < 1) {
      return { ok: false, reason: 'candidate rank is not a positive integer' }
    }
    if (seen.has(rank)) return { ok: false, reason: 'duplicate candidate rank' }
    seen.add(rank)
    if (typeof entry.dir_name !== 'string' || entry.dir_name.length === 0) {
      return { ok: false, reason: 'candidate has no directory' }
    }
    if (typeof entry.score !== 'number' || !Number.isFinite(entry.score)) {
      return { ok: false, reason: 'candidate score is not finite' }
    }
    const source = entry.source
    if (source !== undefined && source !== 'model' && source !== 'reviewer') {
      return { ok: false, reason: 'candidate source is unknown' }
    }
    const dirFiles = entry.dir_files
    if (dirFiles !== undefined && !Array.isArray(dirFiles)) {
      return { ok: false, reason: 'candidate dir_files is not an array' }
    }
    if (Array.isArray(dirFiles) && dirFiles.some((f) => typeof f !== 'string')) {
      return { ok: false, reason: 'candidate dir_files is not a string list' }
    }
    const candidateFiles = validateCandidateFiles(entry.candidate_files)
    if (candidateFiles === false) {
      return { ok: false, reason: 'candidate_files is malformed' }
    }
    predictions.push({
      rank,
      dir_name: entry.dir_name,
      score: entry.score,
      dir_files: (dirFiles as string[] | undefined) ?? [],
      preview_text: typeof entry.preview_text === 'string' ? entry.preview_text : '',
      candidate_files: candidateFiles,
      // Explicit, so every consumer reads the same source for a legacy row.
      source: (source as CandidateSource | undefined) ?? 'model',
      label: typeof entry.label === 'string' ? entry.label : null,
      created_by: typeof entry.created_by === 'string' ? entry.created_by : null,
      seed_query_id:
        typeof entry.seed_query_id === 'number' ? entry.seed_query_id : null,
    })
  }

  return {
    ok: true,
    value: {
      file_id: key.queryId,
      filename: payload.filename,
      model: key.model,
      variant: key.variant,
      predictions,
      status: typeof status === 'string' ? status : null,
      seeded_dirs: seededDirs,
    },
  }
}

export interface PredictionResult {
  /** The query/model/variant this result speaks for. Null when nothing is selected. */
  key: PredictionRequestKey | null
  /**
   * Monotonic id of the request whose result is on screen. 0 while nothing has
   * settled for the current key, including during a refresh.
   *
   * This is a CLIENT request counter. It is not a server ranking revision, and
   * it is not evidence that a human looked at anything.
   */
  generation: number
  phase: PredictionPhase
  /** Validated body, for `ready`, `empty` and `excluded` only. */
  response: PredictionResponse | null
  error: ApiErrorInfo | null
  /** The retrieval run's raw status string, unmapped. Null when unknown. */
  sourceStatus: string | null
  exclusion: ExclusionReason | null
  /** Refetch this key, dropping the cache. Invalidates current evidence at once. */
  refresh: () => void
}

interface PredictionSnapshot {
  key: PredictionRequestKey
  token: number
  generation: number
  phase: PredictionPhase
  response: PredictionResponse | null
  error: ApiErrorInfo | null
}

const MALFORMED_PREDICTIONS: ApiErrorInfo = {
  kind: 'malformed',
  status: 200,
  code: null,
  message: 'The server sent a ranking this app could not read.',
}

function settledPhase(response: PredictionResponse): PredictionPhase {
  if (modelCandidates(response.predictions).length > 0) return 'ready'
  return classifyExclusion(response.status) !== null ? 'excluded' : 'empty'
}

/**
 * The one prediction request for a query/model/variant.
 *
 * Mount it once, behind `PredictionProvider`, rather than per consumer: three
 * independent copies of this hook meant three caches, three requests and three
 * separate answers to "what is the current ranking", so the list could show a
 * failure while the assessment panel offered ranks from a different response.
 */
export function usePredictions(
  queryId: number | null,
  model: string,
  variant: PredictionVariant = DEFAULT_VARIANT,
): PredictionResult {
  const key = useMemo<PredictionRequestKey | null>(
    () => (queryId === null || !model ? null : { queryId, model, variant }),
    [queryId, model, variant],
  )
  const [snapshot, setSnapshot] = useState<PredictionSnapshot | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)
  const cache = useRef(new Map<string, PredictionResponse>())
  // Issued-request counter, and the id of the newest one. A response whose
  // generation is no longer the newest is superseded and is dropped even if its
  // effect's cleanup has not run.
  const generationRef = useRef(0)
  const latestRef = useRef(0)

  const refresh = useCallback(() => {
    cache.current.clear()
    setRefreshToken((token) => token + 1)
  }, [])

  // A new reviewer directory becomes a candidate for every *other* query, so
  // the whole cache is stale, not just the seed's entry.
  useEffect(() => {
    const onUpdate = () => refresh()
    window.addEventListener(REVIEWER_DIRS_UPDATED_EVENT, onUpdate)
    return () => window.removeEventListener(REVIEWER_DIRS_UPDATED_EVENT, onUpdate)
  }, [refresh])

  useEffect(() => {
    if (key === null) {
      latestRef.current = ++generationRef.current
      setSnapshot(null)
      return
    }

    const generation = ++generationRef.current
    latestRef.current = generation
    const cacheKey = predictionKeyString(key)

    const settle = (response: PredictionResponse) => {
      setSnapshot({
        key,
        token: refreshToken,
        generation,
        phase: settledPhase(response),
        response,
        error: null,
      })
    }

    const cached = cache.current.get(cacheKey)
    if (cached) {
      settle(cached)
      return
    }

    let cancelled = false

    apiFetch<unknown>(
      `/api/query/${key.queryId}/predictions?model=${encodeURIComponent(
        key.model,
      )}&variant=${key.variant}`,
    )
      .then((payload) => {
        if (cancelled || latestRef.current !== generation) return
        const validated = validatePredictionResponse(payload, key)
        if (!validated.ok) {
          setSnapshot({
            key,
            token: refreshToken,
            generation,
            phase: 'error',
            response: null,
            error: MALFORMED_PREDICTIONS,
          })
          return
        }
        cache.current.set(cacheKey, validated.value)
        settle(validated.value)
      })
      .catch((err: unknown) => {
        if (cancelled || latestRef.current !== generation) return
        setSnapshot({
          key,
          token: refreshToken,
          generation,
          phase: 'error',
          response: null,
          error: toApiErrorInfo(err, 'Could not load the ranking.'),
        })
      })

    return () => {
      cancelled = true
    }
  }, [key, refreshToken])

  return useMemo<PredictionResult>(() => {
    const base = {
      generation: 0,
      response: null,
      error: null,
      sourceStatus: null,
      exclusion: null,
      refresh,
    }
    if (key === null) return { ...base, key: null, phase: 'idle' }
    // Render-time guard, deliberately not an effect. `key` changes during
    // render and the fetching effect only catches up one commit later, so
    // without this the paint in between hands every consumer the PREVIOUS
    // selection's ranking under the new selection's name. The same comparison
    // makes a refresh invalidate the evidence immediately: until the request
    // for THIS token settles, there is no current ranking.
    if (
      snapshot === null ||
      !samePredictionKey(snapshot.key, key) ||
      snapshot.token !== refreshToken
    ) {
      return { ...base, key, phase: 'loading' }
    }
    return {
      key,
      generation: snapshot.generation,
      phase: snapshot.phase,
      response: snapshot.response,
      error: snapshot.error,
      sourceStatus: snapshot.response?.status ?? null,
      exclusion: classifyExclusion(snapshot.response?.status),
      refresh,
    }
  }, [key, snapshot, refreshToken, refresh])
}
