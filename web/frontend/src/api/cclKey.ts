import { apiFetch } from './client'
import type { PredictionVariant } from './variants'

/**
 * The blue "None of the top N" action, with its optional CCL key (issue #196).
 *
 * A separate module from `api/feedback.ts` on purpose. This is one POST whose
 * body is fixed except for two fields, and it is the ONLY writer of a key; the
 * save path in `api/feedback.ts` belongs to the rank pills and is under
 * concurrent repair in issue #158/#172. Keeping them apart means neither change
 * has to wait for the other, and a reader can see the whole of this request in
 * one screen.
 *
 * It posts to the same endpoint, `/api/feedback`, because there is only one
 * assessment log and one set of rules about what may be written to it. The
 * server decides what the key means -- see `web/routers/feedback.py` and
 * `web/services/ccl_keys.py` -- and answers with what it did.
 */

/** What the server did with the key. Mirrors `CclKeyAction` in web/models.py. */
export type CclKeyAction =
  | 'matched_labelled_dir'
  | 'joined_reviewer_dir'
  | 'created_reviewer_dir'
  | 'seed_taken'

export interface NoneOfTopKReceipt {
  id: number
  query_id: number
  outcome: string
  /** The key as stored: what the reviewer typed, whitespace normalised. */
  ccl_key: string | null
  ccl_key_action: CclKeyAction | null
  /** Labelled directory name, or a reviewer `dir_id`. Null without a key. */
  ccl_key_dir: string | null
}

export interface NoneOfTopKInput {
  queryId: number
  model: string
  variant: PredictionVariant
  /** Empty means no key: record the non-match and nothing else. */
  cclKey: string
  notes: string
}

const ACTIONS: readonly string[] = [
  'matched_labelled_dir',
  'joined_reviewer_dir',
  'created_reviewer_dir',
  'seed_taken',
]

/**
 * A 201 this client cannot read is a FAILURE, not a quiet success.
 *
 * The receipt is what the reviewer is shown, and the four branches say four
 * different things -- one of them is "a permanent group now holds this
 * document". Displaying a branch nobody sent, or none at all, would be worse
 * than saying the outcome is unknown, which is what throwing here produces.
 */
export function assertReceipt(body: unknown): NoneOfTopKReceipt {
  const row = body as NoneOfTopKReceipt | null
  const readable =
    row !== null &&
    typeof row === 'object' &&
    typeof row.id === 'number' &&
    typeof row.query_id === 'number' &&
    row.outcome === 'none_of_top_k' &&
    (row.ccl_key === null || typeof row.ccl_key === 'string') &&
    (row.ccl_key_action === null || ACTIONS.includes(row.ccl_key_action)) &&
    (row.ccl_key_dir === null || typeof row.ccl_key_dir === 'string')
  if (!readable) {
    throw new Error(
      'The server answered, but not with a record this app can read. Your answer may or may not have been saved.',
    )
  }
  return row
}

export async function recordNoneOfTopK(
  input: NoneOfTopKInput,
): Promise<NoneOfTopKReceipt> {
  const key = input.cclKey.trim()
  const body = await apiFetch<unknown>('/api/feedback', {
    method: 'POST',
    body: JSON.stringify({
      query_id: input.queryId,
      model_slug: input.model,
      variant: input.variant,
      outcome: 'none_of_top_k',
      correct_rank: 0,
      // Never an assignment authority, exactly as on the rank path: the server
      // resolves the key against the corpus and its own tables.
      correct_dir: null,
      notes: input.notes,
      // Omitted entirely when blank, so a deployment that predates the field
      // sees the payload it has always seen.
      ...(key ? { ccl_key: key } : {}),
    }),
  })
  return assertReceipt(body)
}
