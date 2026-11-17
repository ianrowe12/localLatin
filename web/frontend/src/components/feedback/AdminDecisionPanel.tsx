import { useEffect, useRef, useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import { useReviewer } from '../../contexts/ReviewerContext'
import {
  fetchAdminReviewItem, saveDecidingAnswer, REVIEW_STATUS_LABELS,
  type AdminReviewItem,
} from '../../api/adminReview'
import { CCL_KEY_SUBJECT, unsafeTextError } from '../../utils/inputSafety'
import AdminReviewAnswers from './AdminReviewAnswers'
import { formatNoteDate } from './noteAttribution'

const INPUT = 'w-full rounded-lg border border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 p-2 text-sm'
const BUTTON = 'min-h-11 px-3 rounded-lg border border-stone-200 dark:border-stone-700 text-sm hover:bg-stone-100 dark:hover:bg-stone-700 disabled:opacity-50 disabled:cursor-not-allowed'

export default function AdminDecisionPanel() {
  const { isPiAdmin, user } = useReviewer()
  const { activeQueryId } = useApp()
  return isPiAdmin && activeQueryId !== null
    ? <DecisionPanel key={`${user?.id}:${activeQueryId}`} queryId={activeQueryId} />
    : null
}

function DecisionPanel({ queryId }: { queryId: number }) {
  const [item, setItem] = useState<AdminReviewItem | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  const [key, setKey] = useState('')
  const [none, setNone] = useState(false)
  const [notes, setNotes] = useState('')
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  useEffect(() => {
    let cancelled = false
    setError(null)
    fetchAdminReviewItem(queryId).then(
      (data) => {
        if (cancelled) return
        setItem(data)
        setKey(data.deciding_answer?.ccl_key ?? '')
        setNone(data.deciding_answer !== null && data.deciding_answer.ccl_key === null)
        setNotes(data.deciding_answer?.notes ?? '')
      },
      (err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load reviewer answers.')
      },
    )
    return () => { cancelled = true }
  }, [queryId, revision])

  const keyError = none ? null : unsafeTextError(key, CCL_KEY_SUBJECT)
  const noteError = unsafeTextError(notes, 'A decision note')
  const final = item?.deciding_answer
  const showForm = item && (!final || editing)

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (!item || saving || keyError || noteError || (!none && !key.trim())) return
    setSaving(true)
    setError(null)
    setNotice(null)
    try {
      const answer = await saveDecidingAnswer(queryId, {
        ccl_key: none ? null : key.trim(), none_of_these: none, notes,
      })
      if (!mounted.current) return
      setItem({ ...item, deciding_answer: answer, status: answer.ccl_key ? 'agreed' : 'no_key' })
      setKey(answer.ccl_key ?? '')
      setNotes(answer.notes)
      setEditing(false)
      setNotice('Final answer recorded.')
    } catch (err) {
      if (mounted.current) setError(err instanceof Error ? err.message : 'Could not record the final answer.')
    } finally {
      if (mounted.current) setSaving(false)
    }
  }

  return (
    <section aria-label="Admin deciding answer" className="max-h-[45vh] overflow-y-auto flex-shrink-0 border-t border-stone-200 dark:border-stone-700 bg-white dark:bg-surface-800 p-4 space-y-3 font-ui">
      <h2 className="text-base font-semibold text-stone-800 dark:text-stone-100">Final answer <span className="text-xs font-normal text-stone-500 dark:text-stone-400">PI/admin only</span></h2>
      {error && <p role="alert" className="text-sm text-red-700 dark:text-red-300">{error}</p>}
      {!item ? error
        ? <button type="button" className={BUTTON} onClick={() => setRevision((value) => value + 1)}>Retry answers</button>
        : <p role="status" className="text-sm text-stone-500">Loading reviewer answers...</p>
        : <>
          <details>
            <summary className="cursor-pointer min-h-11 py-2 text-sm text-stone-700 dark:text-stone-200">
              Reviewer answers ({item.answers.length}): {REVIEW_STATUS_LABELS[item.reviewer_status] ?? 'No reviews'}
            </summary>
            <AdminReviewAnswers answers={item.answers} />
          </details>
          {final && (
            <div className="rounded-lg bg-stone-50 dark:bg-surface-900 p-3 text-sm break-words space-y-1">
              <p className="font-semibold text-stone-800 dark:text-stone-100">Recorded final answer: {final.ccl_key ?? 'None of these'}</p>
              <p className="text-stone-500 dark:text-stone-400">{final.reviewer}, <time dateTime={final.timestamp}>{formatNoteDate(final.timestamp)}</time></p>
              {final.notes && <p className="whitespace-pre-wrap text-stone-700 dark:text-stone-200">{final.notes}</p>}
              {!editing && <button type="button" className={BUTTON} onClick={() => setEditing(true)}>Change final answer</button>}
            </div>
          )}
          {showForm && (
            <form onSubmit={(event) => void save(event)} className="space-y-3">
              <p className="text-sm text-stone-500 dark:text-stone-400">This overrides the CCL key export result. Earlier answers are kept. It does not file the document into a reviewer directory.</p>
              <fieldset disabled={saving} className="space-y-3">
                <legend className="sr-only">Choose the final answer</legend>
                <div className="flex flex-wrap gap-4 text-sm">
                  <label className="flex items-center gap-2 min-h-11"><input type="radio" name="final-kind" checked={!none} onChange={() => setNone(false)} /> CCL key</label>
                  <label className="flex items-center gap-2 min-h-11"><input type="radio" name="final-kind" checked={none} onChange={() => setNone(true)} /> None of these</label>
                </div>
                {!none && <div>
                  <label htmlFor="final-ccl-key" className="block text-sm mb-1">Final CCL key</label>
                  <input id="final-ccl-key" className={INPUT} value={key} maxLength={200} required onChange={(event) => setKey(event.target.value)} aria-invalid={Boolean(keyError)} aria-describedby={keyError ? 'final-key-error' : undefined} />
                  {keyError && <p id="final-key-error" role="alert" className="text-sm text-red-700 dark:text-red-300">{keyError}</p>}
                </div>}
                <div>
                  <label htmlFor="final-note" className="block text-sm mb-1">Decision note (optional)</label>
                  <textarea id="final-note" className={INPUT} value={notes} rows={2} maxLength={10000} onChange={(event) => setNotes(event.target.value)} aria-invalid={Boolean(noteError)} aria-describedby={noteError ? 'final-note-error' : undefined} />
                  {noteError && <p id="final-note-error" role="alert" className="text-sm text-red-700 dark:text-red-300">{noteError}</p>}
                </div>
                <div className="flex gap-2">
                  <button type="submit" className="min-h-11 px-4 rounded-lg bg-accent text-white text-sm hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed" disabled={Boolean(keyError || noteError || (!none && !key.trim()))}>{saving ? 'Recording...' : 'Record final answer'}</button>
                  {final && <button type="button" className={BUTTON} onClick={() => {
                    setKey(final.ccl_key ?? '')
                    setNone(final.ccl_key === null)
                    setNotes(final.notes)
                    setError(null)
                    setEditing(false)
                  }}>Cancel</button>}
                </div>
              </fieldset>
            </form>
          )}
        </>}
      {notice && <p role="status" className="text-sm text-stone-700 dark:text-stone-200">{notice}</p>}
    </section>
  )
}
