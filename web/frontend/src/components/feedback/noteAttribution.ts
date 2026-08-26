import type { FeedbackEntry } from '../../api/feedback'

/**
 * Calendar date of a feedback row.
 *
 * SQLite stamps rows as 'YYYY-MM-DD HH:MM:SS' (UTC, no offset), which `Date`
 * parses inconsistently across engines. Read the date straight off the string
 * when it has that shape and only fall back to `Date` for anything else, such
 * as the ISO strings the mock API returns.
 */
export function formatNoteDate(timestamp: string): string {
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(timestamp)
  if (match) return match[1]
  const parsed = new Date(timestamp)
  return Number.isNaN(parsed.getTime())
    ? timestamp
    : parsed.toISOString().slice(0, 10)
}

/**
 * Attribution line for the shared latest review (issue #96).
 *
 * Notes are team-wide, so the row prefilled into the panel often belongs to
 * someone else and has to say whose it is. Prefers the login name and falls
 * back to the stored display name for rows whose account is gone. Returns null
 * when there is nobody to credit.
 */
export function formatNoteAttribution(entry: FeedbackEntry | null): string | null {
  if (!entry) return null
  const author = entry.reviewer_username?.trim() || entry.reviewer?.trim()
  if (!author) return null
  // A row with no notes still carries a decision worth attributing, but calling
  // it a "note" would be a lie about what is in the box.
  const label = entry.notes.trim() ? 'Last note by' : 'Last review by'
  return `${label} ${author} - ${formatNoteDate(entry.timestamp)}`
}
