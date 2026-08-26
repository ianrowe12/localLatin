import type { ReviewerDir } from '../../api/reviewerDirs'

interface AwaitingMatchBadgeProps {
  /** Reviewer directories seeded by the document being shown. */
  seededDirs: ReviewerDir[]
}

/**
 * Status of the directories this document seeded.
 *
 * The badge belongs to the *document*, not to any candidate card: it answers
 * "what happened to the directory I created here", which is a fact about the
 * seed. A directory is awaiting a match until some other query scores at or
 * above the no-match band against it; once one does, the badge flips and both
 * sides can see the pairing (the other query gets the directory as a
 * candidate, this one gets a matched badge).
 *
 * Nothing renders when the document seeded no directories, so the header is
 * unchanged for the overwhelming majority of queries.
 */
export default function AwaitingMatchBadge({ seededDirs }: AwaitingMatchBadgeProps) {
  if (seededDirs.length === 0) return null

  const awaiting = seededDirs.filter((dir) => dir.status === 'awaiting_match')
  const matched = seededDirs.filter((dir) => dir.status === 'matched')

  // "· lead" means the model already ranks another document above the band
  // against this directory, but no reviewer has confirmed it. Deliberately not
  // the same visual as a confirmed match.
  const leads = awaiting.filter((dir) => dir.has_potential_match)
  const awaitingTitle = awaiting
    .map((dir) =>
      dir.has_potential_match
        ? `${dir.label} — a related document scores ${dir.best_match_score?.toFixed(2)}, unconfirmed`
        : dir.label,
    )
    .join(', ')

  return (
    <span className="flex items-center gap-1.5">
      {awaiting.length > 0 && (
        <span
          data-testid="awaiting-match-badge"
          title={awaitingTitle}
          className="bg-amber-100 dark:bg-amber-500/15 text-amber-800 dark:text-amber-300 text-xs px-2 py-0.5 rounded-full whitespace-nowrap font-ui"
        >
          Awaiting future match
          {awaiting.length > 1 ? ` (${awaiting.length})` : ''}
          {leads.length > 0 ? ' · lead' : ''}
        </span>
      )}
      {matched.length > 0 && (
        <span
          data-testid="matched-dir-badge"
          title={matched.map((dir) => dir.label).join(', ')}
          className="bg-emerald-100 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 text-xs px-2 py-0.5 rounded-full whitespace-nowrap font-ui"
        >
          New directory matched
          {matched.length > 1 ? ` (${matched.length})` : ''}
        </span>
      )}
    </span>
  )
}
