import { DIRECTORY_CREATION_COPY } from '../../utils/reviewerDirectoryCopy'

interface DirectorySavedNoticeProps {
  /** The label the directory was saved under, as the server returned it. */
  label: string
  /** Dropped by the caller's layout; the notice itself owns nothing else. */
  className?: string
}

/**
 * The acknowledgement for a directory that is saved on the server (issue #162).
 *
 * Takes the label as a prop and holds no state, so issue #161 can mount it from
 * a durable saved-directory record, or after rehydrating one from the server on
 * reload, without rewriting a word of it.
 *
 * The wording is conditional on purpose. The old success line said the
 * directory "is now a candidate for every other document", which the scoring
 * cannot promise: members never see their own directory, an excluded query
 * cannot be scored at all, and only the closest few reviewer directories reach
 * any one ranked list.
 */
export default function DirectorySavedNotice({
  label,
  className = '',
}: DirectorySavedNoticeProps) {
  return (
    <div
      data-testid="new-directory-created"
      role="status"
      className={`rounded-lg border border-indigo-300 dark:border-indigo-400/40 bg-indigo-50 dark:bg-indigo-500/10 px-3 py-2 ${className}`}
    >
      <div className="text-xs font-semibold text-indigo-700 dark:text-indigo-300">
        {DIRECTORY_CREATION_COPY.savedHeading}
      </div>
      <div className="font-ui text-xs text-stone-700 dark:text-stone-300 mt-0.5 break-words">
        {label}
      </div>
      <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1">
        {DIRECTORY_CREATION_COPY.savedAvailability}
      </p>
      <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1">
        {DIRECTORY_CREATION_COPY.savedIndependence}
      </p>
    </div>
  )
}
