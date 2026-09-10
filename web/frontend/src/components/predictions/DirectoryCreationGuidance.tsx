import { DIRECTORY_CREATION_COPY } from '../../utils/reviewerDirectoryCopy'

/**
 * What confirming actually does, shown where the reviewer is about to do it
 * (issue #162).
 *
 * Static, propless and deliberately its own file: the naming form's request
 * and pending state are issue #161's, and this block has to survive that
 * rewrite unread. Placed inside the open form rather than under the closed
 * button so it is on screen at the moment of commitment, not competing with
 * the ranked list for attention beforehand.
 */
export default function DirectoryCreationGuidance() {
  return (
    <div
      data-testid="directory-creation-guidance"
      className="mt-2 rounded-md border border-stone-200 dark:border-stone-700 bg-stone-50 dark:bg-surface-800/60 px-2.5 py-2"
    >
      <p className="text-[11px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
        {DIRECTORY_CREATION_COPY.formHeading}
      </p>
      <ul className="mt-1 space-y-1 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300 list-disc pl-4">
        {DIRECTORY_CREATION_COPY.formPoints.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>
    </div>
  )
}
