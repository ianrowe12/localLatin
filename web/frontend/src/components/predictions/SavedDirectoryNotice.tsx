import { useSavedDirectoryFor } from '../../contexts/SavedDirectoryContext'
import { listsItsSeedAsMember } from '../../api/reviewerDirs'
import {
  alsoGroupedNote,
  DIRECTORY_CREATION_COPY,
  savedByNote,
  SEED_NOT_FILED,
  SEED_NOT_FILED_SHORT,
} from '../../utils/reviewerDirectoryCopy'
import AwaitingMatchBadge from './AwaitingMatchBadge'
import DirectorySavedNotice from './DirectorySavedNotice'

interface SavedDirectoryNoticeProps {
  queryId: number
  model: string
}

/**
 * What is already recorded about THIS document's own directory. Read-only.
 *
 * This is what remains of `NewDirectoryCta` after issue #196 retired the red
 * button (Prof. Firey, 15 September 2026: the button and its explanation go, so
 * the review panel keeps its elegance). Every control that could write is gone
 * -- the Create button, the naming form, its guidance block, the retry after an
 * unsettled write -- because there is no longer a creation flow here at all.
 * Declaring a source now happens in the assessment panel, by naming its CCL key
 * beside "None of the top N", and the server decides whether that joins an
 * existing directory or starts one.
 *
 * What is NOT gone is the durable record itself (issue #161). Directories
 * created before this release are permanent, and a reviewer looking at a
 * document that seeds one must still be told so -- otherwise the app quietly
 * hides a grouping it cannot remove. So this reads the same store, in the same
 * states, and simply has nothing to offer in `absent`:
 *
 *   saved       -> the acknowledgement, whatever the prediction request is doing
 *   absent      -> nothing at all
 *   checking    -> nothing; the question is not worth a line if there is no
 *                  action waiting on the answer
 *   unresolved  -> a quiet line, because "could not check" is not "no directory"
 *
 * It still REQUIRES a `SavedDirectoryProvider` above it, and throws without one.
 */
export default function SavedDirectoryNotice({
  queryId,
  model,
}: SavedDirectoryNoticeProps) {
  const { identity } = useSavedDirectoryFor(queryId, { model })

  if (identity.status === 'saved') {
    const { primary, dirs } = identity
    const others = dirs.filter((dir) => dir.dir_id !== primary.dir_id)
    return (
      <div className="mt-2 mx-2" data-testid="new-directory-saved">
        <DirectorySavedNotice label={primary.label} />
        {!listsItsSeedAsMember(primary) && (
          // A preserved partial write from before issue #160. The grouping is
          // real and permanent; the membership row that would make this
          // document one of its members is not there and cannot be added.
          <p
            data-testid="new-directory-membership-gap"
            className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1"
          >
            {SEED_NOT_FILED}
          </p>
        )}
        <div className="mt-1">
          <AwaitingMatchBadge seededDirs={[...dirs]} />
        </div>
        <p
          data-testid="new-directory-attribution"
          className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 mt-1"
        >
          {savedByNote(primary.created_by)}
        </p>
        {others.length > 0 && (
          // Historical duplicates, from before issue #160's atomic creation.
          // Nothing can remove them, so hiding them behind the oldest one would
          // misdescribe the database a reviewer is being asked to trust.
          <div data-testid="new-directory-other-groups" className="mt-1">
            <p className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400">
              {alsoGroupedNote(others.length)}
            </p>
            <ul className="font-ui text-xs leading-snug text-stone-600 dark:text-stone-400 list-disc pl-4">
              {others.map((dir) => (
                <li key={dir.dir_id}>
                  {dir.label} — {savedByNote(dir.created_by)}
                  {!listsItsSeedAsMember(dir) && ` ${SEED_NOT_FILED_SHORT}`}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )
  }

  if (identity.status === 'unresolved') {
    return (
      <p
        role="status"
        data-testid="new-directory-unresolved"
        className="px-2 mt-2 font-ui text-xs leading-snug text-stone-600 dark:text-stone-300"
      >
        {DIRECTORY_CREATION_COPY.unresolvedNote}
      </p>
    )
  }

  return null
}
