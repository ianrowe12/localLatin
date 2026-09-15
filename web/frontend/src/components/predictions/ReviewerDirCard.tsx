import { memo } from 'react'
import type { ReviewerDirCandidate } from '../../api/queries'
import { PROVENANCE_TERMS } from '../../utils/documentProvenance'

interface ReviewerDirCardProps {
  card: ReviewerDirCandidate
  isActive: boolean
  onClick: () => void
}

/**
 * A directory that came from a reviewer, not from the labelled corpus, and NOT
 * from the model's ranking.
 *
 * Deliberately not a variant of PredictionCard. The two carry different kinds
 * of evidence -- a labelled directory is a grouping the corpus already asserts,
 * a reviewer directory is one colleague's provisional judgement over documents
 * that arrived unlabeled -- and a reviewer scanning the list has to be able to
 * tell them apart without reading. (An earlier comment here called the labelled
 * side "an edited, attested source". It is not: the corpus is hand copies,
 * most of them never edited to modern standards, so the distinction is
 * labelled-versus-unlabeled, not edited-versus-provisional.)
 *
 * NO RANK (issue #196). The card used to carry an anchored 11 or 12 in the same
 * numbered circle the model's candidates use, in the same list, and evaluators
 * read it as the model's answer. Abigail: "the red button causes more confusion
 * than the clarity we hoped it would provide", and the numbered reviewer card
 * was the other half of that confusion. The circle is gone, the card sits under
 * its own heading, and it cannot be chosen with a rank pill: a document is filed
 * here by naming this directory's CCL key beside "None of the top N".
 *
 * The score survives, because a score is a property of the pair rather than a
 * position in a list, and it is on the same scale as a model candidate's (max
 * cosine over member documents).
 */
function ReviewerDirCardInner({ card, isActive, onClick }: ReviewerDirCardProps) {
  const title = card.label || card.dir_id
  const memberCount = card.dir_files.length
  // The key joins it; a pre-#196 group has none, and its label is the handle
  // the server falls back to (`_reviewer_dir_for_key`).
  const joinHandle = card.ccl_key || card.label || card.dir_id

  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={`reviewer-dir-card-${card.dir_id}`}
      className={`p-2.5 rounded-lg cursor-pointer transition-all duration-150 text-left w-full border-dashed ${
        isActive
          ? 'border-2 border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10 shadow-sm'
          : 'border-2 border-indigo-300/70 dark:border-indigo-400/40 hover:bg-indigo-50/60 dark:hover:bg-indigo-500/5'
      }`}
      aria-pressed={isActive}
      aria-label={`Reviewer directory ${title}, not ranked`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span
            className="text-[10px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-300 whitespace-nowrap"
            title={`Provisional grouping of originally unlabeled documents. ${PROVENANCE_TERMS.reviewer_group} on the right-hand panel.`}
          >
            Reviewer directory
          </span>
        </div>

        <div
          className="text-sm font-medium truncate max-w-full text-stone-800 dark:text-stone-200"
          title={title}
        >
          {title}
        </div>

        <div className="h-1.5 rounded-full bg-stone-200 dark:bg-stone-700 mt-1.5 overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all"
            style={{ width: `${Math.max(0, card.score) * 100}%` }}
          />
        </div>

        <div className="text-xs text-stone-500 mt-1">
          <span className="font-ui">Similarity:</span>{' '}
          <span className="font-mono">{card.score.toFixed(3)}</span>
        </div>

        <div className="font-ui text-[11px] text-stone-400 dark:text-stone-500 mt-0.5 truncate">
          {card.created_by ? `Created by ${card.created_by}` : 'Reviewer-created'}
          {' · '}
          {memberCount === 1
            ? '1 originally unlabeled document'
            : `${memberCount} originally unlabeled documents`}
        </div>

        {/* How to file this document here, spelled out (review finding 2).
            Filing is by name now, and the name that works is the directory's
            CCL key -- or, for a group made before issue #196 and named after a
            siglum, the label above. Without this line an evaluator looking at
            `New directory from BN2123.89r.5` has no way to know that typing it
            joins the group rather than starting a second one. */}
        <div
          data-testid={`reviewer-dir-join-${card.dir_id}`}
          className="font-ui text-[11px] leading-snug text-stone-500 dark:text-stone-400 mt-1"
        >
          To file this document here, type{' '}
          <span className="font-mono break-all">{joinHandle}</span> in the CCL key
          box.
        </div>
      </div>
    </button>
  )
}

const ReviewerDirCard = memo(ReviewerDirCardInner)
ReviewerDirCard.displayName = 'ReviewerDirCard'

export default ReviewerDirCard
