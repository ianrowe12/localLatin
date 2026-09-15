import type {
  AssessmentCandidate,
  CandidateEvidence,
} from '../../contexts/assessmentEligibility'

/** Why this candidate cannot be chosen, as a fact rather than a verdict. */
export function unreadableCandidateReason(
  candidate: AssessmentCandidate,
): string | null {
  const evidence: CandidateEvidence = candidate.evidence
  if (evidence === 'readable') return null
  if (evidence === 'unidentified') {
    return 'This candidate arrived without a usable directory name or score, so there is nothing to assign the document to.'
  }
  return `No witness of ${candidate.dirName} that this response delivered carries readable text, so there is nothing here to judge it by.`
}

interface MatchPillsProps {
  /** The candidates the current ranking actually offers, in its own order. */
  candidates: AssessmentCandidate[]
  /**
   * Whether this ranking supports an evaluation at all. False disables every
   * pill: a ranking with no readable model candidate cannot be judged, and an
   * enabled control whose click the context refuses is worse than no control.
   */
  canEvaluate: boolean
  /** Confirmed choices, in the order they were made. */
  selectedRanks: number[]
  /** Restored choices whose directory is unknown, awaiting reconfirmation. */
  unconfirmedRanks: number[]
  noneSelected: boolean
  noneAvailable: boolean
  /** How many model candidates "None" is an answer about. */
  modelCandidateCount: number
  onToggleCandidate: (candidate: AssessmentCandidate) => void
  onToggleNone: () => void
}

/**
 * Deliberately large (issue #196). Prof. Firey, 15 September 2026: the new
 * evaluator wants "the big prediction buttons right there" with as much space
 * around them as possible, and this panel now holds only these ten plus the
 * blue action. The old pills were `text-xs py-1.5` in a tight 1.5-unit grid,
 * sized for a panel that also carried a red call to action, its caption and a
 * creation form.
 */
const PILL_BASE =
  'text-base font-semibold py-3 px-2 rounded-xl text-center whitespace-nowrap transition-all focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:cursor-not-allowed'

/** The blue action. Same size logic, full width, and not a rank. */
const NONE_BASE =
  'col-span-5 text-sm font-semibold py-3 px-3 rounded-xl text-center transition-all focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:cursor-not-allowed'

function noneLabel(count: number): string {
  return count === 1 ? 'None of the 1 candidate' : `None of the top ${count}`
}

/**
 * The rank controls, drawn from the ranking on screen (issue #157).
 *
 * They used to be `Array.from({ length: maxRank })`, with `maxRank` falling
 * back to ten whenever the panel had no candidates. That produced ten pressable
 * pills over a failed request, and numbered them 1..n rather than by rank, so a
 * sparse ranking of [1, 11] -- one model candidate and one reviewer directory,
 * which were anchored at eleven -- was offered as "#1" and "#2". Nothing here
 * invents a rank any more: no candidates means no pills.
 *
 * Since issue #196 the candidates it is given are the model's alone
 * (`assessmentEligibility.assessmentEvidence`), so a rank here is always one of
 * the ten the retrieval run produced. The reviewer-directory branches below are
 * kept for a candidate that still arrives marked `reviewer` from a cached
 * response; they cost two lines and they cannot mislabel anything.
 */
export default function MatchPills({
  candidates,
  canEvaluate,
  selectedRanks,
  unconfirmedRanks,
  noneSelected,
  noneAvailable,
  modelCandidateCount,
  onToggleCandidate,
  onToggleNone,
}: MatchPillsProps) {
  return (
    <div data-tour="match-options" className="grid grid-cols-5 gap-2.5">
      {candidates.map((candidate) => {
        const isSelected = selectedRanks.includes(candidate.rank)
        const isUnconfirmed = unconfirmedRanks.includes(candidate.rank)
        const reviewer = candidate.source === 'reviewer'
        const disabled = !candidate.usable || !canEvaluate
        const title = !candidate.usable
          ? (unreadableCandidateReason(candidate) ?? undefined)
          : !canEvaluate
            ? 'This ranking has no model candidate readable on this screen, so there is nothing to judge it against.'
            : isUnconfirmed
              ? `Restored from an earlier draft with no directory recorded. Select #${candidate.rank} again to confirm ${candidate.dirName}.`
              : reviewer
                ? `Reviewer directory ${candidate.label ?? candidate.dirName}`
                : candidate.dirName
        return (
          <button
            key={`${candidate.rank}-${candidate.dirName}`}
            type="button"
            onClick={() => onToggleCandidate(candidate)}
            disabled={disabled}
            title={title}
            aria-label={
              reviewer
                ? `Match reviewer directory #${candidate.rank}`
                : `Match prediction #${candidate.rank}`
            }
            aria-pressed={isSelected}
            data-testid={`match-pill-${candidate.rank}`}
            data-unconfirmed={isUnconfirmed ? 'true' : undefined}
            className={`${PILL_BASE} ${
              disabled
                ? `bg-stone-100 dark:bg-stone-800 text-stone-400 dark:text-stone-500 opacity-70 ${
                    // Struck through only when this candidate itself cannot be
                    // read; a readable candidate in an unjudgeable ranking is
                    // greyed, not crossed out.
                    candidate.usable ? '' : 'line-through'
                  }`
                : isSelected
                  ? 'bg-correct/15 text-correct ring-1 ring-correct/40 font-medium cursor-pointer'
                  : isUnconfirmed
                    ? 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 ring-1 ring-dashed ring-amber-500/70 cursor-pointer'
                    : reviewer
                      ? 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-300 hover:bg-indigo-100 dark:hover:bg-indigo-500/20 cursor-pointer'
                      : 'bg-stone-100 dark:bg-stone-700 text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-600 cursor-pointer'
            }`}
          >
            #{candidate.rank}
          </button>
        )
      })}

      {/* THE BLUE ACTION (issue #196). It is named after the model's
          candidates, which are now the only ranked things on screen, and it is
          the one way to say "the source is elsewhere": pressing it opens a
          single optional field for the CCL key. It is blue rather than red
          because it is an ordinary answer, not a warning -- the red treatment
          belonged to the retired new-directory button. */}
      <button
        type="button"
        onClick={onToggleNone}
        disabled={!noneAvailable && !noneSelected}
        data-testid="match-pill-none"
        title={
          noneAvailable
            ? undefined
            : 'Some candidates cannot be read on this screen, so they cannot be rejected.'
        }
        className={`${NONE_BASE} ${
          !noneAvailable
            ? noneSelected
              ? // Held, not honoured: the reviewer's own answer stays visible
                // and clearable, but it cannot be saved as it stands.
                'bg-indigo-50 dark:bg-indigo-500/5 text-indigo-500/70 ring-1 ring-dashed ring-indigo-400/40 cursor-pointer'
              : 'bg-stone-100 dark:bg-stone-800 text-stone-400 dark:text-stone-500 opacity-70'
            : noneSelected
              ? 'bg-indigo-600 text-white ring-1 ring-indigo-500 cursor-pointer'
              : 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-200 hover:bg-indigo-100 dark:hover:bg-indigo-500/20 cursor-pointer'
        }`}
        aria-pressed={noneSelected}
      >
        {noneLabel(modelCandidateCount)}
      </button>
    </div>
  )
}
