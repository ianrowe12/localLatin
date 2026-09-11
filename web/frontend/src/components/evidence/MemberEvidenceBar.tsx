import { useId } from 'react'
import {
  describeAttributionScope,
  describeScoreAttribution,
  describeSoleWitness,
  describeUnopenableMembers,
  describeWitnessOption,
  memberEvidenceVisible,
  type AttributionScope,
  type MemberEvidence,
} from './memberEvidence'

interface MemberEvidenceBarProps {
  /** Resolved by `resolveMemberEvidence`; null renders nothing. */
  evidence: MemberEvidence | null
  /** Records the reviewer's choice, scoped by `useSelectedMember`. */
  onSelectWitness: (filename: string) => void
  /**
   * Whether the loaded token map describes the witness on screen. Pass the
   * result of `attributionAppliesToWitness` only while the panel is actually
   * shading something; omit it when no highlighting is visible, so the note
   * appears next to the marks it qualifies rather than on every card.
   */
  attribution?: AttributionScope | null
  /** True when the panel is shading word overlap rather than attribution. */
  lexicalHighlighting?: boolean
}

/**
 * The line between a directory's number and the witness under it (issue #163).
 *
 * Presentational only. It fetches nothing, decides nothing about which
 * prediction is current, and never reorders or rescores members: it renders
 * what `resolveMemberEvidence` derived from one candidate payload.
 *
 * The smallest presentation that answers #163's question, in the order a
 * reviewer needs it:
 *
 *   1. what the number covers, and which witness produced it, always in
 *      words rather than in colour;
 *   2. which witness is on screen, changeable through a plain native select
 *      that keeps the API's member order;
 *   3. one click back to the supporting witness, so the evidence for the
 *      number is never more than one action away.
 *
 * The number itself does not change with the selection, because it is the
 * group's maximum and not an individual score. Nothing here invents a
 * per-member number to fill the gap.
 */
export default function MemberEvidenceBar({
  evidence,
  onSelectWitness,
  attribution = null,
  lexicalHighlighting = false,
}: MemberEvidenceBarProps) {
  const selectId = useId()

  const attributionNote = attribution
    ? describeAttributionScope(attribution, { lexicalHighlighting })
    : null

  if (!evidence) return null

  const unopenableNote = describeUnopenableMembers(evidence)
  const soleWitnessNote = describeSoleWitness(evidence)

  // A labelled directory showing its only file, with nothing to warn about,
  // has nothing to say. Staying silent there keeps the strip meaningful.
  // `memberEvidenceVisible` is the same predicate `CenterArea` asks before it
  // leaves the score off the document header, so the number is never printed
  // twice and never dropped from both places.
  if (!memberEvidenceVisible(evidence, attribution, { lexicalHighlighting }))
    return null

  const scoreCopy = describeScoreAttribution(evidence)
  const attention = scoreCopy.tone === 'attention'

  const supportKey =
    evidence.support.kind === 'named' && evidence.support.inspectable
      ? (evidence.witnesses.find((w) => w.isSupporting)?.key ?? null)
      : null

  return (
    <section
      aria-label="Member evidence"
      data-testid="member-evidence"
      className={`px-3 py-1.5 border-b text-xs flex-shrink-0 font-ui ${
        attention
          ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800/40 text-amber-900 dark:text-amber-200'
          : 'bg-stone-100 dark:bg-stone-800 border-stone-200 dark:border-stone-700 text-stone-600 dark:text-stone-300'
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        {attention && (
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className="flex-shrink-0"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="13" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        )}
        <span className="font-semibold uppercase tracking-wider text-[10px] whitespace-nowrap">
          {scoreCopy.label}
        </span>
        <span className="font-mono font-semibold whitespace-nowrap">
          {evidence.score.toFixed(3)}
        </span>
        {/* Announced on selection change, but deliberately not role="status":
            the candidate panel already owns one status region. */}
        <span
          data-testid="member-evidence-attribution"
          aria-live="polite"
          className="leading-snug"
        >
          {scoreCopy.sentence}
        </span>
      </div>

      {evidence.selectable ? (
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <label htmlFor={selectId} className="whitespace-nowrap">
            Viewing witness
          </label>
          <select
            id={selectId}
            data-testid="member-selector"
            value={evidence.selected?.key ?? ''}
            onChange={(e) => onSelectWitness(e.target.value)}
            className="h-8 max-w-[22rem] px-1.5 rounded border border-stone-300 dark:border-stone-600
                       bg-white dark:bg-surface-800 text-xs
                       text-stone-800 dark:text-stone-200
                       focus:outline-none focus:ring-2 focus:ring-accent/40"
          >
            {evidence.witnesses.map((witness) => (
              <option key={witness.key} value={witness.key}>
                {describeWitnessOption(witness)}
              </option>
            ))}
          </select>
          <span className="text-stone-500 dark:text-stone-400 whitespace-nowrap">
            {evidence.unopenableCount > 0
              ? `${evidence.witnesses.length} of ${evidence.memberCount} members`
              : `${evidence.memberCount} members`}
          </span>
          {!evidence.displayedIsSupport && supportKey != null && (
            <button
              type="button"
              data-testid="show-supporting-witness"
              onClick={() => onSelectWitness(supportKey)}
              className="px-1.5 py-0.5 rounded font-medium underline underline-offset-2
                         hover:bg-black/5 dark:hover:bg-white/10
                         focus:outline-none focus:ring-2 focus:ring-accent/40"
            >
              Show supporting witness
            </button>
          )}
        </div>
      ) : (
        soleWitnessNote && (
          <div
            data-testid="member-evidence-single"
            className="mt-1 text-stone-500 dark:text-stone-400"
          >
            {soleWitnessNote}
          </div>
        )
      )}

      {unopenableNote && (
        <div
          data-testid="member-evidence-unopenable"
          className="mt-1 text-stone-500 dark:text-stone-400"
        >
          {unopenableNote}
        </div>
      )}

      {attributionNote && (
        <div
          data-testid="member-evidence-model-attribution"
          className="mt-1 text-stone-500 dark:text-stone-400"
        >
          {attributionNote}
        </div>
      )}
    </section>
  )
}
