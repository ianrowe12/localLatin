import { PROVENANCE_TERMS } from '../../utils/documentProvenance'

export interface TourStep {
  target: string
  title: string
  description: string
  placement?: 'top' | 'bottom' | 'left' | 'right'
}

export const DASHBOARD_TOUR_STEPS: TourStep[] = [
  {
    target: 'stats-cards',
    title: 'Your Progress',
    description:
      "These cards show your review progress at a glance \u2014 how many manuscripts you've reviewed, how many are left, and your overall completion rate.",
    placement: 'bottom',
  },
  {
    target: 'resume-btn',
    title: 'Resume Reviewing',
    description:
      "Click here to jump right back into reviewing where you left off. We'll take you to the next unreviewed manuscript.",
    placement: 'top',
  },
  {
    target: 'recent-activity',
    title: 'Recent Activity',
    description:
      "See the manuscripts you've recently reviewed. Click any entry to revisit that review.",
    placement: 'top',
  },
]

export const REVIEW_TOUR_STEPS: TourStep[] = [
  {
    target: 'query-list',
    title: 'Manuscript List',
    description:
      'Browse all manuscript fragments here. Each card shows a filename and a preview of the Latin text.',
    placement: 'right',
  },
  {
    target: 'search-bar',
    title: 'Search',
    description:
      'Type a filename to quickly find a specific manuscript fragment.',
    placement: 'bottom',
  },
  {
    target: 'filter-chips',
    title: 'Filter by Status',
    description:
      'Filter the list to show only pending (unreviewed), reviewed, or all manuscripts.',
    placement: 'bottom',
  },
  {
    target: 'query-panel',
    title: 'Query Text',
    description: `This panel is always the ${PROVENANCE_TERMS.query.toLowerCase()}: the fragment you are reviewing. Hover over words to see matching words highlighted in the candidate panel on the right.`,
    placement: 'right',
  },
  {
    target: 'candidate-panel',
    title: 'Candidate Match',
    // The reviewer-group half of this sentence went with issue #221: the only
    // way to open such a candidate was the reviewer-directory block, which the
    // reviewer-facing page no longer draws, so the tour stopped describing a
    // caption nobody can reach. `PROVENANCE_TERMS` still carries the term for
    // the components that caption one.
    description: `This panel shows the candidate you have selected, and its second line says which kind it is: a ${PROVENANCE_TERMS.labeled_reference.toLowerCase()} is a text the labelled corpus already groups. Words that match the query highlight when you hover over the query panel.`,
    placement: 'left',
  },
  {
    target: 'model-selector',
    title: 'Choose a Model',
    description:
      "Pick which model's predictions you want to review. Switching models reloads the ranked sources below.",
    placement: 'left',
  },
  {
    target: 'predictions',
    title: 'Predicted Sources',
    description:
      'The model\u2019s top candidate sources for this fragment, ranked by similarity. The banner above the list tells you how much to trust the top hit: red means the model has no useful opinion here, amber means read the evidence carefully, and a plain note means it is a likely match to verify. A red banner is not evidence that the source is missing from the Carolingian Canon Law (CCL) collections, and you are not being asked to search the CCL by hand.',
    placement: 'left',
  },
  {
    // Was "Starting a New Directory", hung off the list container because the
    // red creation button came and went. That button, its naming form and the
    // whole flow are retired (issue #196), so the step now describes the one
    // control that can still start a group and points at it.
    target: 'match-options',
    title: 'When the source is not in the list',
    description:
      'The ranked list is the model’s answer and nothing else. If the right source is not in it, say so with the blue “None of the top” button and, if you know it, give the CCL key: the key is saved with your answer, and it may also start a permanent group for this fragment. A group cannot be renamed or removed, and submitting or skipping afterwards does not undo it.',
    placement: 'left',
  },
  {
    target: 'evidence-view',
    title: 'Evidence View',
    description:
      'Switch how matching words are shown between the two panels \u2014 as highlights or as connection lines.',
    placement: 'right',
  },
  {
    target: 'attribution-controls',
    title: 'Attribution Controls',
    description:
      'Choose which attribution method drives the evidence highlighting.',
    placement: 'right',
  },
  {
    target: 'match-options',
    title: 'Record a Match',
    description:
      'Press the numbered button of the source that is correct. If none of them fits, press the blue \u201cNone of the top\u201d button: it opens one optional box for the CCL key of the source, if you happen to know it, and one button to record the answer. Only the candidates this ranking actually offers are shown, and rejecting them is not a claim that the CCL holds no match for this fragment.',
    placement: 'left',
  },
  {
    target: 'multi-select',
    title: 'Selecting Multiple',
    description:
      'Turn this on when more than one source could be correct, then pick all that apply.',
    placement: 'left',
  },
  {
    target: 'notes',
    title: 'Add Notes',
    description:
      'Leave a note explaining your decision. A note is required when you skip a fragment.',
    placement: 'left',
  },
  {
    target: 'submit-skip',
    title: 'Submit or Skip',
    description:
      'Submit your assessment to save it and move to the next fragment, or skip if you can\u2019t decide. Once you have recorded an answer with the blue button, this one reads \u201cNext\u201d and only moves you on: nothing further is saved. Skipping with a note records uncertainty, which is not the same judgement as recording that none of the candidates fits. Neither one undoes a group that a CCL key has already started.',
    placement: 'left',
  },
]

// The attribution-controls step targets a PI/admin-only control; ordinary
// reviewers never see it, so omit that step from their tour entirely.
// There is no post-processing step any more: the variant picker was removed
// for every role in issue #94, and the tour must not point at a control that
// is not on screen.
// The "source is not in the list" step targets `match-options`, the pill grid
// that holds the blue None control: that is now the only control that can
// start a group, and the grid is mounted whenever there is a ranking to judge.
// It used to hang off `predictions` because the creation button it described
// came and went with the confidence band; the button is retired (issue #196)
// and the description went with it (issue #221). TourOverlay skips a step whose
// target it cannot find, which is the floor if a ranking never arrives.
export function getReviewTourSteps(isPiAdmin: boolean): TourStep[] {
  return REVIEW_TOUR_STEPS.filter(
    (step) => isPiAdmin || step.target !== 'attribution-controls',
  )
}
