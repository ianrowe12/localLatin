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
    description: `This panel shows the candidate you have selected, and its second line says which kind it is: a ${PROVENANCE_TERMS.labeled_reference.toLowerCase()} is a text the labelled corpus already groups, while a ${PROVENANCE_TERMS.reviewer_group.toLowerCase()} is one a colleague grouped provisionally rather than one the corpus attests. Words that match the query highlight when you hover over the query panel.`,
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
      'The model\u2019s top candidate sources for this fragment, ranked by similarity. The banner above the list tells you how much to trust the top hit: red means the model has no useful opinion here, amber means read the evidence carefully, and a plain note means it is a likely match to verify. A red banner is not evidence that the source is missing from the Carolingian Canon Law (CCL) collections, and you are not being asked to search the CCL by hand. Reviewer-created directories are listed after the model\u2019s ten and are marked as such.',
    placement: 'left',
  },
  {
    target: 'predictions',
    title: 'Starting a New Directory',
    description:
      'Below the list you can start a provisional directory for this fragment. Creating one is a permanent write: the directory and its name are saved the moment you confirm, there is no rename or removal, and submitting or skipping afterwards does not undo it. Create one only when a new grouping is warranted, then name and confirm it. It is separate from your assessment, so you can reject the ranked candidates with the None option and a note without creating anything. A directory reads as matched only once a reviewer files a second, distinct document into it, never because a similarity crossed a threshold.',
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
      'Mark which predicted source is correct, or reject the model\u2019s candidates with the \u201cNone\u201d option if none fit. Only the candidates this ranking actually offers are shown. None is an independent choice: it needs no new directory, and it is not a claim that the CCL holds no match for this fragment.',
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
      'Submit your assessment to save it and move to the next fragment, or skip if you can\u2019t decide. Skipping with a note records uncertainty, which is not the same judgement as recording that none of the candidates fits. Neither one undoes a directory you have already created.',
    placement: 'left',
  },
]

// The attribution-controls step targets a PI/admin-only control; ordinary
// reviewers never see it, so omit that step from their tour entirely.
// There is no post-processing step any more: the variant picker was removed
// for every role in issue #94, and the tour must not point at a control that
// is not on screen.
// The directory step targets `predictions`, the list container, rather than the
// creation button: that button is absent above the no-match band and absent
// again once this document already seeds a directory, and TourOverlay skips a
// step whose target it cannot find. The guidance has to be reachable in all
// three states, so it hangs off the container that is always mounted.
export function getReviewTourSteps(isPiAdmin: boolean): TourStep[] {
  return REVIEW_TOUR_STEPS.filter(
    (step) => isPiAdmin || step.target !== 'attribution-controls',
  )
}
