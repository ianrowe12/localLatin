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
    description:
      'This panel shows the unlabelled manuscript fragment. Hover over words to see matching words highlighted in the candidate panel on the right.',
    placement: 'right',
  },
  {
    target: 'candidate-panel',
    title: 'Candidate Match',
    description:
      'This panel shows the predicted matching canon text. Words that match the query will highlight when you hover over the query panel.',
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
      'The model\u2019s top candidate sources for this fragment, ranked by similarity. Click any one to inspect it.',
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
      'Choose which attribution method and variant drive the evidence highlighting.',
    placement: 'right',
  },
  {
    target: 'match-options',
    title: 'Record a Match',
    description:
      'Mark which predicted source is correct, or choose \u201cNone of top N\u201d if none fit.',
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
      'Submit your assessment to save it and move to the next fragment, or skip if you can\u2019t decide.',
    placement: 'left',
  },
]

// The attribution-controls step targets a PI/admin-only control; ordinary
// reviewers never see it, so omit that step from their tour entirely.
export function getReviewTourSteps(isPiAdmin: boolean): TourStep[] {
  return REVIEW_TOUR_STEPS.filter(
    (step) => isPiAdmin || step.target !== 'attribution-controls',
  )
}
