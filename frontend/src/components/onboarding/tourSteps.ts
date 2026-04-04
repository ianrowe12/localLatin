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
    target: 'predictions',
    title: 'Prediction Rankings',
    description:
      'The top 10 predicted matches, ranked by similarity score. Click any prediction to view its candidate text. Use number keys 1\u20139 or arrow keys to navigate.',
    placement: 'left',
  },
  {
    target: 'feedback',
    title: 'Your Assessment',
    description:
      'Record which prediction rank you think is correct (if any), add notes, and submit your review. After submitting, the tool automatically advances to the next manuscript.',
    placement: 'left',
  },
]
