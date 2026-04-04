import { useStats } from '../../api/models'
import StatCard, { StatCardSkeleton } from './StatCard'
import ResumeCard, { ResumeCardSkeleton } from './ResumeCard'
import RecentActivityList, {
  RecentActivitySkeleton,
} from './RecentActivityList'
import ModelProgressBars, {
  ModelProgressSkeleton,
} from './ModelProgressBars'
import RankDistributionChart, {
  RankDistributionSkeleton,
} from './RankDistributionChart'
import ReviewerTable, { ReviewerTableSkeleton } from './ReviewerTable'

// ---------------------------------------------------------------------------
// Icon helpers (inline SVGs matching the codebase style)
// ---------------------------------------------------------------------------

function CheckIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

function ChatIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-3 font-ui">
      {children}
    </h2>
  )
}

// ---------------------------------------------------------------------------
// Loading state
// ---------------------------------------------------------------------------

function DashboardSkeleton() {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* Hero stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>

        {/* Resume card */}
        <ResumeCardSkeleton />

        {/* Two-column: activity + model progress */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <SectionHeader>Recent Activity</SectionHeader>
            <RecentActivitySkeleton />
          </div>
          <div>
            <SectionHeader>Model Progress</SectionHeader>
            <ModelProgressSkeleton />
          </div>
        </div>

        {/* Two-column: rank distribution + reviewer */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <SectionHeader>Rank Distribution</SectionHeader>
            <RankDistributionSkeleton />
          </div>
          <div>
            <SectionHeader>Reviewers</SectionHeader>
            <ReviewerTableSkeleton />
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main dashboard view
// ---------------------------------------------------------------------------

export default function DashboardView() {
  const { data: stats, loading } = useStats()

  if (loading || !stats) {
    return <DashboardSkeleton />
  }

  const completionPct =
    stats.total_queries > 0
      ? Math.round((stats.reviewed_count / stats.total_queries) * 100)
      : 0

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* Section 1: Hero stats */}
        <div
          className="grid grid-cols-2 lg:grid-cols-4 gap-4"
          data-tour="stats-cards"
        >
          <StatCard
            label="Reviewed"
            value={stats.reviewed_count}
            icon={<CheckIcon />}
            color="text-correct"
          />
          <StatCard
            label="Unreviewed"
            value={stats.unreviewed_count}
            icon={<ClockIcon />}
            color="text-highlight"
          />
          <StatCard
            label="Total Feedback"
            value={stats.feedback_count}
            icon={<ChatIcon />}
            color="text-accent"
          />
          <StatCard
            label="Completion"
            value={`${completionPct}%`}
            icon={null}
            ring={{
              reviewed: stats.reviewed_count,
              total: stats.total_queries,
            }}
          />
        </div>

        {/* Section 2: Resume CTA */}
        <ResumeCard stats={stats} />

        {/* Section 3: Activity + Model Progress */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div data-tour="recent-activity">
            <SectionHeader>Recent Activity</SectionHeader>
            <RecentActivityList reviews={stats.recent_reviews ?? []} />
          </div>
          <div>
            <SectionHeader>Model Progress</SectionHeader>
            <ModelProgressBars reviewsByModel={stats.reviews_by_model} />
          </div>
        </div>

        {/* Section 4: Rank Distribution + Reviewers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <SectionHeader>Rank Distribution</SectionHeader>
            <RankDistributionChart
              rankDistribution={stats.rank_distribution}
            />
          </div>
          <div>
            <SectionHeader>Reviewers</SectionHeader>
            <ReviewerTable reviewsByReviewer={stats.reviews_by_reviewer} />
          </div>
        </div>
      </div>
    </div>
  )
}
