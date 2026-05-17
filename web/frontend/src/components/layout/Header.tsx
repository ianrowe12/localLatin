import { useStats } from '../../api/models'
import { apiUrl } from '../../api/client'
import { useApp } from '../../contexts/AppContext'
import { useTour } from '../onboarding/TourProvider'
import { DASHBOARD_TOUR_STEPS, REVIEW_TOUR_STEPS } from '../onboarding/tourSteps'
import { useReviewer } from '../../contexts/ReviewerContext'
import ThemeToggle from '../common/ThemeToggle'

export default function Header() {
  const { activeModel, activeQueryId, currentView, setCurrentView } = useApp()
  const { startTour } = useTour()
  const { reviewerName, isPiAdmin, clearReviewer } = useReviewer()
  const { data: stats } = useStats(isPiAdmin)

  const handleHelp = () => {
    startTour(currentView === 'dashboard' ? DASHBOARD_TOUR_STEPS : REVIEW_TOUR_STEPS)
  }

  const reviewed = stats?.reviewed_count ?? 0
  const total = stats?.total_queries ?? 0
  const progressPercent = total > 0 ? (reviewed / total) * 100 : 0

  return (
    <header
      className="h-[4.25rem] px-4 flex items-center justify-between
                 bg-white/80 dark:bg-surface-800/80 backdrop-blur-sm
                 border-b border-stone-200/60 dark:border-stone-700/60
                 shadow-glass dark:shadow-glass-dark"
    >
      {/* Left: Title */}
      <div className="flex flex-col flex-shrink-0">
        <div className="flex items-baseline gap-2">
          <h1 className="font-latin text-xl font-semibold text-stone-800 dark:text-stone-100">
            LocalLatin
          </h1>
          <span className="font-ui text-sm text-stone-500 dark:text-stone-400">
            Manuscript Review Tool
          </span>
        </div>
        <p className="font-ui text-xs text-stone-400 dark:text-stone-500">
          Review model predictions for Latin manuscript source identification
        </p>
      </div>

      {isPiAdmin ? (
        <div className="flex items-center gap-3 font-ui text-sm text-stone-600 dark:text-stone-300">
          <span>
            {reviewed}
            <span className="text-stone-400 dark:text-stone-500"> / </span>
            {total}
            <span className="ml-1 text-stone-400 dark:text-stone-500">
              reviewed
            </span>
          </span>
          <div className="w-32 h-1.5 rounded-full bg-stone-200 dark:bg-stone-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      ) : (
        <div />
      )}

      {/* Right: Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {currentView !== (isPiAdmin ? 'dashboard' : 'review') && (
          <button
            type="button"
            onClick={() => setCurrentView(isPiAdmin ? 'dashboard' : 'review')}
            className="w-8 h-8 rounded-full flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-600 dark:text-stone-300"
            aria-label={isPiAdmin ? 'Go to dashboard' : 'Go to review'}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
          </button>
        )}
        <button
          type="button"
          onClick={() => setCurrentView('files')}
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors text-stone-600 dark:text-stone-300 ${
            currentView === 'files'
              ? 'bg-stone-200 dark:bg-stone-700'
              : 'hover:bg-stone-200 dark:hover:bg-stone-700'
          }`}
          aria-label="Browse manuscript files"
          title="Browse manuscript files"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="7" height="7" />
            <rect x="14" y="3" width="7" height="7" />
            <rect x="3" y="14" width="7" height="7" />
            <rect x="14" y="14" width="7" height="7" />
          </svg>
        </button>
        <ThemeToggle />

        {/* Reviewer badge */}
        {reviewerName && (
          <button
            type="button"
            onClick={clearReviewer}
            className="flex items-center gap-1.5 px-2 py-1 rounded-full
                       bg-accent/10 dark:bg-accent/15 hover:bg-accent/20 transition-colors"
            title="Sign out"
          >
            <span
              className="w-5 h-5 rounded-full bg-accent text-white
                         flex items-center justify-center text-[10px] font-bold font-ui
                         uppercase leading-none"
              aria-hidden="true"
            >
              {reviewerName[0]}
            </span>
            <span className="text-xs font-ui font-medium text-accent dark:text-accent-light">
              {reviewerName}
            </span>
          </button>
        )}

        <button
          type="button"
          onClick={handleHelp}
          className="w-8 h-8 rounded-full flex items-center justify-center
                     hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                     text-stone-600 dark:text-stone-300"
          aria-label="Help"
        >
          <span className="font-ui text-sm font-semibold">?</span>
        </button>
        {isPiAdmin && activeQueryId !== null && activeModel && (
          <button
            type="button"
            onClick={() => {
              const model = encodeURIComponent(activeModel)
              window.open(
                apiUrl(`/api/packets/review/${activeQueryId}?model=${model}&top_k=10`),
                '_blank',
              )
            }}
            className="w-8 h-8 rounded-full flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-600 dark:text-stone-300"
            aria-label="Download review packet"
            title="Download review packet"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="8" y1="13" x2="16" y2="13" />
              <line x1="8" y1="17" x2="16" y2="17" />
            </svg>
          </button>
        )}
        {isPiAdmin && (
          <button
            type="button"
            onClick={() => window.open(apiUrl('/api/feedback/export'), '_blank')}
            className="w-8 h-8 rounded-full flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-600 dark:text-stone-300"
            aria-label="Export data"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>
        )}
      </div>
    </header>
  )
}
