import { useStats } from '../../api/models'
import { useApp } from '../../contexts/AppContext'
import { useTour } from '../onboarding/TourProvider'
import { DASHBOARD_TOUR_STEPS, REVIEW_TOUR_STEPS } from '../onboarding/tourSteps'
import ThemeToggle from '../common/ThemeToggle'

export default function Header() {
  const { data: stats } = useStats()
  const { currentView, setCurrentView } = useApp()
  const { startTour } = useTour()

  const handleHelp = () => {
    startTour(currentView === 'dashboard' ? DASHBOARD_TOUR_STEPS : REVIEW_TOUR_STEPS)
  }

  const reviewed = stats?.reviewed_count ?? 0
  const total = stats?.total_queries ?? 0
  const progressPercent = total > 0 ? (reviewed / total) * 100 : 0

  return (
    <header
      className="h-14 px-4 flex items-center justify-between
                 bg-white/80 dark:bg-surface-800/80 backdrop-blur-sm
                 border-b border-stone-200/60 dark:border-stone-700/60
                 shadow-glass dark:shadow-glass-dark"
    >
      {/* Left: Title */}
      <div className="flex items-baseline gap-2 flex-shrink-0">
        <h1 className="font-latin text-xl font-semibold text-stone-800 dark:text-stone-100">
          LocalLatin
        </h1>
        <span className="font-ui text-sm text-stone-500 dark:text-stone-400">
          Manuscript Review Tool
        </span>
      </div>

      {/* Center: Stats progress */}
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

      {/* Right: Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {currentView === 'review' && (
          <button
            type="button"
            onClick={() => setCurrentView('dashboard')}
            className="w-8 h-8 rounded-full flex items-center justify-center
                       hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                       text-stone-600 dark:text-stone-300"
            aria-label="Go to dashboard"
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
        <ThemeToggle />
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
        <button
          type="button"
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
      </div>
    </header>
  )
}
