import { useState } from 'react'
import { useApp } from '../../contexts/AppContext'
import Header from './Header'
import LeftSidebar from './LeftSidebar'
import CenterArea from './CenterArea'
import RightSidebar from './RightSidebar'
import DashboardView from '../dashboard/DashboardView'
import FileBrowser from '../files/FileBrowser'
import LabelledKeysPanel from '../keys/LabelledKeysPanel'
import { useReviewer } from '../../contexts/ReviewerContext'
import AdminDecisionPanel from '../feedback/AdminDecisionPanel'

export default function AppShell() {
  const { currentView } = useApp()
  const { isPiAdmin } = useReviewer()
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-surface-50 dark:bg-surface-900">
      <Header />
      {currentView === 'dashboard' && isPiAdmin ? (
        <DashboardView />
      ) : currentView === 'files' ? (
        <FileBrowser />
      ) : currentView === 'keys' ? (
        <LabelledKeysPanel />
      ) : (
        <div className="flex-1 flex overflow-hidden">
          <LeftSidebar
            isOpen={leftOpen}
            onToggle={() => setLeftOpen((prev) => !prev)}
          />
          <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
            <CenterArea />
            <AdminDecisionPanel />
          </div>
          <RightSidebar
            isOpen={rightOpen}
            onToggle={() => setRightOpen((prev) => !prev)}
          />
        </div>
      )}
    </div>
  )
}
