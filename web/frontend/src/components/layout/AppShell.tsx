import { useState } from 'react'
import Header from './Header'
import LeftSidebar from './LeftSidebar'
import CenterArea from './CenterArea'
import RightSidebar from './RightSidebar'

export default function AppShell() {
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-surface-50 dark:bg-surface-900">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <LeftSidebar
          isOpen={leftOpen}
          onToggle={() => setLeftOpen((prev) => !prev)}
        />
        <CenterArea />
        <RightSidebar
          isOpen={rightOpen}
          onToggle={() => setRightOpen((prev) => !prev)}
        />
      </div>
    </div>
  )
}
