import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import type { TourStep } from './tourSteps'
import TourOverlay from './TourOverlay'

interface TourContextValue {
  isActive: boolean
  currentStepIndex: number
  currentStep: TourStep | null
  totalSteps: number
  startTour: (steps: TourStep[]) => void
  stopTour: () => void
  nextStep: () => void
  prevStep: () => void
}

const TourContext = createContext<TourContextValue | null>(null)

export function useTour(): TourContextValue {
  const ctx = useContext(TourContext)
  if (!ctx) {
    throw new Error('useTour must be used within a TourProvider')
  }
  return ctx
}

interface TourProviderProps {
  children: ReactNode
}

export function TourProvider({ children }: TourProviderProps) {
  const [steps, setSteps] = useState<TourStep[]>([])
  const [stepIndex, setStepIndex] = useState(0)
  const [isActive, setIsActive] = useState(false)

  const startTour = useCallback((tourSteps: TourStep[]) => {
    if (tourSteps.length === 0) return
    setSteps(tourSteps)
    setStepIndex(0)
    setIsActive(true)
  }, [])

  const stopTour = useCallback(() => {
    // Persist completion to localStorage using the first step's target as tourId
    if (steps.length > 0) {
      const tourId = steps[0].target
      try {
        localStorage.setItem(`locallatin-tour-completed-${tourId}`, 'true')
      } catch {
        // localStorage may be unavailable; silently ignore
      }
    }
    setIsActive(false)
    setSteps([])
    setStepIndex(0)
  }, [steps])

  const nextStep = useCallback(() => {
    setStepIndex((prev) => {
      const next = prev + 1
      if (next >= steps.length) {
        // Past last step -- stop tour
        // Use setTimeout to avoid state update during render
        setTimeout(() => stopTour(), 0)
        return prev
      }
      return next
    })
  }, [steps.length, stopTour])

  const prevStep = useCallback(() => {
    setStepIndex((prev) => Math.max(0, prev - 1))
  }, [])

  const currentStep = isActive && steps.length > 0 ? steps[stepIndex] ?? null : null

  const value = useMemo<TourContextValue>(
    () => ({
      isActive,
      currentStepIndex: stepIndex,
      currentStep,
      totalSteps: steps.length,
      startTour,
      stopTour,
      nextStep,
      prevStep,
    }),
    [isActive, stepIndex, currentStep, steps.length, startTour, stopTour, nextStep, prevStep],
  )

  return (
    <TourContext.Provider value={value}>
      {children}
      <TourOverlay />
    </TourContext.Provider>
  )
}
