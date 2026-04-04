import { useCallback, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useTour } from './TourProvider'
import TourTooltip from './TourTooltip'

interface Rect {
  x: number
  y: number
  width: number
  height: number
}

const PADDING = 8
const BORDER_RADIUS = 8

export default function TourOverlay() {
  const { isActive, currentStep, currentStepIndex, totalSteps, nextStep, prevStep, stopTour } =
    useTour()

  const [targetRect, setTargetRect] = useState<Rect | null>(null)
  const [skipTimer, setSkipTimer] = useState<ReturnType<typeof setTimeout> | null>(null)

  const measureTarget = useCallback(() => {
    if (!currentStep) {
      setTargetRect(null)
      return
    }
    const el = document.querySelector(`[data-tour="${currentStep.target}"]`)
    if (el) {
      const rect = el.getBoundingClientRect()
      setTargetRect({ x: rect.x, y: rect.y, width: rect.width, height: rect.height })
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })

      // Clear any pending skip timer since we found the element
      if (skipTimer) {
        clearTimeout(skipTimer)
        setSkipTimer(null)
      }
    } else {
      setTargetRect(null)
      // Target not found -- skip to next step after 500ms
      const timer = setTimeout(() => {
        nextStep()
      }, 500)
      setSkipTimer(timer)
    }
  }, [currentStep, nextStep, skipTimer])

  // Measure on step change
  useEffect(() => {
    if (!isActive) return
    measureTarget()
  }, [isActive, currentStepIndex]) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-measure on window resize and scroll
  useEffect(() => {
    if (!isActive) return

    const handleUpdate = () => {
      if (!currentStep) return
      const el = document.querySelector(`[data-tour="${currentStep.target}"]`)
      if (el) {
        const rect = el.getBoundingClientRect()
        setTargetRect({ x: rect.x, y: rect.y, width: rect.width, height: rect.height })
      }
    }

    window.addEventListener('resize', handleUpdate)
    window.addEventListener('scroll', handleUpdate, true)
    return () => {
      window.removeEventListener('resize', handleUpdate)
      window.removeEventListener('scroll', handleUpdate, true)
    }
  }, [isActive, currentStep])

  // Cleanup skip timer on unmount
  useEffect(() => {
    return () => {
      if (skipTimer) clearTimeout(skipTimer)
    }
  }, [skipTimer])

  if (!isActive || !currentStep) return null

  // Build the SVG mask cutout path
  const cutoutX = targetRect ? targetRect.x - PADDING : 0
  const cutoutY = targetRect ? targetRect.y - PADDING : 0
  const cutoutW = targetRect ? targetRect.width + PADDING * 2 : 0
  const cutoutH = targetRect ? targetRect.height + PADDING * 2 : 0

  return (
    <>
      {/* SVG overlay with spotlight mask */}
      <motion.svg
        className="fixed inset-0 w-full h-full z-[9998] pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.25 }}
        aria-hidden="true"
      >
        <defs>
          <mask id="tour-spotlight-mask">
            <rect width="100%" height="100%" fill="white" />
            {targetRect && (
              <motion.rect
                initial={false}
                animate={{
                  x: cutoutX,
                  y: cutoutY,
                  width: cutoutW,
                  height: cutoutH,
                }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                rx={BORDER_RADIUS}
                fill="black"
              />
            )}
          </mask>
        </defs>
        <rect
          width="100%"
          height="100%"
          fill="rgba(0,0,0,0.5)"
          mask="url(#tour-spotlight-mask)"
        />
      </motion.svg>

      {/* Transparent click-catcher -- click outside spotlight to dismiss */}
      <div
        className="fixed inset-0 z-[9998]"
        onClick={stopTour}
        aria-hidden="true"
      />

      {/* Tooltip */}
      {targetRect && (
        <TourTooltip
          step={currentStep}
          stepIndex={currentStepIndex}
          totalSteps={totalSteps}
          onNext={nextStep}
          onPrev={prevStep}
          onSkip={stopTour}
        />
      )}
    </>
  )
}
