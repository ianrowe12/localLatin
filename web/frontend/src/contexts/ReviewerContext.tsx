import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

export interface ReviewerContextValue {
  reviewerName: string | null
  setReviewerName: (name: string) => void
  clearReviewer: () => void
}

const ReviewerContext = createContext<ReviewerContextValue | null>(null)

const REVIEWER_KEY = 'locallatin-reviewer'

export function ReviewerProvider({ children }: { children: ReactNode }) {
  const [reviewerName, setReviewerNameState] = useState<string | null>(() => {
    return localStorage.getItem(REVIEWER_KEY)
  })

  useEffect(() => {
    if (reviewerName === null) {
      localStorage.removeItem(REVIEWER_KEY)
    } else {
      localStorage.setItem(REVIEWER_KEY, reviewerName)
    }
  }, [reviewerName])

  const setReviewerName = useCallback((name: string) => {
    const trimmed = name.trim()
    if (trimmed) setReviewerNameState(trimmed)
  }, [])

  const clearReviewer = useCallback(() => {
    setReviewerNameState(null)
  }, [])

  const value: ReviewerContextValue = {
    reviewerName,
    setReviewerName,
    clearReviewer,
  }

  return (
    <ReviewerContext.Provider value={value}>{children}</ReviewerContext.Provider>
  )
}

export function useReviewer(): ReviewerContextValue {
  const ctx = useContext(ReviewerContext)
  if (!ctx) {
    throw new Error('useReviewer must be used within a ReviewerProvider')
  }
  return ctx
}
