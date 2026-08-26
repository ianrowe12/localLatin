import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProvider } from '../../contexts/AppContext'
import { FeedbackProvider } from '../../contexts/FeedbackContext'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import { getReviewTourSteps } from '../onboarding/tourSteps'
import RightSidebar from './RightSidebar'

const MODEL = 'google_mt5-base'

let role: 'reviewer' | 'pi_admin' = 'reviewer'

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function installFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/auth/me')) {
        return jsonResponse({
          id: 1,
          username: role,
          display_name: role,
          role,
          approval_status: 'approved',
        })
      }
      if (url.includes('/api/models')) {
        return jsonResponse([
          {
            slug: MODEL,
            display_name: 'mT5-base',
            layer: 4,
            pooling: 'mean',
            prediction_count: 2238,
            available_variants: ['raw', 'abtt', 'sif', 'sif_abtt'],
            default_variant: 'sif_abtt',
          },
        ])
      }
      return jsonResponse({})
    }),
  )
}

function renderSidebar() {
  return render(
    <ReviewerProvider>
      <AppProvider>
        <FeedbackProvider>
          <RightSidebar isOpen onToggle={() => {}} />
        </FeedbackProvider>
      </AppProvider>
    </ReviewerProvider>,
  )
}

beforeEach(() => {
  role = 'reviewer'
  installFetch()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('single-pipeline UI (issue #94)', () => {
  it('shows no post-processing picker to a reviewer', async () => {
    renderSidebar()
    await waitFor(() => {
      expect(screen.getByLabelText('Select model')).toBeTruthy()
    })
    expect(screen.queryByRole('radiogroup', { name: 'Post-processing variant' })).toBeNull()
    expect(document.querySelector('[data-tour="variant-selector"]')).toBeNull()
    expect(screen.queryByText('Post-Processing')).toBeNull()
    expect(screen.queryByRole('radio', { name: /SIF/ })).toBeNull()
  })

  it('shows no post-processing picker to a PI/admin either', async () => {
    role = 'pi_admin'
    renderSidebar()
    await waitFor(() => {
      expect(screen.getByLabelText('Select model')).toBeTruthy()
    })
    expect(screen.queryByRole('radiogroup', { name: 'Post-processing variant' })).toBeNull()
    expect(document.querySelector('[data-tour="variant-selector"]')).toBeNull()
    expect(screen.queryByText('Post-Processing')).toBeNull()
  })

  it('drops the variant step from the tour for both roles', () => {
    for (const isPiAdmin of [false, true]) {
      const targets = getReviewTourSteps(isPiAdmin).map((step) => step.target)
      expect(targets).not.toContain('variant-selector')
      // ...and the tour still walks the surviving controls in order.
      expect(targets).toContain('model-selector')
      expect(targets).toContain('predictions')
    }
  })
})
