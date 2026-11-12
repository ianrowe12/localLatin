import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import ReviewerLoginModal from './ReviewerLoginModal'

// Issue #274: a display name is written verbatim into the CCL key export, so
// one a spreadsheet would run is refused inline before the request is sent.

const registerCalls: unknown[] = []

function jsonResponse(data: unknown, status: number): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  registerCalls.length = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/auth/me')) {
        return jsonResponse({ detail: 'Not signed in' }, 401)
      }
      if (url.includes('/api/auth/register')) {
        registerCalls.push(JSON.parse(String(init?.body)))
        return jsonResponse(
          {
            status: 'pending_approval',
            message: 'Account request submitted.',
            account: {
              id: 5,
              username: 'newcomer',
              display_name: 'Ann Example',
              role: 'reviewer',
              approval_status: 'pending',
              must_change_password: false,
            },
          },
          201,
        )
      }
      throw new Error(`unexpected fetch: ${url}`)
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function fillRegistration(displayName: string) {
  const user = userEvent.setup()
  render(
    <ReviewerProvider>
      <ReviewerLoginModal />
    </ReviewerProvider>,
  )
  await user.click(screen.getByRole('button', { name: 'Register' }))
  await user.type(screen.getByLabelText('Username'), 'newcomer')
  await user.type(screen.getByLabelText('Password'), 'correct horse battery staple')
  await user.type(
    screen.getByLabelText('Display name or initials for attribution'),
    displayName,
  )
  return user
}

describe('ReviewerLoginModal display names (issue #274)', () => {
  it.each(['=1+1', '@Ann Example', 'Ann | Example'])(
    'refuses %s inline and sends nothing',
    async (displayName) => {
      const user = await fillRegistration(displayName)
      expect(screen.getByTestId('register-display-name-invalid').textContent).toBe(
        'A display name cannot start with =, +, - or @ (a spreadsheet would run it as a formula) and cannot contain | (the export uses it to separate names).',
      )
      const submit = screen.getByRole('button', { name: 'Create account' })
      expect((submit as HTMLButtonElement).disabled).toBe(true)
      await user.click(submit)
      expect(registerCalls).toHaveLength(0)
    },
  )

  it('lets an ordinary name through', async () => {
    const user = await fillRegistration('Ann-Marie Example')
    expect(screen.queryByTestId('register-display-name-invalid')).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    await waitFor(() => expect(registerCalls).toHaveLength(1))
  })
})
