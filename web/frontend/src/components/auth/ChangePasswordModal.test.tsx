import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import ChangePasswordModal from './ChangePasswordModal'

const CURRENT_USER = {
  id: 2,
  username: 'reviewer',
  display_name: 'External Reviewer',
  role: 'reviewer' as const,
  approval_status: 'approved' as const,
  must_change_password: true,
}

let changeStatus = 200
let changeBody: unknown = { ...CURRENT_USER, must_change_password: false }
const changeCalls: unknown[] = []

function jsonResponse(data: unknown, status: number): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  changeStatus = 200
  changeBody = { ...CURRENT_USER, must_change_password: false }
  changeCalls.length = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/auth/me')) return jsonResponse(CURRENT_USER, 200)
      if (url.includes('/api/auth/change_password')) {
        changeCalls.push(JSON.parse(String(init?.body)))
        return jsonResponse(changeBody, changeStatus)
      }
      throw new Error(`unexpected fetch: ${url}`)
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderModal(forced = false) {
  return render(
    <ReviewerProvider>
      <ChangePasswordModal forced={forced} />
    </ReviewerProvider>,
  )
}

describe('ChangePasswordModal', () => {
  it('posts the change once the new password is long enough and confirmed', async () => {
    const user = userEvent.setup()
    renderModal()

    await user.type(screen.getByLabelText('Current password'), 'old passphrase')
    await user.type(screen.getByLabelText('New password'), 'brand new passphrase')
    const submit = screen.getByRole('button', {
      name: 'Change password',
    }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)

    await user.type(
      screen.getByLabelText('Confirm new password'),
      'brand new passphrase',
    )
    expect(submit.disabled).toBe(false)

    await user.click(submit)
    await waitFor(() => expect(changeCalls).toHaveLength(1))
    expect(changeCalls[0]).toEqual({
      current_password: 'old passphrase',
      new_password: 'brand new passphrase',
    })
  })

  it('keeps submit disabled for a password under the twelve character floor', async () => {
    const user = userEvent.setup()
    renderModal()

    await user.type(screen.getByLabelText('Current password'), 'old passphrase')
    await user.type(screen.getByLabelText('New password'), 'short123')
    await user.type(screen.getByLabelText('Confirm new password'), 'short123')

    expect(screen.getByText('Use at least 12 characters.')).toBeTruthy()
    const submit = screen.getByRole('button', {
      name: 'Change password',
    }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)
    expect(changeCalls).toHaveLength(0)
  })

  it('surfaces a rejected current password from the server', async () => {
    changeStatus = 403
    changeBody = { detail: 'Current password is incorrect' }
    const user = userEvent.setup()
    renderModal(true)

    await user.type(screen.getByLabelText('Temporary password'), 'wrong password')
    await user.type(screen.getByLabelText('New password'), 'brand new passphrase')
    await user.type(
      screen.getByLabelText('Confirm new password'),
      'brand new passphrase',
    )
    await user.click(screen.getByRole('button', { name: 'Change password' }))

    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe(
        'Current password is incorrect',
      ),
    )
  })
})
