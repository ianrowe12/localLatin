import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ReviewerProvider } from '../../contexts/ReviewerContext'
import AccountApprovalPanel from './AccountApprovalPanel'

// Issue #265: the per-account role control.

function account(
  id: number,
  username: string,
  display_name: string,
  role: 'reviewer' | 'pi_admin',
) {
  return {
    id,
    username,
    display_name,
    role,
    approval_status: 'approved' as const,
    must_change_password: false,
    is_active: true,
    created_at: '2026-11-01 10:00:00',
    updated_at: '2026-11-01 10:00:00',
    last_login_at: null,
    approved_at: '2026-11-01 10:00:00',
    approved_by_account_id: 1,
    rejected_at: null,
    approval_note: '',
  }
}

const ME = account(1, 'pi', 'PI Scholar', 'pi_admin')
const NOAH = account(2, 'noah', 'Noah Adler', 'reviewer')
const CO_PI = account(3, 'copi', 'Co PI', 'pi_admin')

let roleStatus = 200
let roleBody: unknown = null
const roleCalls: { url: string; body: unknown }[] = []

function jsonResponse(data: unknown, status: number): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  roleStatus = 200
  roleBody = null
  roleCalls.length = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/auth/me')) return jsonResponse(ME, 200)
      if (/\/api\/auth\/accounts\/\d+\/role$/.test(url)) {
        const body = JSON.parse(String(init?.body))
        roleCalls.push({ url, body })
        const id = Number(url.match(/accounts\/(\d+)\/role/)?.[1])
        const target = [ME, NOAH, CO_PI].find((item) => item.id === id)
        return jsonResponse(roleBody ?? { ...target, role: body.role }, roleStatus)
      }
      if (url.includes('/api/auth/accounts?status=pending')) return jsonResponse([], 200)
      if (url.includes('/api/auth/accounts?status=approved')) {
        return jsonResponse([ME, NOAH, CO_PI], 200)
      }
      throw new Error(`unexpected fetch: ${url}`)
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function renderPanel() {
  render(
    <ReviewerProvider>
      <AccountApprovalPanel />
    </ReviewerProvider>,
  )
  const roles = await screen.findByTestId('account-roles')
  // The own-row guard depends on the signed-in user, so wait for it.
  await waitFor(() =>
    expect((within(roles).getByTestId('role-row-1')).textContent).toContain('(you)'),
  )
  return roles
}

describe('AccountApprovalPanel role control', () => {
  it('disables the control on your own row and says why', async () => {
    const roles = await renderPanel()
    const own = within(roles).getByTestId('role-row-1')
    expect((within(own).getByRole('button', { name: 'Make reviewer' }) as HTMLButtonElement).disabled).toBe(true)
    expect(own.textContent).toContain(
      'You cannot change your own role. Another PI/admin must do it.',
    )
    const noah = within(roles).getByTestId('role-row-2')
    expect((within(noah).getByRole('button', { name: 'Make PI/admin' }) as HTMLButtonElement).disabled).toBe(false)
    const coPi = within(roles).getByTestId('role-row-3')
    expect((within(coPi).getByRole('button', { name: 'Make reviewer' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('asks for confirmation, naming the privileges, before posting anything', async () => {
    const user = userEvent.setup()
    const roles = await renderPanel()
    const noah = within(roles).getByTestId('role-row-2')

    await user.click(within(noah).getByRole('button', { name: 'Make PI/admin' }))
    expect(noah.textContent).toContain(
      'Make Noah Adler a PI/admin? PI/admins can approve accounts, reset passwords and export all feedback.',
    )
    expect(roleCalls).toHaveLength(0)

    await user.click(within(noah).getByRole('button', { name: 'Cancel' }))
    expect(
      within(noah).queryByRole('button', { name: 'Confirm: make PI/admin' }),
    ).toBeNull()
    expect(roleCalls).toHaveLength(0)
  })

  it('promotes on confirm and shows the new role', async () => {
    const user = userEvent.setup()
    const roles = await renderPanel()
    const noah = within(roles).getByTestId('role-row-2')

    await user.click(within(noah).getByRole('button', { name: 'Make PI/admin' }))
    await user.click(within(noah).getByRole('button', { name: 'Confirm: make PI/admin' }))

    await waitFor(() => expect(roleCalls).toHaveLength(1))
    expect(roleCalls[0].url).toMatch(/\/api\/auth\/accounts\/2\/role$/)
    expect(roleCalls[0].body).toEqual({ role: 'pi_admin' })
    expect((await within(roles).findByRole('status')).textContent).toContain(
      'Noah Adler is now a PI/admin.',
    )
    expect(noah.textContent).toContain('PI/admin')
    expect((within(noah).getByRole('button', { name: 'Make reviewer' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('demotes with its own warning', async () => {
    const user = userEvent.setup()
    const roles = await renderPanel()
    const coPi = within(roles).getByTestId('role-row-3')

    await user.click(within(coPi).getByRole('button', { name: 'Make reviewer' }))
    expect(coPi.textContent).toContain(
      'Make Co PI a reviewer? They lose PI/admin access on their next request.',
    )
    await user.click(within(coPi).getByRole('button', { name: 'Confirm: make reviewer' }))
    await waitFor(() => expect(roleCalls).toHaveLength(1))
    expect(roleCalls[0].body).toEqual({ role: 'reviewer' })
    await waitFor(() =>
      expect((within(coPi).getByRole('button', { name: 'Make PI/admin' }) as HTMLButtonElement).disabled).toBe(false),
    )
  })

  it.each([
    [409, 'This is the last approved PI/admin account. Make another account a PI/admin before changing this one.'],
    [400, 'You cannot change your own role. Another PI/admin must do it.'],
  ])('shows the server text on a %i and keeps the role', async (status, detail) => {
    roleStatus = status
    roleBody = { detail }
    const user = userEvent.setup()
    const roles = await renderPanel()
    const coPi = within(roles).getByTestId('role-row-3')

    await user.click(within(coPi).getByRole('button', { name: 'Make reviewer' }))
    await user.click(within(coPi).getByRole('button', { name: 'Confirm: make reviewer' }))

    expect((await within(roles).findByRole('alert')).textContent).toContain(detail)
    expect(within(roles).queryByRole('status')).toBeNull()
    // Unchanged: still a PI/admin, and the confirm step is still open to retry or cancel.
    expect(within(coPi).getByRole('button', { name: 'Make reviewer' })).toBeTruthy()
    expect((within(coPi).getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(false)
  })
})
