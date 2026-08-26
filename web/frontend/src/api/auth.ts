import { apiFetch, apiUrl } from './client'

export interface AuthUser {
  id: number
  username: string
  display_name: string
  role: 'reviewer' | 'pi_admin'
  approval_status: 'pending' | 'approved' | 'rejected'
  must_change_password: boolean
}

export interface RegisterPayload {
  username: string
  display_name: string
  password: string
  admin_code?: string
}

export interface RegistrationResponse {
  status: 'approved' | 'pending_approval'
  message: string
  account: AuthUser
}

export interface AccountPublic extends AuthUser {
  is_active: boolean
  created_at: string
  updated_at: string
  last_login_at: string | null
  approved_at: string | null
  approved_by_account_id: number | null
  rejected_at: string | null
  approval_note: string
}

export interface AccountCreatePayload {
  username: string
  display_name: string
  role: 'reviewer' | 'pi_admin'
  password?: string
  approval_note?: string
}

export interface AccountCreateResponse {
  account: AccountPublic
  temporary_password: string | null
}

export interface SignInPayload {
  username: string
  password: string
}

export function getCurrentUser(): Promise<AuthUser> {
  return apiFetch<AuthUser>('/api/auth/me')
}

export function register(payload: RegisterPayload): Promise<RegistrationResponse> {
  return apiFetch<RegistrationResponse>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function signIn(payload: SignInPayload): Promise<AuthUser> {
  return apiFetch<AuthUser>('/api/auth/signin', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function signOut(): Promise<void> {
  await fetch(apiUrl('/api/auth/signout'), {
    method: 'POST',
    credentials: 'include',
  })
}

export function listAccounts(status = 'pending'): Promise<AccountPublic[]> {
  const params = new URLSearchParams({ status })
  return apiFetch<AccountPublic[]>(`/api/auth/accounts?${params}`)
}

export function createAccount(
  payload: AccountCreatePayload,
): Promise<AccountCreateResponse> {
  return apiFetch<AccountCreateResponse>('/api/auth/accounts', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/** Mirrors MIN_PASSWORD_LENGTH in web/models.py (same floor as registration). */
export const MIN_PASSWORD_LENGTH = 12

export interface PasswordChangePayload {
  current_password: string
  new_password: string
}

export interface PasswordResetResponse {
  account: AccountPublic
  /** Returned once, never stored in plaintext. */
  temporary_password: string
}

export function changePassword(payload: PasswordChangePayload): Promise<AuthUser> {
  return apiFetch<AuthUser>('/api/auth/change_password', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function resetAccountPassword(id: number): Promise<PasswordResetResponse> {
  return apiFetch<PasswordResetResponse>(
    `/api/auth/accounts/${id}/reset_password`,
    { method: 'POST' },
  )
}

export function approveAccount(id: number): Promise<AccountPublic> {
  return apiFetch<AccountPublic>(`/api/auth/accounts/${id}/approve`, {
    method: 'POST',
  })
}

export function rejectAccount(id: number): Promise<AccountPublic> {
  return apiFetch<AccountPublic>(`/api/auth/accounts/${id}/reject`, {
    method: 'POST',
  })
}
