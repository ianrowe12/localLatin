import { apiFetch } from './client'

export interface AuthUser {
  id: number
  username: string
  display_name: string
  role: 'reviewer' | 'pi_admin'
}

export interface RegisterPayload {
  username: string
  display_name: string
  password: string
  admin_code?: string
}

export interface SignInPayload {
  username: string
  password: string
}

export function getCurrentUser(): Promise<AuthUser> {
  return apiFetch<AuthUser>('/api/auth/me')
}

export function register(payload: RegisterPayload): Promise<AuthUser> {
  return apiFetch<AuthUser>('/api/auth/register', {
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
  await fetch('/api/auth/signout', {
    method: 'POST',
    credentials: 'include',
  })
}
