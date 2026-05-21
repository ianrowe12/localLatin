import { useCallback, useEffect, useState } from 'react'
import {
  approveAccount,
  createAccount,
  listAccounts,
  rejectAccount,
  type AccountCreateResponse,
  type AccountPublic,
} from '../../api/auth'

function formatCreatedAt(value: string): string {
  const date = new Date(`${value.replace(' ', 'T')}Z`)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export default function AccountApprovalPanel() {
  const [accounts, setAccounts] = useState<AccountPublic[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actingId, setActingId] = useState<number | null>(null)
  const [creating, setCreating] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [newDisplayName, setNewDisplayName] = useState('')
  const [newRole, setNewRole] = useState<'reviewer' | 'pi_admin'>('reviewer')
  const [newPassword, setNewPassword] = useState('')
  const [createdAccount, setCreatedAccount] =
    useState<AccountCreateResponse | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setAccounts(await listAccounts('pending'))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load accounts')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const decide = async (accountId: number, action: 'approve' | 'reject') => {
    setActingId(accountId)
    setError(null)
    try {
      if (action === 'approve') {
        await approveAccount(accountId)
      } else {
        await rejectAccount(accountId)
      }
      setAccounts((current) =>
        current.filter((account) => account.id !== accountId),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Account update failed')
    } finally {
      setActingId(null)
    }
  }

  const canCreate =
    newUsername.trim().length >= 2 &&
    newDisplayName.trim().length > 0 &&
    (newPassword.length === 0 || newPassword.length >= 12)

  const createApprovedAccount = async () => {
    if (!canCreate || creating) return
    setCreating(true)
    setError(null)
    setCreatedAccount(null)
    try {
      const result = await createAccount({
        username: newUsername,
        display_name: newDisplayName,
        role: newRole,
        password: newPassword || undefined,
      })
      setCreatedAccount(result)
      setNewUsername('')
      setNewDisplayName('')
      setNewRole('reviewer')
      setNewPassword('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Account creation failed')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div
      className="rounded-lg border border-stone-200/70 dark:border-stone-700/70
                 bg-white dark:bg-surface-800 shadow-sm overflow-hidden"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-stone-100 dark:border-stone-700/70">
        <div>
          <h3 className="font-ui text-sm font-semibold text-stone-700 dark:text-stone-200">
            Account Approval
          </h3>
          <p className="font-ui text-xs text-stone-500 dark:text-stone-400">
            Pending reviewer signup requests
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={loading}
          className="h-8 px-3 rounded-md text-xs font-ui font-medium
                     bg-stone-100 dark:bg-surface-900 text-stone-600 dark:text-stone-300
                     hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                     disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="px-4 py-3 font-ui text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      )}

      <div className="px-4 py-4 border-b border-stone-100 dark:border-stone-700/70">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-3">
          <input
            type="text"
            value={newUsername}
            onChange={(event) => setNewUsername(event.target.value)}
            placeholder="Email or username"
            className="h-9 px-3 rounded-md border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Email or username for new account"
          />
          <input
            type="text"
            value={newDisplayName}
            onChange={(event) => setNewDisplayName(event.target.value)}
            placeholder="Display name"
            className="h-9 px-3 rounded-md border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Display name for new account"
          />
          <select
            value={newRole}
            onChange={(event) =>
              setNewRole(event.target.value as 'reviewer' | 'pi_admin')
            }
            className="h-9 px-3 rounded-md border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Role for new account"
          >
            <option value="reviewer">Reviewer</option>
            <option value="pi_admin">PI/admin</option>
          </select>
        </div>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3">
          <input
            type="text"
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
            placeholder="Temporary password (optional)"
            className="h-9 px-3 rounded-md border border-stone-200 dark:border-stone-700
                       bg-white dark:bg-surface-900 text-sm font-ui text-stone-800 dark:text-stone-200
                       placeholder:text-stone-400 dark:placeholder:text-stone-500
                       focus:outline-none focus:ring-2 focus:ring-accent/30"
            aria-label="Temporary password for new account"
          />
          <button
            type="button"
            onClick={() => void createApprovedAccount()}
            disabled={!canCreate || creating}
            className="h-9 px-4 rounded-md font-ui text-xs font-medium
                       text-white bg-accent hover:bg-accent-dark transition-colors
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? 'Creating...' : 'Create Approved Account'}
          </button>
        </div>

        {createdAccount && (
          <div className="mt-3 rounded-md bg-accent/10 dark:bg-accent/15 px-3 py-2 font-ui text-sm text-stone-700 dark:text-stone-200">
            Created {createdAccount.account.display_name} as{' '}
            {createdAccount.account.role === 'pi_admin' ? 'PI/admin' : 'reviewer'}
            {createdAccount.temporary_password && (
              <>
                {' '}
                with temporary password{' '}
                <span className="font-mono text-xs text-accent dark:text-accent-light">
                  {createdAccount.temporary_password}
                </span>
              </>
            )}
            .
          </div>
        )}
      </div>

      {loading ? (
        <div className="px-4 py-5 font-ui text-sm text-stone-500 dark:text-stone-400">
          Loading account requests...
        </div>
      ) : accounts.length === 0 ? (
        <div className="px-4 py-5 font-ui text-sm text-stone-500 dark:text-stone-400">
          No pending account requests.
        </div>
      ) : (
        <div className="divide-y divide-stone-100 dark:divide-stone-700/70">
          {accounts.map((account) => (
            <div
              key={account.id}
              className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                  <span className="font-ui text-sm font-semibold text-stone-800 dark:text-stone-100">
                    {account.display_name}
                  </span>
                  <span className="font-ui text-xs text-stone-500 dark:text-stone-400">
                    @{account.username}
                  </span>
                </div>
                <p className="font-ui text-xs text-stone-400 dark:text-stone-500">
                  Requested {formatCreatedAt(account.created_at)}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void decide(account.id, 'reject')}
                  disabled={actingId === account.id}
                  className="h-8 px-3 rounded-md font-ui text-xs font-medium
                             text-stone-600 dark:text-stone-300
                             bg-stone-100 dark:bg-surface-900
                             hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                             disabled:opacity-50"
                >
                  Reject
                </button>
                <button
                  type="button"
                  onClick={() => void decide(account.id, 'approve')}
                  disabled={actingId === account.id}
                  className="h-8 px-3 rounded-md font-ui text-xs font-medium
                             text-white bg-accent hover:bg-accent-dark transition-colors
                             disabled:opacity-50"
                >
                  Approve
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
