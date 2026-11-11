import { useCallback, useEffect, useMemo, useState } from 'react'
import { useReviewer } from '../../contexts/ReviewerContext'
import {
  approveAccount,
  createAccount,
  listAccounts,
  rejectAccount,
  resetAccountPassword,
  setAccountRole,
  type AccountCreateResponse,
  type AccountPublic,
  type AccountRole,
} from '../../api/auth'

function roleLabel(role: AccountRole): string {
  return role === 'pi_admin' ? 'PI/admin' : 'Reviewer'
}

/** What the confirm step says before a role change is sent (issue #265). */
function roleChangeWarning(
  account: Pick<AccountPublic, 'display_name'>,
  role: AccountRole,
): string {
  return role === 'pi_admin'
    ? `Make ${account.display_name} a PI/admin? PI/admins can approve accounts, reset passwords and export all feedback.`
    : `Make ${account.display_name} a reviewer? They lose PI/admin access on their next request.`
}

const OWN_ROLE_REASON =
  'You cannot change your own role. Another PI/admin must do it.'

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
  const { user } = useReviewer()
  const [accounts, setAccounts] = useState<AccountPublic[]>([])
  const [approved, setApproved] = useState<AccountPublic[]>([])
  const [resettingId, setResettingId] = useState<number | null>(null)
  const [resetResult, setResetResult] = useState<{
    username: string
    password: string
  } | null>(null)
  const [copied, setCopied] = useState(false)
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
  const [roleConfirm, setRoleConfirm] = useState<{
    id: number
    role: AccountRole
  } | null>(null)
  const [roleChangingId, setRoleChangingId] = useState<number | null>(null)
  const [roleError, setRoleError] = useState<string | null>(null)
  const [roleNotice, setRoleNotice] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [pending, all] = await Promise.all([
        listAccounts('pending'),
        listAccounts('approved'),
      ])
      setAccounts(pending)
      setApproved(all)
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

  // The caller is never in this list. A self-reset would kill their own session
  // and leave the only copy of the temporary password in a browser tab, with no
  // email recovery; the backend refuses it too. Admins use Change password
  // instead, and a second PI/admin resets one who is genuinely locked out.
  const resettable = useMemo(
    () => approved.filter((account) => account.id !== user?.id),
    [approved, user],
  )

  const resetPassword = async (account: AccountPublic) => {
    if (resettingId !== null) return
    setResettingId(account.id)
    setError(null)
    setResetResult(null)
    setCopied(false)
    try {
      const result = await resetAccountPassword(account.id)
      // Shown once: the backend stores only the hash.
      setResetResult({
        username: result.account.username,
        password: result.temporary_password,
      })
      setApproved((current) =>
        current.map((item) => (item.id === account.id ? result.account : item)),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Password reset failed')
    } finally {
      setResettingId(null)
    }
  }

  // Two steps on purpose: a PI/admin can approve accounts, reset passwords and
  // export all feedback, so a promotion is never one stray click. The server
  // owns the guards (own role 400, last PI/admin 409) and its text is shown.
  const changeRole = async (account: AccountPublic, role: AccountRole) => {
    if (roleChangingId !== null) return
    setRoleChangingId(account.id)
    setRoleError(null)
    setRoleNotice(null)
    try {
      const updated = await setAccountRole(account.id, { role })
      setApproved((current) =>
        current.map((item) => (item.id === account.id ? updated : item)),
      )
      setRoleNotice(
        `${updated.display_name} is now ${
          updated.role === 'pi_admin' ? 'a PI/admin' : 'a reviewer'
        }.`,
      )
      setRoleConfirm(null)
    } catch (err) {
      setRoleError(err instanceof Error ? err.message : 'Role change failed')
    } finally {
      setRoleChangingId(null)
    }
  }

  const copyTemporaryPassword = async () => {
    if (!resetResult) return
    try {
      await navigator.clipboard.writeText(resetResult.password)
      setCopied(true)
    } catch {
      setCopied(false)
      setError('Could not copy. Select the password and copy it manually.')
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

      <div className="border-t border-stone-100 dark:border-stone-700/70">
        <div className="px-4 py-3">
          <h4 className="font-ui text-sm font-semibold text-stone-700 dark:text-stone-200">
            Password Resets
          </h4>
          <p className="font-ui text-xs text-stone-500 dark:text-stone-400">
            Generates a one-time temporary password, signs the account out
            everywhere, and forces a change at next sign-in.
          </p>
        </div>

        {resetResult && (
          <div className="mx-4 mb-3 rounded-md bg-accent/10 dark:bg-accent/15 px-3 py-2">
            <p className="font-ui text-sm text-stone-700 dark:text-stone-200">
              Temporary password for @{resetResult.username}, shown once:
            </p>
            <div className="mt-1 flex items-center gap-2">
              <code className="font-mono text-xs text-accent dark:text-accent-light break-all">
                {resetResult.password}
              </code>
              <button
                type="button"
                onClick={() => {
                  setResetResult(null)
                  setCopied(false)
                }}
                className="h-7 px-2 rounded-md font-ui text-xs font-medium
                           bg-white dark:bg-surface-900 text-stone-600 dark:text-stone-300
                           border border-stone-200 dark:border-stone-700
                           hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors"
              >
                Dismiss
              </button>
              <button
                type="button"
                onClick={() => void copyTemporaryPassword()}
                className="h-7 px-2 rounded-md font-ui text-xs font-medium
                           bg-white dark:bg-surface-900 text-stone-600 dark:text-stone-300
                           border border-stone-200 dark:border-stone-700
                           hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors"
              >
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          </div>
        )}

        {resettable.length === 0 ? (
          <div className="px-4 pb-5 font-ui text-sm text-stone-500 dark:text-stone-400">
            No other approved accounts. Use the account menu to change your own
            password.
          </div>
        ) : (
          <div className="divide-y divide-stone-100 dark:divide-stone-700/70">
            {resettable.map((account) => (
              <div
                key={account.id}
                className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
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
                    {account.role === 'pi_admin' ? 'PI/admin' : 'Reviewer'}
                    {account.must_change_password
                      ? ' - password change pending'
                      : ''}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void resetPassword(account)}
                  disabled={resettingId === account.id}
                  className="h-8 px-3 rounded-md font-ui text-xs font-medium
                             text-stone-600 dark:text-stone-300
                             bg-stone-100 dark:bg-surface-900
                             hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                             disabled:opacity-50"
                >
                  {resettingId === account.id ? 'Resetting...' : 'Reset password'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div
        className="border-t border-stone-100 dark:border-stone-700/70"
        data-testid="account-roles"
      >
        <div className="px-4 py-3">
          <h4 className="font-ui text-sm font-semibold text-stone-700 dark:text-stone-200">
            Roles
          </h4>
          <p className="font-ui text-xs text-stone-500 dark:text-stone-400">
            PI/admins can approve accounts, reset passwords and export all
            feedback. A change applies on the account&apos;s next request.
          </p>
        </div>

        {roleError && (
          <div
            role="alert"
            className="mx-4 mb-3 font-ui text-sm text-red-600 dark:text-red-400"
          >
            {roleError}
          </div>
        )}
        {roleNotice && (
          <div
            role="status"
            className="mx-4 mb-3 rounded-md bg-accent/10 dark:bg-accent/15 px-3 py-2 font-ui text-sm text-stone-700 dark:text-stone-200"
          >
            {roleNotice}
          </div>
        )}

        {approved.length === 0 ? (
          <div className="px-4 pb-5 font-ui text-sm text-stone-500 dark:text-stone-400">
            No approved accounts.
          </div>
        ) : (
          <div className="divide-y divide-stone-100 dark:divide-stone-700/70">
            {approved.map((account) => {
              const isSelf = account.id === user?.id
              const target: AccountRole =
                account.role === 'pi_admin' ? 'reviewer' : 'pi_admin'
              const confirming = roleConfirm?.id === account.id
              const busy = roleChangingId === account.id
              return (
                <div
                  key={account.id}
                  data-testid={`role-row-${account.id}`}
                  className="flex flex-col gap-2 px-4 py-3"
                >
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
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
                        {roleLabel(account.role)}
                        {isSelf ? ' (you)' : ''}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setRoleError(null)
                        setRoleNotice(null)
                        setRoleConfirm({ id: account.id, role: target })
                      }}
                      disabled={isSelf || busy || confirming}
                      title={isSelf ? OWN_ROLE_REASON : undefined}
                      className="h-8 px-3 rounded-md font-ui text-xs font-medium
                                 text-stone-600 dark:text-stone-300
                                 bg-stone-100 dark:bg-surface-900
                                 hover:bg-stone-200 dark:hover:bg-stone-700 transition-colors
                                 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {target === 'pi_admin' ? 'Make PI/admin' : 'Make reviewer'}
                    </button>
                  </div>
                  {isSelf && (
                    <p className="font-ui text-xs text-stone-500 dark:text-stone-400">
                      {OWN_ROLE_REASON}
                    </p>
                  )}
                  {confirming && roleConfirm && (
                    <div className="rounded-md border border-amber-300/70 dark:border-amber-600/50 bg-amber-50 dark:bg-amber-900/20 px-3 py-2">
                      <p className="font-ui text-sm text-stone-700 dark:text-stone-200">
                        {roleChangeWarning(account, roleConfirm.role)}
                      </p>
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setRoleConfirm(null)}
                          disabled={busy}
                          className="h-7 px-2 rounded-md font-ui text-xs font-medium
                                     bg-white dark:bg-surface-900 text-stone-600 dark:text-stone-300
                                     border border-stone-200 dark:border-stone-700
                                     hover:bg-stone-100 dark:hover:bg-stone-700 transition-colors
                                     disabled:opacity-50"
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          onClick={() => void changeRole(account, roleConfirm.role)}
                          disabled={busy}
                          className="h-7 px-2 rounded-md font-ui text-xs font-medium
                                     text-white bg-accent hover:bg-accent-dark transition-colors
                                     disabled:opacity-50"
                        >
                          {busy
                            ? 'Saving...'
                            : roleConfirm.role === 'pi_admin'
                              ? 'Confirm: make PI/admin'
                              : 'Confirm: make reviewer'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
