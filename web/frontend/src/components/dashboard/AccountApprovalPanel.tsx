import { useCallback, useEffect, useState } from 'react'
import {
  approveAccount,
  listAccounts,
  rejectAccount,
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
