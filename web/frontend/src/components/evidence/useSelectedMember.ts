import { useCallback, useRef, useState } from 'react'

export interface SelectedMember {
  /** The reviewer's choice for the CURRENT identity, or null for the default. */
  filename: string | null
  select: (filename: string | null) => void
}

/**
 * A member choice that cannot outlive the prediction it was made against.
 *
 * The reset happens during render, not in an effect, because an effect runs a
 * commit too late: the first paint after a query, model, directory or refresh
 * generation change would still carry the previous group's witness, and that
 * paint is exactly the one a reviewer reads. Comparing the key while
 * rendering makes the stale choice unavailable from the first render, which
 * is the same discipline `CenterArea` already applies to prediction data.
 *
 * `identityKey` must come from `memberEvidenceKey`, so it includes the
 * directory and the shared request generation and never a rank alone.
 */
export function useSelectedMember(identityKey: string): SelectedMember {
  const [state, setState] = useState<{ key: string; filename: string | null }>(
    () => ({ key: identityKey, filename: null }),
  )

  // React's documented "adjust state when a prop changes" pattern: setting
  // state during render of this same component re-renders it immediately,
  // before anything is committed or painted.
  if (state.key !== identityKey) {
    setState({ key: identityKey, filename: null })
  }

  const keyRef = useRef(identityKey)
  keyRef.current = identityKey

  // Records the choice under the key that was on screen when it was made, so
  // a click cannot land on a different prediction's group.
  const select = useCallback((filename: string | null) => {
    setState({ key: keyRef.current, filename })
  }, [])

  return {
    filename: state.key === identityKey ? state.filename : null,
    select,
  }
}
