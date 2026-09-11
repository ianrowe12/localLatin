import { useCallback, useState } from 'react'

export interface SelectedMember {
  /** The reviewer's choice for the CURRENT identity, or null for the default. */
  filename: string | null
  /**
   * Record a choice made against `owner`.
   *
   * Returns whether it was accepted. An event from an obsolete owner is
   * refused outright rather than re-read as a choice about the current group:
   * see the note on the hook.
   */
  select: (filename: string | null, owner: string) => boolean
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

  // The caller names the group its control was rendered for. Reading the
  // current key from a ref instead would label every event as current, which
  // is wrong for a control that is still interactive while it animates away:
  // the group underneath has already changed, and a filename the two groups
  // happen to share -- sigla repeat across directories -- would pass the
  // membership check and silently move the incoming panel to a witness nobody
  // chose. Both routes into this, the native select (mouse or keyboard) and
  // the "Show supporting witness" button, go through one bound handler, so
  // neither can navigate the new group from the old group's control.
  const select = useCallback(
    (filename: string | null, owner: string) => {
      if (owner !== identityKey) return false
      setState({ key: owner, filename })
      return true
    },
    [identityKey],
  )

  return {
    filename: state.key === identityKey ? state.filename : null,
    select,
  }
}
