import { useCallback, useRef, useState } from 'react'

export interface SelectedMember {
  /** The reviewer's choice for the CURRENT identity, or null for the default. */
  filename: string | null
  /**
   * Record a choice made against `owner`.
   *
   * Returns whether it was accepted. An event from an obsolete owner is
   * refused outright rather than re-read as a choice about the current group:
   * see the note on the hook. A refusal writes nothing, so the current
   * selection survives it untouched and the caller has nothing to announce.
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

  /**
   * The identity that is actually current, kept for COMPARISON only.
   *
   * Written during render, beside the reset above, so it is already right in
   * the first render of a new group rather than a commit later. It is never
   * read to decide what an event is ABOUT -- doing that is the original defect,
   * and the reason the old `keyRef` was removed: reading the current key as the
   * event's own owner relabels an obsolete action as a current one. Here the
   * event still names its own owner, and this only answers "is that owner still
   * the live one?".
   */
  const liveKeyRef = useRef(identityKey)
  liveKeyRef.current = identityKey

  // The caller names the group its control was rendered for. Reading the
  // current key from a ref instead would label every event as current, which
  // is wrong for a control that is still interactive while it animates away:
  // the group underneath has already changed, and a filename the two groups
  // happen to share -- sigla repeat across directories -- would pass the
  // membership check and silently move the incoming panel to a witness nobody
  // chose. Both routes into this, the native select (mouse or keyboard) and
  // the "Show supporting witness" button, go through one bound handler, so
  // neither can navigate the new group from the old group's control.
  //
  // Two checks, because the closure alone is not enough. A bar animating away
  // holds the callback it was rendered with, whose `identityKey` is its OWN old
  // group -- so an event from that bar agrees with it, and agreement between
  // two values captured together proves nothing. The second check is against
  // the live identity, and it is made BEFORE any state is written and before
  // the caller is told the choice was taken, so a refused event leaves no
  // selection behind and produces no announcement.
  const select = useCallback(
    (filename: string | null, owner: string) => {
      if (owner !== identityKey) return false
      if (owner !== liveKeyRef.current) return false
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
