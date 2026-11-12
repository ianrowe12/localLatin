/**
 * Typed values that end up verbatim in the CCL key export (issue #274).
 *
 * The frontend copy of `web/services/input_safety.py`, so a scholar sees why a
 * value is refused before pressing anything. The server is the authority and
 * refuses the same values with the same sentence; both sides are tested
 * against `inputSafetyCases.json`, which holds the cases and the messages.
 *
 * A value is refused when, after leading whitespace and control characters
 * (tab, CR, LF, ...), it starts with =, +, - or @, which a spreadsheet runs as
 * a formula; or when it contains | anywhere, because the export joins lists
 * with ' | ' and a value ending in ' |' corrupts the split as surely as one
 * containing ' | '. Nothing already stored is checked or rewritten.
 */

export type SafeTextSubject = 'A CCL key' | 'A directory name' | 'A display name'

export const CCL_KEY_SUBJECT: SafeTextSubject = 'A CCL key'
export const DIRECTORY_NAME_SUBJECT: SafeTextSubject = 'A directory name'
export const DISPLAY_NAME_SUBJECT: SafeTextSubject = 'A display name'

// JS \s plus C0 and C1 controls and the BOM: the same set as the server's
// pattern (Python's \s differs from JS's only inside the control ranges and
// at the BOM, both listed explicitly on each side).
const LEADING_FORMULA = /^[\s\x00-\x1f\x7f-\x9f﻿]*[=+\-@]/

export function isUnsafeText(value: string | null | undefined): boolean {
  if (!value) return false
  return value.includes('|') || LEADING_FORMULA.test(value)
}

export function unsafeTextMessage(subject: SafeTextSubject): string {
  return `${subject} cannot start with =, +, - or @ (a spreadsheet would run it as a formula) and cannot contain | (the export uses it to separate names).`
}

/** The inline message for `value`, or null when it may be submitted. */
export function unsafeTextError(
  value: string | null | undefined,
  subject: SafeTextSubject,
): string | null {
  return isUnsafeText(value) ? unsafeTextMessage(subject) : null
}
