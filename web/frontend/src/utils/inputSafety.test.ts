import { describe, expect, it } from 'vitest'
import cases from './inputSafetyCases.json'
import {
  CCL_KEY_SUBJECT,
  DIRECTORY_NAME_SUBJECT,
  DISPLAY_NAME_SUBJECT,
  isUnsafeText,
  unsafeTextError,
  unsafeTextMessage,
  type SafeTextSubject,
} from './inputSafety'

// The same cases `web/tests/test_input_safety.py` runs against the server's
// rule, so the inline message and the server's 422 can never disagree.
describe('input safety (issue #274)', () => {
  it.each(cases.rejected.map((c) => [c.why, c.value] as const))(
    'refuses: %s',
    (_why, value) => {
      expect(isUnsafeText(value)).toBe(true)
    },
  )

  it.each(cases.accepted.map((c) => [c.why, c.value] as const))(
    'accepts: %s',
    (_why, value) => {
      expect(isUnsafeText(value)).toBe(false)
    },
  )

  it('accepts null and undefined', () => {
    expect(isUnsafeText(null)).toBe(false)
    expect(isUnsafeText(undefined)).toBe(false)
  })

  it.each([CCL_KEY_SUBJECT, DIRECTORY_NAME_SUBJECT, DISPLAY_NAME_SUBJECT])(
    'says the same sentence as the server for %s',
    (subject: SafeTextSubject) => {
      const messages = cases.messages as Record<string, string>
      expect(unsafeTextMessage(subject)).toBe(messages[subject])
    },
  )

  it('returns the message only for a refused value', () => {
    expect(unsafeTextError('CTOU.567.16', CCL_KEY_SUBJECT)).toBeNull()
    expect(unsafeTextError('=1+1', CCL_KEY_SUBJECT)).toBe(
      unsafeTextMessage(CCL_KEY_SUBJECT),
    )
  })
})
