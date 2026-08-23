import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// jsdom has no layout engine, so framer-motion's layout animations would warn
// on every render. It also lacks matchMedia, which AppProvider reads to pick
// the initial theme.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia
}

afterEach(() => {
  cleanup()
  localStorage.clear()
})
