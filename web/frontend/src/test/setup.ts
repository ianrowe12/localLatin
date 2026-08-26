import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import { __resetModelsCache } from '../api/models'

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

// ConnectionOverlay's line updater observes the center panel for resizes.
// jsdom has no layout engine and therefore no ResizeObserver.
if (!('ResizeObserver' in globalThis)) {
  class NoopResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver =
    NoopResizeObserver
}

afterEach(() => {
  cleanup()
  localStorage.clear()
  __resetModelsCache()
})
