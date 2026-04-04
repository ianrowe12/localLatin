import { useEffect } from 'react'

type KeyHandler = (e: KeyboardEvent) => void

export function useKeyboardShortcuts(handlers: Record<string, KeyHandler>) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Don't fire shortcuts when typing in inputs
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      const key = e.key
      const handler = handlers[key]
      if (handler) {
        handler(e)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [handlers])
}
