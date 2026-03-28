import {
  createContext,
  useContext,
  useRef,
  useMemo,
  type ReactNode,
} from 'react'

type TokenId = string // "query:42" or "candidate:17"

export interface TokenRefRegistryValue {
  registerRef: (tokenId: TokenId) => (el: HTMLSpanElement | null) => void
  getRect: (tokenId: TokenId) => DOMRect | null
}

const TokenRefRegistryContext = createContext<TokenRefRegistryValue | null>(null)

export function TokenRefProvider({ children }: { children: ReactNode }) {
  const mapRef = useRef<Map<TokenId, HTMLSpanElement>>(new Map())

  const registry = useMemo<TokenRefRegistryValue>(() => {
    const callbackCache = new Map<TokenId, (el: HTMLSpanElement | null) => void>()

    return {
      registerRef(tokenId: TokenId) {
        let cb = callbackCache.get(tokenId)
        if (!cb) {
          cb = (el: HTMLSpanElement | null) => {
            if (el) {
              mapRef.current.set(tokenId, el)
            } else {
              mapRef.current.delete(tokenId)
            }
          }
          callbackCache.set(tokenId, cb)
        }
        return cb
      },
      getRect(tokenId: TokenId) {
        const el = mapRef.current.get(tokenId)
        return el ? el.getBoundingClientRect() : null
      },
    }
  }, [])

  return (
    <TokenRefRegistryContext.Provider value={registry}>
      {children}
    </TokenRefRegistryContext.Provider>
  )
}

export function useTokenRefs(): TokenRefRegistryValue {
  const ctx = useContext(TokenRefRegistryContext)
  if (!ctx) {
    throw new Error('useTokenRefs must be used within a TokenRefProvider')
  }
  return ctx
}
