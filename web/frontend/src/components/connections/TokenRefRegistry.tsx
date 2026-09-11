import {
  createContext,
  useContext,
  useRef,
  useMemo,
  type ReactNode,
} from 'react'

type TokenId = string // "query:42" or "candidate:17"

export interface TokenRefRegistryValue {
  /**
   * Register an element as token `tokenId` OF THE PAIR `owner`.
   *
   * The owner travels with the element rather than with the registry, because
   * a panel being animated away is still mounted and still holds the DOM for
   * `candidate:1` -- just not the `candidate:1` anyone is asking about.
   */
  registerRef: (
    owner: string,
    tokenId: TokenId,
  ) => (el: HTMLSpanElement | null) => void
  /**
   * The rectangle of token `tokenId`, but only if the element registered
   * under it belongs to `owner`. A foreign owner answers null: the geometry
   * is absent, not approximate.
   */
  getRect: (tokenId: TokenId, owner: string) => DOMRect | null
  /** Bumped whenever a side's registered owner changes; see `subscribe`. */
  getOwnerVersion: () => number
  /**
   * Notified when the pair whose words occupy a side actually changes.
   *
   * Geometry is read from live elements, so a consumer that caches it has to
   * be told when those elements are replaced. Nothing else about a rendering
   * necessarily changes at that moment -- same connections, same container
   * size, no scroll -- so without this a cached path outlives the words it
   * was measured from.
   */
  subscribe: (listener: () => void) => () => void
}

const TokenRefRegistryContext = createContext<TokenRefRegistryValue | null>(null)

interface Registration {
  owner: string
  el: HTMLSpanElement
}

function sideOf(tokenId: TokenId): string {
  const colon = tokenId.indexOf(':')
  return colon === -1 ? tokenId : tokenId.slice(0, colon)
}

export function TokenRefProvider({ children }: { children: ReactNode }) {
  const mapRef = useRef<Map<TokenId, Registration>>(new Map())
  // Which pair currently owns the DOM of each side, as opposed to which pair
  // the rest of the app has moved on to.
  const sideOwnersRef = useRef<Map<string, string>>(new Map())
  const versionRef = useRef(0)
  const listenersRef = useRef<Set<() => void>>(new Set())

  const registry = useMemo<TokenRefRegistryValue>(() => {
    const callbackCache = new Map<string, (el: HTMLSpanElement | null) => void>()

    const announce = (side: string, owner: string | null) => {
      const owners = sideOwnersRef.current
      if (owner === null) {
        if (!owners.has(side)) return
        owners.delete(side)
      } else {
        if (owners.get(side) === owner) return
        owners.set(side, owner)
      }
      versionRef.current += 1
      for (const listener of listenersRef.current) listener()
    }

    return {
      registerRef(owner: string, tokenId: TokenId) {
        // Cached per owner as well as per token, so a span's ref prop is
        // stable across renders of one pair -- TokenSpan's comparator reads
        // it -- and genuinely different for the next pair.
        const cacheKey = `${owner}\u0000${tokenId}`
        let cb = callbackCache.get(cacheKey)
        if (!cb) {
          cb = (el: HTMLSpanElement | null) => {
            const map = mapRef.current
            if (el) {
              map.set(tokenId, { owner, el })
              announce(sideOf(tokenId), owner)
              return
            }
            // Detach only our own registration. React calls an outgoing ref
            // with null after the incoming one has registered in some orders,
            // and a blind delete there would drop the live element.
            const current = map.get(tokenId)
            if (!current || current.owner !== owner) return
            map.delete(tokenId)
            const side = sideOf(tokenId)
            if (sideOwnersRef.current.get(side) !== owner) return
            let survivor: string | null = null
            for (const [id, reg] of map) {
              if (sideOf(id) === side) {
                survivor = reg.owner
                break
              }
            }
            announce(side, survivor)
          }
          callbackCache.set(cacheKey, cb)
        }
        return cb
      },
      getRect(tokenId: TokenId, owner: string) {
        const reg = mapRef.current.get(tokenId)
        if (!reg || reg.owner !== owner) return null
        return reg.el.getBoundingClientRect()
      },
      getOwnerVersion() {
        return versionRef.current
      },
      subscribe(listener: () => void) {
        listenersRef.current.add(listener)
        return () => {
          listenersRef.current.delete(listener)
        }
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
