export type TokenCategory = 'content' | 'punctuation' | 'empty' | 'number' | 'short_subword'

const LATIN_PUNCTUATION = new Set('.,;:!?()[]{}"\'-/\\@#$%^&*+=<>~`'.split(''))

export function classifyToken(text: string): TokenCategory {
  const cleaned = text.replace(/^[▁##Ġ\s]+/, '').trim()
  if (!cleaned) return 'empty'
  if ([...cleaned].every(c => LATIN_PUNCTUATION.has(c))) return 'punctuation'
  if (/^\d+$/.test(cleaned)) return 'number'
  if (cleaned.length <= 2) return 'short_subword'
  return 'content'
}

export function isContentToken(category: TokenCategory): boolean {
  return category === 'content' || category === 'number'
}
