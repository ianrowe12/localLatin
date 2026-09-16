import { describe, expect, it } from 'vitest'
import {
  alignWordsToTokens,
  matrixOverTokens,
  normalizeWord,
  pieceTokens,
} from './wordHighlights'

const words = (texts: string[]) =>
  texts.map((text, idx) => ({ idx, text, piece_indices: [idx] }))

const tokens = (texts: string[]) =>
  texts.map((text, index) => ({ text, index, category: 'content' }))

describe('normalizeWord', () => {
  it('drops case, punctuation and diacritics', () => {
    expect(normalizeWord('Supra.')).toBe('supra')
    expect(normalizeWord('Quæ')).toBe('quæ')
    expect(normalizeWord('prouinciæ,')).toBe('prouinciæ')
    expect(normalizeWord('Ecclésia')).toBe('ecclesia')
    expect(normalizeWord('.')).toBe('')
  })
})

describe('alignWordsToTokens', () => {
  it('pairs the model words with the words on screen', () => {
    const map = alignWordsToTokens(
      words(['Episcopus', 'aut', 'presbiter']),
      tokens(['Episcopus', 'aut', 'presbiter']),
    )
    expect(map).toEqual([0, 1, 2])
  })

  it('skips punctuation tokens the query panel splits out', () => {
    // `latin_tokenize` emits "VII" and "." as two tokens; the model's word list
    // has one.
    const map = alignWordsToTokens(
      words(['VII.', 'Episcopus']),
      tokens(['VII', '.', 'Episcopus']),
    )
    expect(map).toEqual([0, -1, 1])
  })

  it('leaves the text past the model truncation unmatched', () => {
    const map = alignWordsToTokens(
      words(['Episcopus', 'aut']),
      tokens(['Episcopus', 'aut', 'presbiter', 'aut', 'diaconus']),
    )
    expect(map).toEqual([0, 1, -1, -1, -1])
  })

  it('is monotone, so a repeated word cannot pull a highlight backwards', () => {
    const map = alignWordsToTokens(
      words(['aut', 'presbiter', 'aut', 'diaconus']),
      tokens(['aut', 'presbiter', 'aut', 'diaconus']),
    )
    expect(map).toEqual([0, 1, 2, 3])
  })

  it('returns no pairing at all when there are no words', () => {
    expect(alignWordsToTokens([], tokens(['Episcopus']))).toEqual([-1])
  })

  it('withholds the highlight past the words the grids cover', () => {
    // The response keeps the whole word list so the walk still lines up, and
    // reports how far the matrices actually reach: the model read two words of
    // this file. The third is a word on screen with no row behind it.
    const map = alignWordsToTokens(
      words(['Episcopus', 'aut', 'presbiter', 'aut']),
      tokens(['Episcopus', 'aut', 'presbiter', 'aut']),
      2,
    )
    expect(map).toEqual([0, 1, -1, -1])
  })

  it('leaves the pairing alone when no bound is given', () => {
    const map = alignWordsToTokens(
      words(['Episcopus', 'aut']),
      tokens(['Episcopus', 'aut']),
    )
    expect(map).toEqual([0, 1])
  })
})

describe('matrixOverTokens', () => {
  it('re-indexes a word grid onto the displayed tokens', () => {
    // Two query words over three displayed tokens (one is punctuation), two
    // candidate words over two displayed tokens.
    const wordMatrix = [
      [0.2, 0.8],
      [0.1, 0.0],
    ]
    const out = matrixOverTokens(wordMatrix, [0, -1, 1], [0, 1])
    expect(out).toEqual([
      [0.2, 0.8],
      [0, 0],
      [0.1, 0.0],
    ])
  })
})

describe('pieceTokens', () => {
  it('keeps the model order and marks the short pieces', () => {
    const out = pieceTokens([
      { idx: 0, text: 'Epi', is_content: true },
      { idx: 1, text: '##scop', is_content: false },
    ])
    expect(out).toEqual([
      { text: 'Epi', index: 0, category: 'content' },
      { text: '##scop', index: 1, category: 'short_subword' },
    ])
  })
})
