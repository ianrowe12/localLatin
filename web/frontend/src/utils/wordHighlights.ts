/**
 * Word-level highlighting for the token maps (issue #211).
 *
 * The backend groups the model's subword pieces into the words of the original
 * text and returns both: `query_words` / `candidate_words` (with the piece
 * indices behind each word) alongside the untouched piece grids. This module is
 * the display half.
 *
 * Two jobs.
 *
 * 1. **Line the words up with the words on screen.** The panels render the
 *    file's own text — `latin_tokenize` on the query side, a whitespace split on
 *    the candidate side — so neither is index-identical with the model's word
 *    list: the query side splits punctuation into its own tokens, and either
 *    side can run past the model's truncation point. A monotone two-pointer over
 *    the normalised forms pairs them and leaves the rest unmatched.
 * 2. **Re-index the matrix.** Once every displayed token knows its word, the
 *    word x word grid can be rebuilt over displayed-token indices, which is the
 *    shape `DocumentPanel` already reads. A fragmented word then gets ONE score
 *    on ONE span instead of a different shade on each of its pieces.
 *
 * What this deliberately does not do is filter anything by frequency. The
 * analysis behind the issue measured prefix and high-frequency pieces at or
 * below their share of the attribution under the deployed variant, so a
 * frequency mask would hide evidence twice
 * (`docs/research/prefix_attribution_analysis.md`).
 */

export interface WordSpanLike {
  idx: number
  text: string
  piece_indices?: number[]
}

export interface DisplayTokenLike {
  text: string
  index: number
  category?: string
}

/** Lowercase, drop punctuation and diacritics. Mirrors the backend's rule. */
export function normalizeWord(text: string): string {
  const stripped = text
    .normalize('NFD')
    // Combining marks, then anything that is not a letter or a digit.
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\p{L}\p{N}]/gu, '')
  return stripped.toLowerCase()
}

/**
 * Map each displayed token to a word index, or -1.
 *
 * Monotone by construction: a token can never be paired with a word earlier
 * than its predecessor's, so a repeated word ("aut ... aut") cannot pull a
 * highlight backwards. Unmatched on either side is normal and harmless —
 * punctuation tokens on the query side, and every word past the model's
 * truncation point — and shows up as no highlight rather than a wrong one.
 *
 * `scoredWords` is the response's `query_words_scored` / `candidate_words_scored`:
 * the word grids stop there, because the model stopped reading there. A word at
 * or beyond it still takes part in the walk, so the words after it keep lining
 * up, but carries no highlight.
 */
export function alignWordsToTokens(
  words: WordSpanLike[],
  tokens: DisplayTokenLike[],
  scoredWords?: number,
): number[] {
  const out = new Array<number>(tokens.length).fill(-1)
  if (words.length === 0) return out

  const normWords = words.map((w) => normalizeWord(w.text))
  let wi = 0
  for (let ti = 0; ti < tokens.length; ti++) {
    const token = normalizeWord(tokens[ti]?.text ?? '')
    if (!token) continue // punctuation-only display token
    // Look a little ahead: the model's list can skip a word the tokenizer
    // swallowed, and the display can hold one the model never saw.
    const LOOKAHEAD = 3
    let found = -1
    for (let k = wi; k < Math.min(words.length, wi + LOOKAHEAD + 1); k++) {
      if (normWords[k] === token) {
        found = k
        break
      }
    }
    if (found < 0) continue
    // The walk advances either way; only the highlight is withheld.
    out[ti] = scoredWords !== undefined && found >= scoredWords ? -1 : found
    wi = found + 1
  }
  return out
}

/**
 * Rebuild a word x word matrix over displayed-token indices.
 *
 * Every displayed token of a word carries its word's score, so the highlight is
 * the same shade across the whole word however the two tokenisations disagree.
 */
export function matrixOverTokens(
  wordMatrix: number[][],
  queryMap: number[],
  candidateMap: number[],
): number[][] {
  const out: number[][] = []
  for (let ti = 0; ti < queryMap.length; ti++) {
    const wi = queryMap[ti]
    const row = wi >= 0 ? wordMatrix[wi] : undefined
    const built = new Array<number>(candidateMap.length).fill(0)
    if (row) {
      for (let tj = 0; tj < candidateMap.length; tj++) {
        const wj = candidateMap[tj]
        if (wj >= 0 && wj < row.length) built[tj] = row[wj]
      }
    }
    out.push(built)
  }
  return out
}

/** Display tokens for the pieces view: the model's own units, in order. */
export function pieceTokens(
  pieces: { idx: number; text: string; is_content?: boolean }[],
): { text: string; index: number; category: string }[] {
  return pieces.map((p) => ({
    text: p.text,
    index: p.idx,
    category: p.is_content === false ? 'short_subword' : 'content',
  }))
}
