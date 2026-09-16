import type { WordSpan } from '../../api/tokenMap'

/**
 * Word spans for a token-map fixture whose units are already whole words.
 *
 * Since issue #211 every real `/token_map` response carries `query_words` /
 * `candidate_words`, and the panels shade the WORD grid: the piece grid is
 * indexed by subword piece, so painting it over the reader's text by index is
 * the defect that issue fixed. A fixture that omits the word spans therefore
 * gets no highlight at all, which is correct behaviour and useless for a test
 * that is about something else.
 *
 * These fixtures predate the split and use one unit per displayed word, so the
 * two grids are the same numbers under two names and the spans are a
 * one-to-one relabelling.
 */
export function wordSpans(texts: string[]): WordSpan[] {
  return texts.map((text, idx) => ({
    idx,
    text,
    piece_indices: [idx],
    score: 0,
    score_pos: 0,
    score_neg: 0,
    is_content: true,
  }))
}

/** The words of a candidate file's text, as the candidate panel splits them. */
export function wordsOf(text: string): string[] {
  return text.split(/\s+/).filter(Boolean)
}
