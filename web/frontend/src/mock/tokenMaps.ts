import type {
  TokenMapResponse,
  TokenEntry,
  TopMatch,
  WordSpan,
} from '../api/tokenMap'

// ---------------------------------------------------------------------------
// Helper: generate a similarity matrix with realistic structure
// ---------------------------------------------------------------------------

function makeSimilarityMatrix(
  queryTokens: TokenEntry[],
  candidateTokens: TokenEntry[],
): number[][] {
  const qLen = queryTokens.length
  const cLen = candidateTokens.length
  const matrix: number[][] = []

  // Precompute exact and partial matches for realistic scores
  const exactPairs = new Set<string>()
  const partialPairs = new Set<string>()
  for (let qi = 0; qi < qLen; qi++) {
    const qText = queryTokens[qi].text.toLowerCase()
    for (let ci = 0; ci < cLen; ci++) {
      const cText = candidateTokens[ci].text.toLowerCase()
      if (qText === cText && qText.length > 2) {
        exactPairs.add(`${qi}-${ci}`)
      } else if (
        qText.length > 3 &&
        cText.length > 3 &&
        (qText.startsWith(cText.slice(0, 4)) || cText.startsWith(qText.slice(0, 4)))
      ) {
        partialPairs.add(`${qi}-${ci}`)
      }
    }
  }

  // Seeded pseudo-random for deterministic output
  let seed = 42
  function rand(): number {
    seed = (seed * 16807 + 0) % 2147483647
    return seed / 2147483647
  }

  for (let qi = 0; qi < qLen; qi++) {
    const row: number[] = []
    for (let ci = 0; ci < cLen; ci++) {
      const key = `${qi}-${ci}`
      if (exactPairs.has(key)) {
        row.push(0.70 + rand() * 0.25) // 0.70 - 0.95
      } else if (partialPairs.has(key)) {
        row.push(0.40 + rand() * 0.30) // 0.40 - 0.70
      } else {
        row.push(0.05 + rand() * 0.20) // 0.05 - 0.25 noise
      }
    }
    matrix.push(row)
  }

  return matrix
}

// ---------------------------------------------------------------------------
// Helper: derive top matches from a similarity matrix
// ---------------------------------------------------------------------------

function deriveTopMatches(
  matrix: number[][],
  queryTokens: TokenEntry[],
): Record<string, TopMatch[]> {
  const result: Record<string, TopMatch[]> = {}
  for (let qi = 0; qi < matrix.length; qi++) {
    if (!queryTokens[qi].is_content) continue
    const row = matrix[qi]
    const sorted = row
      .map((score, ci) => ({ candidate_idx: ci, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
    result[String(qi)] = sorted
  }
  return result
}

// ---------------------------------------------------------------------------
// Helper: generate IG scores (mostly low, a few spikes)
// ---------------------------------------------------------------------------

function makeIgScores(tokens: TokenEntry[]): number[] {
  let seed = 137
  function rand(): number {
    seed = (seed * 16807 + 0) % 2147483647
    return seed / 2147483647
  }

  return tokens.map((t) => {
    if (!t.is_content) return 0.0
    const r = rand()
    // ~15% chance of a spike
    if (r > 0.85) return 0.20 + rand() * 0.30
    return rand() * 0.10
  })
}

// ---------------------------------------------------------------------------
// Query 1 + Prediction 1: BAV1341.13v.1 vs Can.apost.42
// ---------------------------------------------------------------------------

const Q1_TOKENS: TokenEntry[] = [
  { idx: 0, text: 'I', is_content: true },
  { idx: 1, text: '.', is_content: false },
  { idx: 2, text: 'DE', is_content: true },
  { idx: 3, text: 'HIS', is_content: true },
  { idx: 4, text: 'QUI', is_content: true },
  { idx: 5, text: 'NUPTIAS', is_content: true },
  { idx: 6, text: 'DAMPNANT', is_content: true },
  { idx: 7, text: '.', is_content: false },
  { idx: 8, text: 'Si', is_content: true },
  { idx: 9, text: 'quis', is_content: true },
  { idx: 10, text: 'nuptias', is_content: true },
  { idx: 11, text: 'in', is_content: true },
  { idx: 12, text: 'accusationem', is_content: true },
  { idx: 13, text: 'deduxerit', is_content: true },
  { idx: 14, text: ',', is_content: false },
  { idx: 15, text: 'et', is_content: true },
  { idx: 16, text: 'mulierem', is_content: true },
  { idx: 17, text: 'fidelem', is_content: true },
  { idx: 18, text: 'ac', is_content: true },
  { idx: 19, text: 'religiosam', is_content: true },
  { idx: 20, text: 'cum', is_content: true },
  { idx: 21, text: 'viro', is_content: true },
  { idx: 22, text: 'suo', is_content: true },
  { idx: 23, text: 'dormientem', is_content: true },
  { idx: 24, text: 'abhominandam', is_content: true },
  { idx: 25, text: 'crediderit', is_content: true },
  { idx: 26, text: ',', is_content: false },
  { idx: 27, text: 'aut', is_content: true },
  { idx: 28, text: 'etiam', is_content: true },
  { idx: 29, text: 'accusandam', is_content: true },
  { idx: 30, text: 'tamquam', is_content: true },
  { idx: 31, text: 'non', is_content: true },
  { idx: 32, text: 'posse', is_content: true },
  { idx: 33, text: 'coniugatos', is_content: true },
  { idx: 34, text: 'in', is_content: true },
  { idx: 35, text: 'regnum', is_content: true },
  { idx: 36, text: 'dei', is_content: true },
  { idx: 37, text: 'ingredi', is_content: true },
  { idx: 38, text: ',', is_content: false },
  { idx: 39, text: 'anathema', is_content: true },
  { idx: 40, text: 'sit', is_content: true },
  { idx: 41, text: '.', is_content: false },
]

const C1_TOKENS: TokenEntry[] = [
  { idx: 0, text: 'XLII', is_content: true },
  { idx: 1, text: '.', is_content: false },
  { idx: 2, text: 'Si', is_content: true },
  { idx: 3, text: 'quis', is_content: true },
  { idx: 4, text: 'episcopus', is_content: true },
  { idx: 5, text: 'aut', is_content: true },
  { idx: 6, text: 'presbiter', is_content: true },
  { idx: 7, text: 'aut', is_content: true },
  { idx: 8, text: 'diaconus', is_content: true },
  { idx: 9, text: 'aleae', is_content: true },
  { idx: 10, text: 'id', is_content: true },
  { idx: 11, text: 'est', is_content: true },
  { idx: 12, text: 'tabulae', is_content: true },
  { idx: 13, text: 'ludo', is_content: true },
  { idx: 14, text: 'se', is_content: true },
  { idx: 15, text: 'voluptuosius', is_content: true },
  { idx: 16, text: 'dederit', is_content: true },
  { idx: 17, text: ',', is_content: false },
  { idx: 18, text: 'aut', is_content: true },
  { idx: 19, text: 'desinat', is_content: true },
  { idx: 20, text: 'aut', is_content: true },
  { idx: 21, text: 'certe', is_content: true },
  { idx: 22, text: 'damnetur', is_content: true },
  { idx: 23, text: '.', is_content: false },
]

const SIM_Q1_C1 = makeSimilarityMatrix(Q1_TOKENS, C1_TOKENS)
const TOP_Q1_C1 = deriveTopMatches(SIM_Q1_C1, Q1_TOKENS)

// ---------------------------------------------------------------------------
// Query 1 + Prediction 3: BAV1341.13v.1 vs CNEO.315.6
// (simpler version with fewer tokens)
// ---------------------------------------------------------------------------

const C1_P3_TOKENS: TokenEntry[] = [
  { idx: 0, text: 'Si', is_content: true },
  { idx: 1, text: 'quis', is_content: true },
  { idx: 2, text: 'dixerit', is_content: true },
  { idx: 3, text: 'nuptias', is_content: true },
  { idx: 4, text: 'detestabiles', is_content: true },
  { idx: 5, text: 'esse', is_content: true },
  { idx: 6, text: 'et', is_content: true },
  { idx: 7, text: 'procreationem', is_content: true },
  { idx: 8, text: 'filiorum', is_content: true },
  { idx: 9, text: 'malam', is_content: true },
  { idx: 10, text: ',', is_content: false },
  { idx: 11, text: 'anathema', is_content: true },
  { idx: 12, text: 'sit', is_content: true },
  { idx: 13, text: '.', is_content: false },
]

const SIM_Q1_P3 = makeSimilarityMatrix(Q1_TOKENS, C1_P3_TOKENS)
const TOP_Q1_P3 = deriveTopMatches(SIM_Q1_P3, Q1_TOKENS)

// ---------------------------------------------------------------------------
// Query 3 + Prediction 1: Hat42.148v.3 vs Can.apost.41
// ---------------------------------------------------------------------------

const Q3_TOKENS: TokenEntry[] = [
  { idx: 0, text: 'XLIIII', is_content: true },
  { idx: 1, text: '.', is_content: false },
  { idx: 2, text: 'Si', is_content: true },
  { idx: 3, text: 'quis', is_content: true },
  { idx: 4, text: 'episcopus', is_content: true },
  { idx: 5, text: 'aut', is_content: true },
  { idx: 6, text: 'presbiter', is_content: true },
  { idx: 7, text: 'aut', is_content: true },
  { idx: 8, text: 'diaconus', is_content: true },
  { idx: 9, text: 'aliquem', is_content: true },
  { idx: 10, text: 'a', is_content: true },
  { idx: 11, text: 'communione', is_content: true },
  { idx: 12, text: 'sine', is_content: true },
  { idx: 13, text: 'causa', is_content: true },
  { idx: 14, text: 'reppulerit', is_content: true },
  { idx: 15, text: ',', is_content: false },
  { idx: 16, text: 'deponatur', is_content: true },
  { idx: 17, text: '.', is_content: false },
]

const C3_TOKENS: TokenEntry[] = [
  { idx: 0, text: 'Praecipimus', is_content: true },
  { idx: 1, text: 'episcopum', is_content: true },
  { idx: 2, text: 'habere', is_content: true },
  { idx: 3, text: 'rerum', is_content: true },
  { idx: 4, text: 'ecclesiasticarum', is_content: true },
  { idx: 5, text: 'potestatem', is_content: true },
  { idx: 6, text: '.', is_content: false },
  { idx: 7, text: 'Si', is_content: true },
  { idx: 8, text: 'enim', is_content: true },
  { idx: 9, text: 'animae', is_content: true },
  { idx: 10, text: 'hominum', is_content: true },
  { idx: 11, text: 'pretiosae', is_content: true },
  { idx: 12, text: 'illi', is_content: true },
  { idx: 13, text: 'sunt', is_content: true },
  { idx: 14, text: 'commissae', is_content: true },
  { idx: 15, text: '.', is_content: false },
]

const SIM_Q3_C1 = makeSimilarityMatrix(Q3_TOKENS, C3_TOKENS)
const TOP_Q3_C1 = deriveTopMatches(SIM_Q3_C1, Q3_TOKENS)

// ---------------------------------------------------------------------------
// Helper: the word-level view (issue #211)
// ---------------------------------------------------------------------------
//
// The real backend groups the model's subword pieces into the words of the
// original text. These fixtures were written before that existed and their
// "pieces" are already whole words, so the grouping here is one word per
// content piece -- enough for `npm run dev:mock` to exercise the word view and
// the "Show pieces" toggle, which otherwise never appear outside the tests.
//
// Added to ONE pair on purpose. The other two keep the pre-#211 shape, so the
// piece-only path a bulk artifact takes stays reachable in the mock too.

function makeWordSpans(tokens: TokenEntry[]): WordSpan[] {
  const words: WordSpan[] = []
  for (const token of tokens) {
    if (!token.is_content) continue // punctuation joins no word
    words.push({
      idx: words.length,
      text: token.text,
      piece_indices: [token.idx],
      score: 0,
      score_pos: 0,
      score_neg: 0,
      is_content: true,
    })
  }
  return words
}

/** The piece grid re-indexed onto those words. One piece each, so a lookup. */
function makeWordMatrix(
  matrix: number[][],
  queryWords: WordSpan[],
  candidateWords: WordSpan[],
): number[][] {
  return queryWords.map((qw) =>
    candidateWords.map((cw) => matrix[qw.piece_indices[0]][cw.piece_indices[0]]),
  )
}

const Q1_WORDS = makeWordSpans(Q1_TOKENS)
const C1_WORDS = makeWordSpans(C1_TOKENS)
const WORD_SIM_Q1_C1 = makeWordMatrix(SIM_Q1_C1, Q1_WORDS, C1_WORDS)

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export const MOCK_TOKEN_MAPS: Map<string, TokenMapResponse> = new Map([
  [
    '1-Can.apost.42',
    {
      example_id: 1,
      model: 'bowphs_LaTa',
      layer: 4,
      D: 10,
      bucket: 'true_positive',
      query_path: 'canon/BAV1341/BAV1341.13v.1.txt',
      candidate_path: 'canon/Can.apost.42/Can.apost.42_a.txt',
      query_tokens: Q1_TOKENS,
      candidate_tokens: C1_TOKENS,
      similarity_matrix: SIM_Q1_C1,
      ig_weighted_matrix: null,
      top_matches: TOP_Q1_C1,
      query_ig_baseline: makeIgScores(Q1_TOKENS),
      query_ig_abtt: makeIgScores(Q1_TOKENS),
      candidate_ig_baseline: makeIgScores(C1_TOKENS),
      candidate_ig_abtt: makeIgScores(C1_TOKENS),
      auto_highlights: null,
      // The word view, on this pair only (see makeWordSpans above).
      query_words: Q1_WORDS,
      candidate_words: C1_WORDS,
      query_words_scored: Q1_WORDS.length,
      candidate_words_scored: C1_WORDS.length,
      word_segmentation: 'text',
      word_aggregation: 'sum',
      word_similarity_matrix: WORD_SIM_Q1_C1,
    },
  ],
  [
    '1-CNEO.315.6',
    {
      example_id: 1,
      model: 'bowphs_LaTa',
      layer: 4,
      D: 10,
      bucket: 'false_positive',
      query_path: 'canon/BAV1341/BAV1341.13v.1.txt',
      candidate_path: 'canon/CNEO.315.6/CNEO.315.6_a.txt',
      query_tokens: Q1_TOKENS,
      candidate_tokens: C1_P3_TOKENS,
      similarity_matrix: SIM_Q1_P3,
      ig_weighted_matrix: null,
      top_matches: TOP_Q1_P3,
      query_ig_baseline: makeIgScores(Q1_TOKENS),
      query_ig_abtt: makeIgScores(Q1_TOKENS),
      candidate_ig_baseline: makeIgScores(C1_P3_TOKENS),
      candidate_ig_abtt: makeIgScores(C1_P3_TOKENS),
      auto_highlights: null,
    },
  ],
  [
    '3-Can.apost.41',
    {
      example_id: 3,
      model: 'bowphs_LaTa',
      layer: 4,
      D: 10,
      bucket: 'true_positive',
      query_path: 'canon/Hat42/Hat42.148v.3.txt',
      candidate_path: 'canon/Can.apost.41/Can.apost.41_a.txt',
      query_tokens: Q3_TOKENS,
      candidate_tokens: C3_TOKENS,
      similarity_matrix: SIM_Q3_C1,
      ig_weighted_matrix: null,
      top_matches: TOP_Q3_C1,
      query_ig_baseline: makeIgScores(Q3_TOKENS),
      query_ig_abtt: makeIgScores(Q3_TOKENS),
      candidate_ig_baseline: makeIgScores(C3_TOKENS),
      candidate_ig_abtt: makeIgScores(C3_TOKENS),
      auto_highlights: null,
    },
  ],
])
