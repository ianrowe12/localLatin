/**
 * Compute a word-level similarity matrix from query and candidate tokens.
 * Used as a fallback when no pre-computed IG token map is available.
 */

interface TokenLike {
  text: string
  index: number
  category: string
}

interface WordMatchMap {
  similarity_matrix: number[][]
  top_matches: Record<string, { candidate_idx: number; score: number }[]>
}

/** Strip punctuation and lowercase for Latin word comparison. */
function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-zà-ÿ]/g, '')
}

/** Score two normalized tokens: 1.0 exact, 0.6 shared prefix ≥ 4, else 0. */
function score(a: string, b: string): number {
  if (!a || !b) return 0
  if (a === b) return 1.0
  const minLen = Math.min(a.length, b.length)
  if (minLen >= 4) {
    let shared = 0
    while (shared < minLen && a[shared] === b[shared]) shared++
    if (shared >= 4) return 0.6
  }
  return 0
}

export function buildWordMatchMap(
  queryTokens: TokenLike[],
  candidateTokens: TokenLike[],
): WordMatchMap {
  const qNorms = queryTokens.map((t) => normalize(t.text))
  const cNorms = candidateTokens.map((t) => normalize(t.text))

  const matrix: number[][] = []
  const topMatches: Record<string, { candidate_idx: number; score: number }[]> = {}

  for (let qi = 0; qi < qNorms.length; qi++) {
    const row: number[] = []
    for (let ci = 0; ci < cNorms.length; ci++) {
      row.push(score(qNorms[qi], cNorms[ci]))
    }
    matrix.push(row)

    // Top 3 candidates by score (descending)
    const ranked = row
      .map((s, ci) => ({ candidate_idx: ci, score: s }))
      .filter((m) => m.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
    if (ranked.length > 0) {
      topMatches[String(qi)] = ranked
    }
  }

  return { similarity_matrix: matrix, top_matches: topMatches }
}
