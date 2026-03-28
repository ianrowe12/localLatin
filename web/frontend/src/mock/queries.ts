import type { QueryListItem, QueryDetail, TokenInfo } from '../api/queries'

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

function tokenize(text: string): TokenInfo[] {
  const tokens: TokenInfo[] = []
  // Split on word boundaries, keeping punctuation as separate tokens
  const parts = text.match(/[A-Za-z\u00C0-\u024F]+|[.,;:!?()]+|\d+/g)
  if (!parts) return tokens

  let idx = 0
  for (const part of parts) {
    const isPunct = /^[.,;:!?()]+$/.test(part)
    const isNumber = /^\d+$/.test(part)
    tokens.push({
      text: part,
      index: idx,
      category: isPunct ? 'punctuation' : isNumber ? 'number' : 'content',
    })
    idx++
  }
  return tokens
}

// ---------------------------------------------------------------------------
// Raw Latin texts
// ---------------------------------------------------------------------------

const TEXT_1 =
  'I.   DE HIS QUI NUPTIAS DAMPNANT.   Si quis nuptias in accusationem deduxerit, et mulierem fidelem ac religiosam cum viro suo dormientem abhominandam crediderit, aut etiam accusandam tamquam non posse coniugatos in regnum dei ingredi, anathema sit.'

const TEXT_2 =
  'III.   DE HIS QUI SE A CARNIBUS ABSTINENT.   Si quis presbiter aut diaconus aut quilibet clericus in ecclesia constitutus a nuptiis et carnibus et vino non pro exercitio se sed pro detestatione abstinuerit, oblitus quia omnia valde bona fecit deus et quia masculum et feminam fecit eos, sed blasphemans accusat ea quae in usu fidelium ad dei ministerium data sunt, aut corrigatur aut deponatur et de ecclesia eiciatur. Similiter et laicus.'

const TEXT_3 =
  'XLIIII.   Si quis episcopus aut presbiter aut diaconus aliquem a communione sine causa reppulerit, aut eicere convicerit, deponatur, eo quod contristaverit innocentem.'

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export const MOCK_QUERIES: QueryListItem[] = [
  {
    file_id: 1,
    filename: 'BAV1341.13v.1.txt',
    text_preview: TEXT_1.slice(0, 80),
    review_status: 'reviewed',
    review_count: 1,
  },
  {
    file_id: 2,
    filename: 'BAV1341.14r.2.txt',
    text_preview: TEXT_2.slice(0, 80),
    review_status: 'unreviewed',
    review_count: 0,
  },
  {
    file_id: 3,
    filename: 'Hat42.148v.3.txt',
    text_preview: TEXT_3.slice(0, 80),
    review_status: 'reviewed',
    review_count: 2,
  },
]

export const MOCK_QUERY_DETAILS: Map<number, QueryDetail> = new Map([
  [
    1,
    {
      file_id: 1,
      filename: 'BAV1341.13v.1.txt',
      text: TEXT_1,
      tokens: tokenize(TEXT_1),
      char_count: TEXT_1.length,
      token_count: tokenize(TEXT_1).length,
    },
  ],
  [
    2,
    {
      file_id: 2,
      filename: 'BAV1341.14r.2.txt',
      text: TEXT_2,
      tokens: tokenize(TEXT_2),
      char_count: TEXT_2.length,
      token_count: tokenize(TEXT_2).length,
    },
  ],
  [
    3,
    {
      file_id: 3,
      filename: 'Hat42.148v.3.txt',
      text: TEXT_3,
      tokens: tokenize(TEXT_3),
      char_count: TEXT_3.length,
      token_count: tokenize(TEXT_3).length,
    },
  ],
])
