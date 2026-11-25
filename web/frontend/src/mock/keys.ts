import type {
  CandidateFile,
  QueryKeyMatch,
  QueryListItem,
} from '../api/queries'
import type { LabelledKeyItem, LabelledKeyListResponse } from '../api/keys'

// Fixture labelled keys: C-, D- and R-shaped keys, edition-style names, and the
// awkward ones real directories have (a double space, a comma).
const KEY_UNITS: [string, number][] = [
  ['CNIC.325.1', 5],
  ['CNIC.325.2', 3],
  ['CNIC.325.4  alt', 2],
  ['CNIC.325.20', 6],
  ['CNEO.315.2', 2],
  ['CNEO.315.6', 4],
  ['CARL.813.12', 3],
  ['CARL.813.42', 2],
  ['CANT.341.6', 1],
  ['DSIR.384.255 cap 7', 5],
  ['DSIR.384.255 cap 8', 2],
  ['DINN.401.3', 4],
  ['DLEO.447.1', 7],
  ['DLEO.447.14', 3],
  ['DGEL.494.12', 2],
  ['DHOR.514.5', 1],
  ['RCAP.789.62', 3],
  ['RCAP.802.5', 2],
  ['RLUD.829.3', 4],
  ['RLUD.829.19', 1],
  ['Can.apost.42', 2],
  ['Can.apost.44', 1],
  ['Can.apost.51', 2],
  ['Capit.Martini.12', 2],
  ['Gangr.c.14', 1],
  ['Mansi 2, 359 B', 1],
  ['Ambrose, De spirit.sanct.3.6 (ed. Faller, 164-65, lines 12-22)', 1],
  ['Source unidentified', 1],
]

const UNIT_TEXTS = [
  'Si quis episcopus aut presbiter aut diaconus aliquem a communione sine causa reppulerit, deponatur, eo quod contristaverit innocentem.',
  'Placuit ut quotienscumque concilium congregandum est, episcopi qui neque aegritudine neque aliqua necessitate impediuntur conveniant.',
  'De his qui ad clerum promoveri debent, ut nullus neophitus ordinetur episcopus nisi probatus per annos.',
  'Item placuit ut si quis nuptias dampnaverit et mulierem fidelem abhominandam putaverit, anathema sit.',
]

const normalize = (value: string) => value.trim().replace(/\s+/g, ' ').toLowerCase()

const naturalCompare = new Intl.Collator('en', { numeric: true }).compare

export function handleLabelledKeys(url: string): LabelledKeyListResponse {
  const params = new URL(url, 'http://localhost').searchParams
  const target = normalize(params.get('search') ?? '')
  const page = parseInt(params.get('page') || '1', 10)
  const pageSize = parseInt(params.get('page_size') || '50', 10)

  const hits = KEY_UNITS.filter(([name]) => normalize(name).includes(target)).sort(
    ([a], [b]) => {
      const aPrefix = normalize(a).startsWith(target)
      const bPrefix = normalize(b).startsWith(target)
      if (aPrefix !== bPrefix) return aPrefix ? -1 : 1
      return naturalCompare(normalize(a), normalize(b))
    },
  )
  const start = (page - 1) * pageSize
  const items: LabelledKeyItem[] = hits
    .slice(start, start + pageSize)
    .map(([key, unit_count]) => ({ key, unit_count }))
  return {
    items,
    total: hits.length,
    page,
    page_size: pageSize,
    has_more: start + pageSize < hits.length,
  }
}

export function handleCandidateDirFiles(dir: string): CandidateFile[] | null {
  const entry = KEY_UNITS.find(([name]) => name === dir)
  if (!entry) return null
  return Array.from({ length: entry[1] }, (_, i) => ({
    filename: `${dir.replace(/[^\w.-]+/g, '_')}_${String.fromCharCode(97 + i)}.txt`,
    text: UNIT_TEXTS[(i + dir.length) % UNIT_TEXTS.length],
  }))
}

const MOCK_MODEL = 'bowphs_LaTa'

// Deterministic stand-in for the model's ranked top ten: ten distinct keys per
// text, drawn from the fixtures by file id.
function mockRanking(fileId: number): QueryKeyMatch[] {
  const names = KEY_UNITS.map(([name]) => name)
  return Array.from({ length: 10 }, (_, i) => ({
    model: MOCK_MODEL,
    rank: i + 1,
    dir_name: names[(fileId * 7 + i * 5) % names.length],
  })).filter(
    (hit, i, all) => all.findIndex((other) => other.dir_name === hit.dir_name) === i,
  )
}

export function filterQueriesByKey(
  queries: QueryListItem[],
  url: string,
): QueryListItem[] | null {
  const params = new URL(url, 'http://localhost').searchParams
  const key = params.get('key')
  if (key === null) return null

  const target = normalize(key)
  const exact = params.get('key_match') === 'exact'
  const model = params.get('key_model')
  const top1 = params.get('key_top1') === 'true'

  const result: QueryListItem[] = []
  for (const query of queries) {
    const matches = mockRanking(query.file_id).filter((hit) => {
      const name = normalize(hit.dir_name)
      return (
        (exact ? name === target : name.startsWith(target)) &&
        (!model || hit.model === model) &&
        (!top1 || hit.rank === 1)
      )
    })
    if (matches.length > 0) result.push({ ...query, key_matches: matches })
  }
  return result
}
