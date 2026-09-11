import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AppProvider } from '../../contexts/AppContext'
import { TokenProvider } from '../../contexts/TokenContext'
import { TokenRefProvider } from '../connections/TokenRefRegistry'
import DocumentPanel from './DocumentPanel'
import {
  PROVENANCE_TERMS,
  provenanceOf,
  type DocumentProvenance,
} from '../../utils/documentProvenance'

/**
 * Panel provenance (issue #162).
 *
 * The right-hand caption used to read "Predicted Source" for everything on it,
 * which is a corpus claim the app cannot make about a reviewer's provisional
 * grouping of documents that arrived unlabeled. These tests pin the three
 * distinctions and, just as importantly, pin the honest fallback: an
 * unidentified candidate must not be captioned as a labelled reference.
 */

const TOKENS = [{ text: 'Osius', index: 0, category: 'content' }]

function renderPanel(
  props: Partial<React.ComponentProps<typeof DocumentPanel>> = {},
) {
  return render(
    <AppProvider>
      <TokenProvider>
        <TokenRefProvider>
          <DocumentPanel
            side="candidate"
            filename="CSAR.347.17.txt"
            tokens={TOKENS}
            evidenceOwner="provenance-fixture"
            {...props}
          />
        </TokenRefProvider>
      </TokenProvider>
    </AppProvider>,
  )
}

describe('provenanceOf', () => {
  it('takes the current prediction as authoritative', () => {
    expect(provenanceOf({ source: 'model', dirName: 'CSAR.347.17' })).toBe(
      'labeled_reference',
    )
    expect(
      provenanceOf({ source: 'reviewer', dirName: 'reviewer-dir-1' }),
    ).toBe('reviewer_group')
  })

  it('reads a bare directory name off the backend prefix rule', () => {
    // Only the prefix is disjoint by construction (web/services/reviewer_dirs.py
    // routes correct_dir on the same test), so it is a rule, not an inference
    // from the label or the score.
    expect(provenanceOf({ dirName: 'reviewer-dir-42' })).toBe('reviewer_group')
    expect(provenanceOf({ dirName: 'CSAR.347.17' })).toBe('labeled_reference')
  })

  it('stays unknown when the caller knows neither', () => {
    expect(provenanceOf({})).toBe('unknown')
    expect(provenanceOf({ source: null, dirName: null })).toBe('unknown')
  })

  it('never lets a reviewer group read as a labelled reference', () => {
    // The one substitution that would put a colleague's provisional judgement
    // on screen as a corpus fact.
    const cases: Array<Parameters<typeof provenanceOf>[0]> = [
      { source: 'reviewer' },
      { source: 'reviewer', dirName: 'CSAR.347.17' },
      { dirName: 'reviewer-dir-7' },
    ]
    for (const candidate of cases) {
      expect(provenanceOf(candidate)).not.toBe('labeled_reference')
    }
  })
})

describe('document panel captions', () => {
  it('names the left panel as the query witness whatever it is passed', () => {
    renderPanel({ side: 'query', filename: 'query-7.txt', provenance: 'labeled_reference' })
    expect(screen.getByTestId('document-provenance-query').textContent).toBe(
      PROVENANCE_TERMS.query,
    )
  })

  it('names a model candidate as a labelled-reference witness, with its rank', () => {
    renderPanel({ provenance: 'labeled_reference', rank: 3 })
    expect(screen.getByTestId('document-provenance-candidate').textContent).toBe(
      `${PROVENANCE_TERMS.labeled_reference} · Rank 3`,
    )
  })

  it('names a reviewer-group member as originally unlabeled', () => {
    renderPanel({ provenance: 'reviewer_group', rank: 11 })
    const caption = screen.getByTestId('document-provenance-candidate').textContent
    expect(caption).toBe(`${PROVENANCE_TERMS.reviewer_group} · Rank 11`)
    expect(caption).toContain('originally unlabeled')
  })

  it('claims nothing about a candidate opened without a prediction', () => {
    // The examples gallery opens an off-list directory with no rank and no
    // prediction metadata. Silence is correct here; "Predicted Source" was not.
    renderPanel({ dirLabel: 'CSAR.347.17' })
    expect(screen.getByTestId('document-provenance-candidate').textContent).toBe(
      PROVENANCE_TERMS.unknown,
    )
  })

  it('never captions any panel "Predicted Source"', () => {
    const provenances: DocumentProvenance[] = [
      'labeled_reference',
      'reviewer_group',
      'unknown',
    ]
    for (const provenance of provenances) {
      const { unmount } = renderPanel({ provenance, rank: 1 })
      expect(document.body.textContent).not.toContain('Predicted Source')
      unmount()
    }
  })

  it('keeps the caption on the header while the panel is still loading', () => {
    renderPanel({ provenance: 'reviewer_group', rank: 11, loading: true })
    expect(screen.getByTestId('document-provenance-candidate').textContent).toContain(
      PROVENANCE_TERMS.reviewer_group,
    )
  })
})
