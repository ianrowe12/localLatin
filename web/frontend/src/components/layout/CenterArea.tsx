import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../../contexts/AppContext'
import { usePredictionState } from '../../contexts/PredictionContext'
import {
  useQueryDetail,
  useCandidateDirFiles,
  fileHasText,
  type CandidateFile,
} from '../../api/queries'
import { TokenRefProvider } from '../connections/TokenRefRegistry'
import ConnectionOverlay from '../connections/ConnectionOverlay'
import DocumentPanel from '../document/DocumentPanel'
import MemberEvidenceBar from '../evidence/MemberEvidenceBar'
import {
  attributionAppliesToWitness,
  displayedWitnessKey,
  memberEvidenceKey,
  memberEvidenceVisible,
  resolveMemberEvidence,
} from '../evidence/memberEvidence'
import { useSelectedMember } from '../evidence/useSelectedMember'
import { provenanceOf } from '../../utils/documentProvenance'
import AwaitingMatchBadge from '../predictions/AwaitingMatchBadge'
import DraggableDivider from './DraggableDivider'
import { buildWordMatchMap } from '../../utils/wordSimilarity'
import { useTokenMap, type TokenMapResponse, type TopMatch } from '../../api/tokenMap'
import { toAttributionVariant } from '../../api/variants'
import { useTokens, WitnessTokenScope } from '../../contexts/TokenContext'
import { METHODS } from '../common/AttributionMethodSelector'

export default function CenterArea() {
  const [splitPercent, setSplitPercent] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const queryScrollRef = useRef<HTMLDivElement>(null)
  const candidateScrollRef = useRef<HTMLDivElement>(null)

  const {
    activeQueryId,
    activePredictionRank,
    activeModel,
    activeVariant,
    overrideCandidateDir,
    setOverrideCandidateDir,
  } = useApp()

  // Fetch query detail. Its own render-time key guard means the text on screen
  // is this query's text or nothing -- never the previous document's words
  // beside the new document's ranking (issue #156).
  const queryDetail = useQueryDetail(activeQueryId)

  // The one shared ranking for the current query/model/variant. The provider
  // owns the request; this view, the prediction list and the assessment panel
  // read the same result, so they cannot disagree about what is on screen.
  const predictions = usePredictionState()

  // Derive current prediction.
  //
  // The identity check that used to live here -- comparing the response's
  // `model` and `variant` against the selection, because `usePredictions`
  // cleared its data an effect too late -- now lives in the shared state, which
  // refuses to expose old-key data from the first render after a change and
  // validates the response's identity before exposing it at all. What remains
  // here is the rank lookup.
  //
  // Looked up BY RANK, not by array index. Reviewer directories are anchored at
  // rank 11 regardless of how many model candidates came back, so the list can
  // have a gap in it and index arithmetic would pair a rank with the wrong card.
  const currentPrediction = predictions.getByRank(activePredictionRank)

  // Derive candidate info — override wins over the normal prediction path.
  //
  // `overrideCandidateDir` is already bound to the query it was chosen for
  // (AppContext, issue #156): it reads null from the first render for any other
  // document, so a gallery example cannot survive a navigation into the next
  // document's evidence, its file request or its token map.
  const candidateDir = overrideCandidateDir ?? currentPrediction?.dir_name ?? null

  /**
   * What kind of witness is in the candidate panel (issue #162).
   *
   * Derived by the SAME override-wins rule as `candidateDir` above, so the
   * caption can never describe one document while the panel shows another. In
   * override mode the ranked prediction is not consulted at all: a gallery
   * example carries its own identity, and a rank the reviewer happens to have
   * selected underneath it is about a different directory entirely.
   *
   * For a ranked candidate the answer is the server's `source`, and
   * `currentPrediction` is null until the shared state exposes a result for the
   * current query, model, pipeline and generation -- so a stale ranking cannot
   * caption the panel either. An override has no prediction behind it, so
   * `provenanceOf` falls back to the backend's own `reviewer-dir-` partition,
   * the same test its candidate-files route resolves the directory by.
   */
  const candidateProvenance = useMemo(
    () =>
      provenanceOf(
        overrideCandidateDir !== null
          ? { dirName: overrideCandidateDir }
          : {
              source: currentPrediction?.source,
              dirName: currentPrediction?.dir_name,
            },
      ),
    [overrideCandidateDir, currentPrediction],
  )

  // For override-mode candidates, fetch the directory's files on demand.
  // (Regular predictions already carry candidate_files in their payload.)
  const overrideCandidateFiles = useCandidateDirFiles(overrideCandidateDir)

  // Every witness behind the panel, by whichever route it arrived.
  const candidateWitnesses = useMemo<CandidateFile[]>(() => {
    if (overrideCandidateDir) return overrideCandidateFiles.data ?? []
    return currentPrediction?.candidate_files ?? []
  }, [overrideCandidateDir, overrideCandidateFiles.data, currentPrediction])

  /**
   * Which member of the candidate group is on screen (issue #163).
   *
   * Scoped by `memberEvidenceKey`, which carries the document, model,
   * pipeline, request generation, directory and provenance -- deliberately
   * NOT the rank, because the same rank names a different directory after a
   * refresh and a choice scoped to a rank would survive into a group that
   * never contained that member.
   *
   * A gallery override has no prediction behind it, so it has no supporting
   * member and no group maximum to attribute: it keeps the directory-file
   * inspection path #156 gave it, and the key still changes on entering and
   * leaving so a choice cannot cross the boundary.
   */
  const memberIdentity = useMemo(
    () =>
      memberEvidenceKey({
        queryId: activeQueryId,
        model: activeModel,
        variant: activeVariant,
        generation: predictions.generation,
        dirName: candidateDir,
        source:
          overrideCandidateDir !== null ? undefined : currentPrediction?.source,
      }),
    [
      activeQueryId,
      activeModel,
      activeVariant,
      predictions.generation,
      candidateDir,
      overrideCandidateDir,
      currentPrediction,
    ],
  )

  const memberSelection = useSelectedMember(memberIdentity)

  const memberEvidence = useMemo(
    () =>
      overrideCandidateDir !== null
        ? null
        : resolveMemberEvidence(currentPrediction, memberSelection.filename),
    [overrideCandidateDir, currentPrediction, memberSelection.filename],
  )

  // The one witness DocumentPanel puts on screen. For a ranked candidate that
  // is the reviewer's choice, defaulting to the witness the server designated
  // as the number's support; the positional fallback is for the override path
  // and for a group whose members could not be resolved at all.
  const candidateFile: CandidateFile | null =
    memberEvidence?.selected ?? candidateWitnesses[0] ?? null

  // Where the displayed witness sits in the response's member order, so copy
  // about "the other files" can exclude the right one.
  const displayedPosition = memberEvidence?.selected?.position ?? 0

  const candidateLoading = overrideCandidateDir
    ? overrideCandidateFiles.loading
    : predictions.isLoading

  // Candidate tokens: simple tokenization from candidate file text
  const candidateTokens = useMemo(() => {
    if (candidateFile?.text) {
      return candidateFile.text
        .split(/\s+/)
        .filter((t) => t.length > 0)
        .map((t, i) => ({
          text: t,
          index: i,
          category: /^[.,;:!?()\[\]]+$/.test(t) ? 'punctuation' : 'content',
        }))
    }
    return undefined
  }, [candidateFile])

  /**
   * The boundary the candidate panel's animation is allowed to live inside.
   *
   * Everything that changes WHICH PAIR is under review: the document, the
   * model, the pipeline, the request generation, and whether a gallery example
   * is open. Not the rank, because moving between candidates of one settled
   * ranking is the transition the crossfade exists for.
   *
   * This keys the `AnimatePresence` element itself, so a change here unmounts
   * the presence owner and every panel it was holding, in the same commit and
   * with no exit animation to sit through. Keying only the inner child cannot
   * do that: a key change is what STARTS an exit, and the outgoing panel --
   * with its old token map and its old highlights -- stays mounted for the
   * length of it. Two gallery examples that share a directory are the case
   * that proves the difference: their evidence identities differ only by
   * query, both are non-null, and nothing in between is ever null, so there is
   * no loading render to rescue the boundary.
   */
  const presenceBoundary = [
    activeQueryId ?? 'no-query',
    activeModel,
    activeVariant,
    `g${predictions.generation}`,
    overrideCandidateDir ?? 'ranked',
  ].join('|')

  /**
   * Identity of the evidence actually on screen, or null when there is none.
   *
   * Within one boundary this distinguishes one candidate from the next, and
   * null means there is nothing to animate at all -- so the panel renders
   * outside `AnimatePresence` entirely and cannot be retained by an exit.
   *
   * For a ranked candidate the identity comes from the provider and therefore
   * carries the request generation. For a gallery inspection it carries the
   * query, and it stays null until the directory's files have actually
   * arrived, so a failed fetch animates nothing.
   */
  const evidenceIdentity = useMemo(() => {
    if (activeQueryId === null) return null
    if (overrideCandidateDir !== null) {
      return overrideCandidateFiles.data === null
        ? null
        : `override|${activeQueryId}|${overrideCandidateDir}`
    }
    if (currentPrediction === null) return null
    return predictions.identityOf(currentPrediction)
  }, [
    activeQueryId,
    overrideCandidateDir,
    overrideCandidateFiles.data,
    currentPrediction,
    predictions,
  ])

  // Whether the app has actually SEEN this candidate's files. A ranking carries
  // them inline, so a prediction is proof; a gallery inspection has to fetch
  // them, and a fetch can fail.
  const candidateEvidenceLoaded = overrideCandidateDir
    ? overrideCandidateFiles.data !== null
    : currentPrediction !== null

  /**
   * Why the candidate pane has nothing in it, when there is a knowable reason.
   *
   * A ranking that failed, was excluded or came back empty leaves this panel
   * blank, and a blank panel reads as a verdict on the document. It is not one.
   *
   * The blank-text branch asks about THE WITNESS ON SCREEN, not about the
   * directory. `web/routers/predictions.py` fills `candidate_files` with
   * `texts.get(fname, "")`, and the panel shows one of them, so a directory
   * whose displayed witness is empty and whose others are readable used to
   * render an unexplained blank pane while a whole-directory check said there
   * was text. The two cases now get two different sentences, because they are
   * two different facts. Since the reviewer can change which member is shown
   * (issue #163), "the other files" means every position except the displayed
   * one, not every position except the first.
   *
   * And a request that never arrived is a third fact again. Only files this app
   * has actually read can support a claim about what a manuscript contains.
   */
  const candidateEvidenceNote = useMemo(() => {
    if (overrideCandidateDir) {
      // A failed fetch is not a blank manuscript. Saying "no readable text"
      // here would report a transport failure as a fact about the source.
      if (overrideCandidateFiles.error !== null) {
        return `The files for this example candidate did not load (${overrideCandidateFiles.error}), so there is nothing to compare yet. That is a failed request, not an empty directory: leave the example and open it again to retry.`
      }
    } else {
      // The phase notes are about the ranking, so a gallery inspection -- which
      // carries its own candidate regardless of the ranking -- is exempt.
      if (predictions.phase === 'error') {
        return 'The ranking for this document did not load, so there is no candidate to compare against. See the prediction list for details.'
      }
      if (predictions.phase === 'excluded') {
        return 'The retrieval run excluded this document for this model, so it has no candidates to compare against.'
      }
      if (predictions.phase === 'empty') {
        return 'No candidates came back for this document, and no reason was recorded.'
      }
    }
    // A gallery inspection can be just as blank, so this half is not exempt --
    // but only once its files are in hand.
    if (
      candidateDir !== null &&
      !candidateLoading &&
      candidateEvidenceLoaded &&
      !fileHasText(candidateFile)
    ) {
      if (
        candidateWitnesses.some(
          (file, position) => position !== displayedPosition && fileHasText(file),
        )
      ) {
        return `The file shown here (${candidateFile?.filename ?? 'the first in the directory'}) has no readable text in this deployment, so there is nothing to compare word by word. Other files in this directory do carry text.`
      }
      return 'This candidate directory has no readable text in this deployment, so there is nothing to compare word by word.'
    }
    return null
  }, [
    predictions.phase,
    overrideCandidateDir,
    overrideCandidateFiles.error,
    candidateDir,
    candidateLoading,
    candidateEvidenceLoaded,
    candidateFile,
    candidateWitnesses,
    displayedPosition,
  ])

  // Word-match similarity for cross-panel highlighting
  const wordMatchMap = useMemo(() => {
    if (!queryDetail.data?.tokens || !candidateTokens) return null
    return buildWordMatchMap(queryDetail.data.tokens, candidateTokens)
  }, [queryDetail.data?.tokens, candidateTokens])

  const { selectedMethod, viewMode, setAvailableMethods, clearAllPins } = useTokens()

  // The highlights are computed from the same pipeline as the ranking, so the
  // evidence a reviewer reads always belongs to the ranking they are judging.
  // The artifacts call the uncorrected variant "baseline" where the prediction
  // CSVs call it "raw" -- toAttributionVariant is the one place that bridges
  // the two vocabularies.
  const attributionVariant = toAttributionVariant(activeVariant)
  // The reviewer-facing method name, not the artifact slug: a PI-admin on
  // attention_weighted should read "Attn-W", the label the method picker uses.
  const methodLabel =
    METHODS.find((m) => m.key === selectedMethod)?.label ?? selectedMethod

  const tokenMapResult = useTokenMap(
    activeQueryId,
    candidateDir,
    activeModel || undefined,
    selectedMethod,
    attributionVariant,
  )

  // Keep the AttributionMethodSelector's available list in sync with whatever
  // pair is currently loaded. Also clear any leftover pins from a previous
  // session/pair so each new pair starts with hover-only behavior (no sticky
  // lines from stale pinnedTokens state).
  const availableMethodsKey = tokenMapResult.data?.example_id
  useEffect(() => {
    setAvailableMethods(tokenMapResult.data?.available_methods ?? [])
    clearAllPins()
  }, [availableMethodsKey, tokenMapResult.data, setAvailableMethods, clearAllPins])

  // When an attribution method is selected and a matching matrix is present
  // in the token-map payload, swap similarity_matrix and recompute top_matches
  // so DocumentPanel highlights reflect the selection. Otherwise fall through
  // to the raw token map (or the word-match fallback if the map is absent).
  //
  // Different attribution methods have wildly different value ranges (IG ~±0.03,
  // BERTScore [0,1], OT ~[0,0.07], attention ~[0,0.015], DLA baseline ~[0,0.9]),
  // so we take |value| and divide by the matrix max to put every method on a
  // common [0,1] scale. DocumentPanel's existing thresholding expects this.
  const selectedMatrix = useMemo(() => {
    const data = tokenMapResult.data
    if (!data || !selectedMethod || !data.pair_matrices) return undefined
    return data.pair_matrices[selectedMethod]?.[attributionVariant]
  }, [tokenMapResult.data, selectedMethod, attributionVariant])

  // The payload's own `similarity_matrix` is plain cosine over the raw hidden
  // states: identical for every attribution method and every post-processing
  // variant. Falling through to it when the selected cell of `pair_matrices`
  // is absent is what made the highlights look frozen across a switch (issue
  // #73) -- the view had silently stopped showing attribution at all. Since
  // issue #94 removed the variant picker, the switch a PI-admin can still make
  // is the method one, and the failure mode is identical. Say so instead of
  // showing a grid that cannot answer the question just asked.
  //
  // `available_methods` is the test, NOT the shape of `pair_matrices`. Both
  // are declared with a default factory (web/models.py:117,121), so the API
  // always serialises both keys -- the original `pair_matrices != null` was
  // true for every real response and false only under `npm run dev:mock`.
  // Emptiness does not separate the two cases either: `load_token_map` builds
  // `pair_matrices` with `setdefault(m, {})[v]` only when a matrix is actually
  // added, so an artifact that carries attribution but not for the requested
  // variant emits exactly the same `{}` as one that carries none at all.
  // `available_methods` describes the whole artifact regardless of the
  // ?method=/?variant= filter, which is precisely the distinction wanted:
  // empty means the artifact predates attribution, and for those the cosine
  // map is the honest whole story, so they keep the old behaviour.
  const artifactHasAttribution =
    (tokenMapResult.data?.available_methods?.length ?? 0) > 0

  const attributionUnavailable =
    tokenMapResult.data != null &&
    artifactHasAttribution &&
    selectedMethod != null &&
    selectedMatrix === undefined

  // On current main the notice above is defence in depth: #90's resolver keys
  // artifacts per layer and filters on `variants_available`, so a pair that
  // resolves at all carries the variant it resolved for. What a reviewer does
  // hit is the other branch -- `resolve_example_id` returning None, i.e. a 404
  // -- which until now rendered an unexplained empty evidence panel. Both
  // states mean the same thing to the reader, so they share one message.
  const attributionMissing = tokenMapResult.error != null

  /**
   * Does the loaded artifact describe the witness actually on screen?
   *
   * `/api/token_map` is keyed by candidate DIRECTORY, so its matrix is about
   * whichever file the artifact was built from -- `candidate_path` is the only
   * field that says which. Painting it over another member would attribute one
   * witness's model evidence to a different manuscript, which is precisely the
   * confusion #163 exists to remove, and the answer does not change because
   * the reviewer happened to pick a member the artifact does not cover.
   */
  const attributionScope = useMemo(
    () =>
      attributionAppliesToWitness({
        candidatePath: tokenMapResult.data?.candidate_path ?? null,
        dirName: candidateDir,
        filename: candidateFile?.filename ?? null,
        source:
          overrideCandidateDir !== null ? undefined : currentPrediction?.source,
      }),
    [
      tokenMapResult.data,
      candidateDir,
      candidateFile,
      overrideCandidateDir,
      currentPrediction,
    ],
  )

  const effectiveTokenMap = useMemo(() => {
    const data = tokenMapResult.data
    if (!data) return wordMatchMap
    // Hoisted out of the `!selectedMatrix` branch it used to live in, which it
    // already implies: `attributionUnavailable` is only ever true when the
    // selected cell is missing.
    if (attributionUnavailable) return null
    // Word overlap is lexical and belongs to whatever text is on screen, so it
    // stays. The artifact does not, and MemberEvidenceBar says which of the two
    // the marks below are.
    if (!attributionScope.applicable) return wordMatchMap
    if (!selectedMatrix) return data
    const selected = selectedMatrix

    // Per-pair |max| over all cells; abs+normalize to [0,1].
    let absMax = 0
    for (let qi = 0; qi < selected.length; qi++) {
      const row = selected[qi]
      if (!row) continue
      for (let ci = 0; ci < row.length; ci++) {
        const v = Math.abs(row[ci])
        if (v > absMax) absMax = v
      }
    }
    const denom = absMax > 1e-12 ? absMax : 1
    const normalized: number[][] = selected.map((row) =>
      row ? row.map((s) => Math.abs(s) / denom) : [],
    )

    const topMatches: Record<string, TopMatch[]> = {}
    for (let qi = 0; qi < normalized.length; qi++) {
      const row = normalized[qi]
      if (!row) continue
      const indexed: TopMatch[] = row.map((s, ci) => ({
        candidate_idx: ci,
        score: s,
      }))
      indexed.sort((a, b) => b.score - a.score)
      topMatches[String(qi)] = indexed.slice(0, 3)
    }
    const swapped: TokenMapResponse = {
      ...data,
      similarity_matrix: normalized,
      top_matches: topMatches,
    }
    return swapped
  }, [
    tokenMapResult.data,
    selectedMatrix,
    attributionUnavailable,
    attributionScope,
    wordMatchMap,
  ])

  // What the marks on screen actually are. Word overlap is returned by
  // identity above, so this is an exact test rather than a reconstruction.
  const lexicalHighlighting =
    effectiveTokenMap !== null && effectiveTokenMap === wordMatchMap

  // The scope note belongs beside the marks it qualifies, so it is withheld
  // when the panels are shading nothing at all.
  const shownAttributionScope = effectiveTokenMap !== null ? attributionScope : null

  // The bar prints the number with a label that says what it covers. The
  // document header prints the same figure as a bare "Similarity", which for a
  // group maximum names the wrong thing as well as saying it twice.
  const barCarriesScore = memberEvidenceVisible(
    memberEvidence,
    shownAttributionScope,
    { lexicalHighlighting },
  )

  /**
   * The pair that pins, auto-highlights and hover currently describe.
   *
   * A pin means "this query token matches that candidate token", and the
   * candidate token is a different word as soon as another member is shown, so
   * the witness -- not just the directory -- is part of the identity.
   */
  const witnessScope = displayedWitnessKey(
    evidenceIdentity ?? presenceBoundary,
    memberEvidence?.selected?.key ?? candidateFile?.filename ?? null,
  )

  // Note: we deliberately do NOT auto-pin top-attribution tokens on pair entry.
  // Connection lines are drawn purely on hover (see useConnectionState). Token
  // background shading already encodes the selected attribution matrix via
  // DocumentPanel's row/col-max derivation from the swapped similarity_matrix.

  const handleDrag = useCallback((newPercent: number) => {
    setSplitPercent(newPercent)
  }, [])

  // Built once and rendered from either branch below, so the animated and the
  // plain path show the same panel rather than two drifting copies of it. The
  // member strip travels WITH the panel: during a crossfade it must describe
  // the witness beside it, not the one arriving.
  const candidatePanel = (
    <>
      <MemberEvidenceBar
        evidence={memberEvidence}
        onSelectWitness={memberSelection.select}
        attribution={shownAttributionScope}
        lexicalHighlighting={lexicalHighlighting}
      />
      <DocumentPanel
        side="candidate"
        filename={candidateFile?.filename}
        dirLabel={
          // Reviewer directories show their human label; the opaque
          // reviewer-dir-N id would tell the reviewer nothing.
          overrideCandidateDir ??
          currentPrediction?.label ??
          currentPrediction?.dir_name
        }
        score={
          overrideCandidateDir || barCarriesScore
            ? undefined
            : currentPrediction?.score
        }
        rank={overrideCandidateDir ? undefined : activePredictionRank}
        provenance={candidateProvenance}
        tokens={candidateTokens}
        tokenMap={effectiveTokenMap}
        loading={candidateLoading}
        scrollRef={candidateScrollRef}
      />
    </>
  )

  return (
    <TokenRefProvider>
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {(attributionUnavailable || attributionMissing) && (
          <div
            role="status"
            className="px-3 py-1.5 bg-stone-100 dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 text-xs text-stone-600 dark:text-stone-300 flex-shrink-0"
          >
            {attributionUnavailable ? (
              <>
                No {methodLabel} attribution for this pair. Highlights are off
                rather than falling back to a plain similarity grid, which
                would look the same for every method.
              </>
            ) : (
              <>
                No attribution available for this pair. Highlights are off; the
                ranking above is unaffected.
              </>
            )}
          </div>
        )}
        <div
          ref={containerRef}
          className="relative flex-1 flex overflow-hidden"
        >
          {/* Both panels and the connection overlay read pins, auto-highlights
              and hover through this scope, so a member change empties them in
              the same commit rather than one frame later (issue #163). */}
          <WitnessTokenScope witness={witnessScope}>
            {/* Query panel */}
            <div
              data-tour="query-panel"
              style={{ width: `${splitPercent}%` }}
              className="h-full overflow-hidden flex flex-col"
            >
              <DocumentPanel
                side="query"
                filename={queryDetail.data?.filename}
                tokens={queryDetail.data?.tokens}
                tokenMap={effectiveTokenMap}
                loading={queryDetail.loading}
                scrollRef={queryScrollRef}
                // The badge reports the fate of directories *this* document
                // seeded, so it belongs on the query panel, not on a candidate.
                badge={
                  <AwaitingMatchBadge
                    seededDirs={predictions.seededDirs}
                  />
                }
              />
            </div>

            <DraggableDivider onDrag={handleDrag} />

            {/* Candidate panel */}
            <div
              data-tour="candidate-panel"
              style={{ width: `${100 - splitPercent}%` }}
              className="h-full overflow-hidden flex flex-col"
            >
              {overrideCandidateDir && (
                <div className="px-3 py-1.5 bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800/40 text-xs text-amber-800 dark:text-amber-200 flex items-center justify-between flex-shrink-0">
                  <span>Viewing example pair · candidate may be outside top-10</span>
                  <button
                    type="button"
                    onClick={() => setOverrideCandidateDir(null)}
                    className="px-2 py-0.5 rounded hover:bg-amber-100 dark:hover:bg-amber-900/40 font-medium"
                  >
                    Exit
                  </button>
                </div>
              )}
              {/* Why the panel is empty, when it is empty for a knowable reason.
                  An unexplained blank pane reads as "the model found nothing",
                  and none of these states is that (issue #156). Which of them can
                  appear during a gallery inspection is decided where the note is
                  built, not here. */}
              {candidateEvidenceNote && (
                <div
                  role="status"
                  data-testid="candidate-evidence-note"
                  className="px-3 py-1.5 bg-stone-100 dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700 text-xs text-stone-600 dark:text-stone-300 flex-shrink-0"
                >
                  {candidateEvidenceNote}
                </div>
              )}
              {/* The animated wrapper is keyed on the pair boundary, so a new
                  document, model, pipeline or request replaces the presence owner
                  outright rather than asking it to animate its old panel away.
                  Inside one boundary the inner key still crossfades between
                  candidates of the same settled ranking, which is the only
                  transition that can never show evidence under a new identity. */}
              {evidenceIdentity !== null ? (
                <AnimatePresence key={presenceBoundary} mode="wait">
                  <motion.div
                    key={evidenceIdentity}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="h-full flex flex-col"
                  >
                    {candidatePanel}
                  </motion.div>
                </AnimatePresence>
              ) : (
                <div className="h-full flex flex-col">{candidatePanel}</div>
              )}
            </div>

            {/* SVG connection overlay */}
            {viewMode !== 'heatmap' && (
              <ConnectionOverlay
                containerRef={containerRef}
                leftPanelRef={queryScrollRef}
                rightPanelRef={candidateScrollRef}
              />
            )}
          </WitnessTokenScope>
        </div>
      </div>
    </TokenRefProvider>
  )
}
