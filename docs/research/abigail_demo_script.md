# Abigail demo script

One-page walkthrough of the reviewer instance at <https://ai.csr.uky.edu>. Roughly 10 minutes.
Everything below is checked against the predictions currently deployed (regenerated after the
Task B split redesign, issue #66).

## 0. Before the call (2 minutes)

1. Open <https://ai.csr.uky.edu> and sign in with the PI/admin account.
2. Confirm the header shows the model picker, and that it has landed on **mT5-base**. There is no
   post-processing toggle any more (issue #94): the app serves one pipeline, SIF+ABTT, for the
   ranking, the highlights and the recorded feedback alike. If a model's prediction list comes up
   empty, its predictions CSV did not reach the host: re-run the deploy with `DATA_RELEASE_TAG`
   set (see `deploy/REVIEWER_PILOT_READINESS.md`).
3. Switch the picker to **LaTa** before you start. mT5-base is where a reviewer lands, but every
   query below runs on LaTa or PhilTa, and the model is named at the top of each section. Say the
   switch out loud rather than doing it silently: "the reviewer's default is mT5-base, I am moving
   to the model these examples were prepared on."
4. Have this page open in a second window for the numbers.

## 1. The one-sentence framing

"The model reads a manuscript fragment and proposes which known text it belongs to. Everything on
screen comes from one corrected pipeline, and the coloured note above the ranked list tells you how
much of an opinion the model actually has."

## 2. Reading the confidence band

The banner above the ranked sources is the first thing to point at, because it is the first thing a
reviewer reads. Thresholds are on the displayed SIF+ABTT similarity and live in one place in the
code (`web/frontend/src/utils/confidenceBands.ts`):

| Displayed similarity | What the reviewer sees | What it means |
|---|---|---|
| below 0.50 | red **Potentially no match**, with **New directory / New file** as the default top option | the ranking below is probably noise; this fragment may belong to a text the corpus does not have |
| 0.50 to 0.70 | amber **Review this match carefully** | genuinely uncertain: read the evidence before accepting or rejecting |
| 0.70 and above | plain **Likely match - verify** | the ordinary case: verify and record |

Worth saying out loud: 0.50 is deliberately conservative. The learned per-model threshold under
SIF+ABTT sits at 0.34 to 0.43 on the training fit, so the red band starts well above where the
model itself would stop calling something a match.

The **New directory / New file** button is the entry point to the reviewer-created-directory loop,
and it is live: it opens a one-field naming form, and creating the directory makes it a scored
candidate on every other query from this point on. Say that the badge on the seed reads
"Awaiting future match" until a reviewer actually files a second document into it -- similarity
alone never flips it, deliberately, so the green state means a human judgement and not an
embedding-space coincidence.

## 3. Query A: what the correction buys (LaTa)

Model **LaTa**. Type `C1525.56v.3` into the sidebar search and open it (file_id 1147).
The text is Serdica canon XVII, `Osius episcopus dixit`.

The app shows the SIF+ABTT row. The rest of this table is a PI-side talking point, not something to
click through: it is the paper's argument, and it is why the deployed pipeline is the corrected one.

| Variant | Top-1 | Score | Runner-up | Gap to runner-up |
|---|---|---|---|---|
| Raw | CANT.328.12 | 0.8541 | CANT.328.5 at 0.8535 | **0.0006** |
| ABTT | CSAR.347.17 | 0.7246 | Can.apost.45 at 0.4053 | 0.319 |
| SIF | CSAR.347.17 | 0.9277 | CANT.328.25 at 0.7579 | 0.170 |
| **SIF+ABTT (deployed)** | CSAR.347.17 | 0.7899 | CVAI.442.8 at 0.3422 | 0.448 |

Uncorrected, the top three candidates span 0.0014 in total (rank 1 to rank 2 is 0.0006), so the
ranking is effectively a coin flip and it lands on Antioch. Every corrected variant instead puts
Serdica canon 17 first, and pulls it clear of the runner-up: by 0.17 under SIF, 0.32 under ABTT and
0.45 under SIF+ABTT. `CSAR` is the Serdica collection, and the query is a Serdica canon, so the
corrected ranking is also the right one.

The size of that gap is variant-specific. SIF alone reorders correctly but still leaves the field
bunched; it takes ABTT, which removes the dominant embedding directions, to open real distance.
That collapse of the uncorrected scores into a narrow band near the top is the anisotropy the
post-processing removes, and it is the reason the reviewer never sees the uncorrected ranking.

On screen: 0.790 at rank 1 and 0.342 at rank 2, so the banner reads **Likely match - verify** and
the rank-2 card is already flagged as a potential non-match. That contrast on one screen is the
band system doing its job.

## 4. Query B: short and unambiguous (PhilTa)

Model **PhilTa**, search `C1525.35r.9` (file_id 1076). Only 28 words, so the whole query fits on
screen: `Cap. XLVII. De his, qui in aegritudine baptizantur.`

Deployed top-1 is CLAO.300.47 at 0.599, which lands in the amber **Review this match carefully**
band. The query is chapter 47 and the answer is Laodicea canon 47, so this is a good query to hand
over to Abigail: she can judge it herself without reading any Latin, and it shows that "amber" means
uncertain rather than wrong. For the record, uncorrected the same query ranks CNEO.315.12 first at
0.747 and pushes CLAO.300.47 down to third.

## 5. Query C (backup): the coin flip (PhilTa)

Model **PhilTa**, search `BAV1341.16r.7` (file_id 40), Serdica canon XXI.
Deployed: CSAR.347.21 at 0.544 with the next candidate at 0.225, so the banner is amber and the gap
is enormous. Uncorrected, CANC.314.3 and CSAR.347.21 sat 0.0003 apart.

A second backup, if a query fails to load: **LaTa**, `C1525.54r.2` (file_id 1132). The deployed
pipeline picks Serdica canon 2, which is what the query is; uncorrected it picked a Nicaea canon.

## 6. What to show in the highlight view

With a candidate expanded, the **Highlights** and **Connections** view modes tint the query and
candidate text by token similarity, computed from the same SIF+ABTT weighting as the ranking. Point
at the function words shared by every canon (`episcopus`, `dixit`, `qui`, `est`) and note how little
weight they carry: that is what "down-weights frequent words" means, made visible. The uncorrected
comparison, where those same function words soak up the highlighting, is a slide-worthy point rather
than a click.

The third view mode, **Attribution** (integrated gradients and the other six attribution methods),
is PI-only and needs a precomputed artifact for the exact query/candidate pair. Most of the 128
artifacts cover the labelled canon pairs used in the paper, so on an arbitrary live query the
toggle is absent.

The four demo queries above are the exception. Eight artifacts (issue #53) cover each demo query
against both the directory the uncorrected pipeline picks and the directory the corrected one picks,
under the model listed for that query:

| Query | Model | Uncorrected top-1 | Deployed top-1 |
|---|---|---|---|
| `C1525.56v.3` | LaTa | CANT.328.12 | CSAR.347.17 |
| `C1525.35r.9` | PhilTa | CNEO.315.12 | CLAO.300.47 |
| `BAV1341.16r.7` | PhilTa | CANC.314.3 | CSAR.347.21 |
| `C1525.54r.2` | LaTa | CNIC.325.16 | CSAR.347.2 |

Each artifact still carries all four variants, but the app reads the SIF+ABTT panel, which is the
one that matches the ranking on screen. It is cleaned in the deployed SIF-pooled space with the D
that variant's own sweep chose (LaTa layer 1: SIF D=3), so the panel removes the directions the
ranking removed. One caveat worth knowing rather than discovering:

* Attribution truncates at 256 tokens while the retrieval embeddings were pooled at 512. Query A
  (`C1525.56v.3`) is 294 tokens, so roughly the last 13% of it is not shown in the attribution
  panel. The other three queries are 51, 144 and 147 tokens and are fully covered.

That caveat does not affect the Highlights and Connections view modes in the paragraph above,
which are computed live.

One detail only worth raising if asked how the panels are built: the per-token integrated-gradient
scores behind the IG and OT methods are attributions of a mean-pooled target, and the SIF variants
express SIF-ness through the token weights rather than by re-running IG. The four other methods
(BERTScore, DLA and the two attention ones) do not use them at all.

Stay on the demo queries if Abigail asks to see attribution. Off the demo set the toggle will not
appear, and that is expected: an artifact costs a GPU pass per pair.

## 7. Notes reload and multi-select

On Query B:

1. Tick **Select multiple** above the rank pills.
2. Click rank 1 and rank 2. Both stay lit. This is the answer shape reviewers asked for when two
   candidate directories are both plausible readings of the same fragment.
3. Type a note, for example `Both Laodicea 47 and Neocaesarea 12 are defensible; 47 matches the
   chapter number.`
4. Submit.
5. Navigate to another query and come back. The note and both selected pills are prefilled, with
   the hint "Loaded from your last submitted review".

Every submitted row still records which pipeline it was given for, so the feedback log stays
interpretable if a future deployment ever serves a different one. Reviewers no longer have to think
about that: there is one ranking to review, not four.

## 8. Expect the rankings to have moved

Say this before Abigail compares against anything she wrote down earlier. The predictions on this
instance were regenerated from the current Task B split, replacing the frozen pre-variant file.
Against that old file, SIF+ABTT top-1 changed for **35.5% of queries under LaTa** and **16.4% under
PhilTa**. The demo queries above happen to be stable, but any note she took against an
earlier session may point at a rank that has moved.

## 9. If something breaks

| Symptom | Cause | Say |
|---|---|---|
| Prediction list empty for a model | that model's predictions CSV missing on host | "one data file did not sync, the other models work" |
| New-directory CTA refuses with "already started a directory" | this document already seeds one | expected, one directory per document |
| Reviewer directories never appear as candidates | q-q matrices missing from the data release | "one data file did not sync"; re-deploy with a newer `DATA_RELEASE_TAG` |
| Attribution toggle absent off the demo set | no IG artifact for this pair | expected, see section 6 |
| Attribution toggle absent *on* a demo pair | data release predates issue #53 | re-run the deploy with the newer `DATA_RELEASE_TAG` |
| Query text blank | canon corpus missing on host | fall back to a different query |

Deploy state, workflow runs and the data payload mechanism are documented in
`deploy/REVIEWER_PILOT_READINESS.md`.
