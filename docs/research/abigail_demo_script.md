# Abigail demo script

One-page walkthrough of the reviewer instance at <https://ai.csr.uky.edu>. Roughly 10 minutes.
Everything below is checked against the predictions currently deployed (regenerated after the
Task B split redesign, issue #66).

## 0. Before the call (2 minutes)

1. Open <https://ai.csr.uky.edu> and sign in with the PI/admin account.
2. Confirm the header shows the model picker **and** the four-way post-processing toggle:
   `Raw | ABTT | SIF | SIF+ABTT`. SIF+ABTT is the default. If any of the four is greyed out, a
   predictions CSV did not reach the host: re-run the deploy with `DATA_RELEASE_TAG` set
   (see `deploy/REVIEWER_PILOT_READINESS.md`).
3. Have this page open in a second window for the numbers.

## 1. The one-sentence framing

"The model reads a manuscript fragment and proposes which known text it belongs to. The toggle
changes only the post-processing applied to the embeddings, not the model. Watch what happens to
the ranking."

## 2. Query A: the three-way tie (LaTa)

Model **LaTa**. Type `C1525.56v.3` into the sidebar search and open it (file_id 1147).
The text is Serdica canon XVII, `Osius episcopus dixit`.

| Variant | Top-1 | Score | Runner-up | Gap to runner-up |
|---|---|---|---|---|
| Raw | CANT.328.12 | 0.8541 | CANT.328.5 at 0.8535 | **0.0006** |
| ABTT | CSAR.347.17 | 0.7246 | Can.apost.45 at 0.4053 | 0.319 |
| SIF | CSAR.347.17 | 0.9277 | CANT.328.25 at 0.7579 | 0.170 |
| SIF+ABTT | CSAR.347.17 | 0.7899 | CVAI.442.8 at 0.3422 | 0.448 |

This is the paper's argument in one screen. Under **Raw** the top three candidates span
0.0014 in total (rank 1 to rank 2 is 0.0006), so the ranking is effectively a coin flip and it
lands on Antioch. Every corrected variant instead puts Serdica canon 17 first, and pulls it clear
of the runner-up: by 0.17 under SIF, 0.32 under ABTT and 0.45 under SIF+ABTT. `CSAR` is the
Serdica collection, and the query is a Serdica canon, so the corrected variants are also right.

Worth saying plainly: the size of that gap is variant-specific. SIF alone reorders correctly but
still leaves the field bunched; it takes ABTT, which removes the dominant embedding directions, to
open real distance.

Point at the score column while switching the toggle. The collapse of the Raw scores into a narrow
band near the top is the anisotropy the post-processing removes.

## 3. Query B: short and unambiguous (PhilTa)

Model **PhilTa**, search `C1525.35r.9` (file_id 1076). Only 28 words, so the whole query fits on
screen: `Cap. XLVII. De his, qui in aegritudine baptizantur.`

| Variant | Top-1 | Score |
|---|---|---|
| Raw | CNEO.315.12 | 0.747 (CLAO.300.47 is third, 0.733) |
| ABTT | CLAO.300.47 | 0.590 |
| SIF | CLAO.300.47 | 0.725 |
| SIF+ABTT | CLAO.300.47 | 0.599 |

The query is chapter 47; the answer the corrected variants promote from rank 3 to rank 1 is
Laodicea canon 47. Good query to hand over to Abigail so she can judge it herself, because the
answer does not need any Latin reading to check.

## 4. Query C (backup): the coin flip (PhilTa)

Model **PhilTa**, search `BAV1341.16r.7` (file_id 40), Serdica canon XXI.
Raw ranks CANC.314.3 at 0.87795 and CSAR.347.21 at 0.87769, a gap of 0.0003. SIF+ABTT gives
CSAR.347.21 at 0.544 with the next candidate at 0.225.

A second backup, if a query fails to load: **LaTa**, `C1525.54r.2` (file_id 1132). Raw picks a
Nicaea canon; all three corrected variants pick Serdica canon 2, which is what the query is.

## 5. What to show in the highlight view, per variant

With a candidate expanded, the **Highlights** and **Connections** view modes tint the query and
candidate text by token similarity, and the tint is recomputed from the variant currently selected.
Toggle Raw against SIF on Query A and note that Raw spreads weak highlighting over the function
words shared by every canon (`episcopus`, `dixit`, `qui`, `est`), while SIF pulls the weight onto
the content words that actually distinguish this canon. That is what "down-weights frequent words"
means, made visible.

The third view mode, **Attribution** (integrated gradients and the other six attribution methods,
per variant), is PI-only and needs a precomputed artifact for the exact query/candidate pair. Most
of the 128 artifacts cover the labelled canon pairs used in the paper, so on an arbitrary live
query the toggle is still absent.

The four demo queries above are the exception. Eight artifacts (issue #53) cover each demo query
against both the directory Raw picks and the directory the corrected variants pick, under the model
listed for that query:

| Query | Model | Raw top-1 | Corrected top-1 |
|---|---|---|---|
| `C1525.56v.3` | LaTa | CANT.328.12 | CSAR.347.17 |
| `C1525.35r.9` | PhilTa | CNEO.315.12 | CLAO.300.47 |
| `BAV1341.16r.7` | PhilTa | CANC.314.3 | CSAR.347.21 |
| `C1525.54r.2` | LaTa | CNIC.325.16 | CSAR.347.2 |

Each of those artifacts carries all four variants, so the Attribution toggle follows the same
`Raw | ABTT | SIF | SIF+ABTT` control as the rest of the page. They were built at the layer and D
the deployed retrieval actually uses, and each ABTT panel is cleaned in its own pooling space: the
ABTT panel uses the mean-pooled fit and the SIF+ABTT panel uses the deployed SIF-pooled cleaner,
with the D that variant's own sweep chose (LaTa layer 1: mean D=10, SIF D=3). All four panels
therefore remove the directions the ranking on screen removed. One caveat worth knowing rather
than discovering:

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

## 6. Notes reload and multi-select

On Query B, with SIF+ABTT selected:

1. Tick **Select multiple** above the rank pills.
2. Click rank 1 and rank 2. Both stay lit. This is the answer shape reviewers asked for when two
   candidate directories are both plausible readings of the same fragment.
3. Type a note, for example `Both Laodicea 47 and Neocaesarea 12 are defensible; 47 matches the
   chapter number.`
4. Submit.
5. Navigate to another query and come back. The note and both selected pills are prefilled, above
   an attribution line reading "Last note by &lt;username&gt; - &lt;date&gt;". Notes are shared across the
   review team, so that line credits whoever last recorded a review on the query, which is often
   somebody else. Saving never edits their row; it appends a new one under whoever is signed in.
6. Now switch the toggle to **Raw**. The note does not follow. Feedback is recorded against the
   variant it was given for, so a reviewer's judgement of the SIF+ABTT ranking is never silently
   attributed to the Raw ranking. Switch back to SIF+ABTT and the note returns.

Step 6 is the point worth making slowly: the four variants are four separate things to review, not
four views of one review.

## 7. Expect the rankings to have moved

Say this before Abigail compares against anything she wrote down earlier. The predictions on this
instance were regenerated from the current Task B split, replacing the frozen pre-variant file.
Against that old file, SIF+ABTT top-1 changed for **35.5% of queries under LaTa** and **16.4% under
PhilTa**. The three demo queries above happen to be stable, but any note she took against an
earlier session may point at a rank that has moved.

## 8. If something breaks

| Symptom | Cause | Say |
|---|---|---|
| Variant greyed out | predictions CSV missing on host | "one data file did not sync, the other three variants work" |
| Attribution toggle absent off the demo set | no IG artifact for this pair | expected, see section 5 |
| Attribution toggle absent *on* a demo pair | data release predates issue #53 | re-run the deploy with the newer `DATA_RELEASE_TAG` |
| Query text blank | canon corpus missing on host | fall back to a different query |

Deploy state, workflow runs and the data payload mechanism are documented in
`deploy/REVIEWER_PILOT_READINESS.md`.
