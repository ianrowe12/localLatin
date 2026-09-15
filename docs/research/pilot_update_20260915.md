# Reviewer pilot update, 15 September 2026

For Abigail, and for Siddique's information. These are interface changes, not a
data refresh: the fragments, the models, the rankings and every review already
recorded are exactly as they were.

## Highlighting (issue #211)

The highlighting now marks whole words instead of the pieces a model cuts them
into, so a match on `Episcopus` is shown as `Episcopus` rather than as `Epi` and
`scop` separately. The highlighting also follows the model setting you have
selected, so what is outlined is the evidence behind the ranking you are
looking at.

A "Show pieces" box above the two texts puts the model's own units back on
screen when you want to see them.

Two things worth knowing about what is highlighted, both measured over the whole
deployed set (`docs/research/prefix_attribution_analysis.md`). Frequent Latin
prefixes are **not** over-weighted: they hold 6 to 8 percent of the text and
receive less than their share of the highlight weight under the setting the app
serves. And the system deliberately weights rare wording more heavily than
common wording when it decides a match, so the highlight favours unusual words
over shared formulae. A match you confirm on a commonplace phrase may therefore
carry no highlight there, and that is the method working as designed.
