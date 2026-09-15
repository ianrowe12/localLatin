# Reviewer pilot update, 15 September 2026

For Abigail, and for Siddique's information. These are interface changes, not a
data refresh: the fragments, the models, the rankings and every review already
recorded are exactly as they were.

## In two lines

The red "New directory / New file" button is gone, along with its explanation.
The blue "None of the top 10" button now has one optional box beside it, "CCL
key of the source, if known", and one button to record the answer.

## What changed

**The red button is retired.** It created a directory named after whatever was
typed into it, and the directories it made were then scored against every other
fragment and listed as candidates 11 and 12 on other people's shortlists, which
read as though the model had predicted them. That was the source of the
confusion you described. Nothing in the review panel creates a directory any
more.

**The blue button carries the key.** Pressing "None of the top 10" opens one
field. Leave it blank and the app records only that none of the ten candidates
match, exactly as before. Type a CCL key and the app decides what it means,
which is the part that used to be impossible to express:

- the key names a source already in our labelled set: the answer is recorded as
  a match to that key, the receipt says whether the ranking offered it and at
  which rank, and nothing new is created. This is your case of finding the
  source in the CCL when the ten candidates are all wrong;
- the key names a group a reviewer has already started: this fragment joins it,
  so the two witnesses sit together;
- the key is new to the app: a group under that key is started and this fragment
  is its first member.

Whichever happens, the app says so in one sentence after you press Record, and
that sentence stays on screen. The answer and the grouping are saved together or
not at all.

**Your answer is shown back when you return.** Come back to a fragment you have
already answered this way and the app shows what you recorded, with the key and
what it did, and a "Change this answer" button. It does not re-open an empty box
over an answer you already gave. Recording again adds a new answer rather than
replacing the old one, and the app says so; pressing Record twice on an
unchanged answer records it once.

**Groups are named by key, not by siglum.** A group started from the key field
is called by the key. Groups that already exist keep the names they were given;
none of them was renamed. To file a fragment into one of those older groups,
type its name into the key box: each card in the reviewer block says which words
to type.

**Reviewer-made directories left the shortlist.** They are no longer numbered
11 and 12 beside the model's ten. If any are related to the fragment you are
looking at, they appear in a separate block headed "Directories created by
reviewers", with no rank and with a note that they are not part of the model's
answer. You can open one to read its documents. This matches what the printed
review packets have always done.

**The panel has more room.** The ten prediction buttons are larger and more
widely spaced, and the blue action sits under them. Those eleven controls, the
notes box and the Save/Skip buttons are the whole panel now.

## What did not change

- **Your feedback is untouched.** Every note, every decision, every directory
  you or Siddique created is still there, under the same names. The feedback
  database is not written by a deployment, and nothing in this change edits,
  renames or removes a stored row.
- Reviews that recorded a rank of 11 or 12 keep that rank in the record, and the
  export and the PDF packets still print them. They were true when they were
  made; the app simply does not offer those positions any more.
- The red low-confidence notice stays. When the best candidate scores below the
  0.5 line you still get the warning that the ranking below is probably noise,
  and the reminder that a low score is not evidence that the CCL lacks the
  source. It is now a hint with no button in it.
- The corpus, the models, the layers they serve and the rankings themselves are
  unchanged. This release ships no new fragments.

## One thing to watch for

One fragment can start only one group. If you press Record with a key on a
fragment that already starts a group of its own, the app records your answer and
the key and tells you that no second group was started for it. The group that
fragment already starts is shown in the left-hand panel. Tell us if you hit that
and think the two really are different sources; merging or splitting groups is
not something the app can do, so it needs a decision rather than a click.

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
