# How the plain-text corpus is derived from the CCL TEI

Status: verified and scripted (issue #111, epic #107). This answers open decision (c)
from issue #39: we now know how `data/canon_labelled/` and `data/canon_unlabelled/`
relate to the published CCL TEI, and the step is reproducible.

Converter: `scripts/data/tei_to_canon.py`. Unit tests: `tests/test_tei_to_canon.py`.

## Source

Prof. Firey sent a TEI-P5 export of Paris, BnF, lat. 2123 (siglum `BN2123`) in
September 2026. The working copy on the cluster is `/u/irowerojas/BN2123.xml`
(199 KB, 508 unit divs). It is deliberately **not** committed: the repo is public and
CCL licensing terms are still unconfirmed (open decision (b) of issue #39). Anyone
re-running the verification below needs that file, or a fresh export.

Export shape:

```xml
<div type="canon" n="115" xml:id="BN2123.89r.5">
  <div type="msSource"><p> CANONES APOSTOLORUM Hira XLUIII </p></div>
  <div type="scholSource"><p>Can.apost.49</p></div>
  <div type="canonText">
    <p> Si quis iuxta praeceptum <note type="editorial">right margin rubrics: SPE</note>
        domini non baptizauerit ... </p>
  </div>
</div>
```

## The derivation rule

One `.txt` file per unit div that carries an `xml:id`.

1. **File name** is `<xml:id>.txt`, e.g. `BN2123.89r.5.txt`. The part before the first
   dot is the manuscript siglum.
2. **Directory** is the CCL source key in `<div type="scholSource">`, e.g.
   `Can.apost.49`. A unit without a `scholSource` is unlabelled and goes to the flat
   `canon_unlabelled/` pool. This is the only thing the key is used for: it is
   **stripped from the text**, so no file leaks its own label.
3. **Text** is every remaining text-bearing block of the unit, in document order: the
   `<p>`/`<ab>` children of the unit's child divs (`msSource`, `number`, `rubric`,
   `canonText`) plus any `<p>`, `<ab>` or `<note>` that is a direct child of the unit.
   In practice this is the manuscript rubric followed by the canon text.
4. **Inline markup is flattened, not interpreted.** Every nested element contributes
   one space before and one space after its rendered content. `<note type="editorial">`,
   `<sic>`, `<unclear>`, `<add>` and `<seg>` keep their content; `<del>` loses its
   content (a deletion is not part of the reading text) but keeps its boundary spaces.
   Empty milestones such as `<pb/>` therefore render as whitespace, which is why a word
   interrupted by a page break stays split (`peni tentiam`) exactly as the corpus has it.
5. **Layout whitespace collapses**: newline plus indentation inside a block becomes a
   single space. Blocks are joined with two spaces and the unit is prefixed with three
   spaces, which reproduces the leading run the existing files carry (four spaces
   whenever the first `<p>` starts with a space, three otherwise) and the trailing space
   whenever the last `<p>` ends with one.

Nothing else is normalised: no case folding, no punctuation stripping, no Unicode
normalisation. Characters like `·` and `ȩ` survive verbatim.

## Verification against the existing corpus

```bash
python scripts/data/tei_to_canon.py /u/irowerojas/BN2123.xml --dry-run \
    --data-root data --report /tmp/bn2123_diff.csv
```

Result on the 2026-09 export, against the corpus after the issue #112 key corrections:

| Status | Units |
|---|---|
| byte-identical | 158 |
| identical after whitespace normalisation | 121 |
| text differs | 0 |
| directory differs | 0 |
| **reproduction (279 units present in both)** | **279/279 = 100.0% normalised, 158/279 = 56.6% byte-identical** |
| in the export, not in `data/` | 228 |
| in `data/`, not in the export | 9 |

The acceptance bar in issue #111 was 95% after whitespace normalisation; the rule
clears it at 100%, with zero character-level disagreements across all 279 units. The
directory assignment agrees for all 279 as well, including which units carry no key and
therefore sit in the unlabelled pool, so the key-to-directory half of the rule is
confirmed independently of the text half.

The 121 whitespace-only cases are runs of spaces around inline elements: the older
export the corpus was built from indented and broke lines slightly differently, so a
gap that is two spaces in `data/` comes out as three or four here. No token boundary
moves, so text pipelines see identical input after `" ".join(text.split())`, which is
what `run_resubmit_data_prep.py` and the extraction CLIs already do.

## What the 2026-09 export says about the corpus

* **228 new units.** Our BN2123 slice covers 288 units; the export has 507 distinct
  ones. The extra material is the register (`regCanon`, 144 units), topic titles (45),
  decretal excerpts (28) and 11 units of other types (paracanonical, creed, floating
  text, preface). Every `canon` unit in the export is one we already hold, so the new
  material is structurally different from the benchmark, not more of it. Per epic #107 these are
  **not** ingested: benchmark v1 is frozen (see `benchmark_v1.md`) and new CCL material
  flows to the webapp / v2 only.
* **9 units we hold are absent from the export**: `BN2123.72v.2`, `BN2123.72v.3`,
  `BN2123.89v.5`, `BN2123.84v.1`, `BN2123.84v.3`, `BN2123.84v.4`, `BN2123.84v.6`,
  `BN2123.84v.8`, `BN2123.84v.10`. Their ids appear to have been renumbered upstream.
  Worth asking Prof. Firey about, and a reason not to treat an export as authoritative
  for deletions.
* **One duplicate `xml:id`.** Two different canons (`n="206"`, key `CMAC.585.5`, and
  `n="207"`, key `CLYO.518.4`) both carry `xml:id="BN2123.104r.2"`. The converter keeps
  the first and warns; our corpus file is the first one, so the collision only hides the
  second canon. This should be reported upstream.
* **Two key corrections** that Prof. Firey also sent by email are already in the export
  (`BN2123.89r.5` -> `Can.apost.49`, `BN2123.89r.6` -> `Can.apost.50`); issue #112
  applies them to `data/`.

## Reviewing a future export

```bash
# What would change if we ingested this export?
python scripts/data/tei_to_canon.py <export.xml> --dry-run --report /tmp/diff.csv

# Convert to a scratch tree (canon_labelled/<key>/ and canon_unlabelled/)
python scripts/data/tei_to_canon.py <export.xml> --out-dir /tmp/converted

# Only the unit types the paper corpus uses
python scripts/data/tei_to_canon.py <export.xml> --dry-run --types canon
```

`--dry-run` writes nothing and classifies every unit as `identical`, `whitespace`,
`changed`, `moved`, `moved+changed`, `new` or `absent_from_export`. Scope for
`absent_from_export` is limited to the sigla the export covers, so running it on one
manuscript never reports the rest of the corpus as missing.
