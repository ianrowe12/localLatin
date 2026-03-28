# Empty Token Report — Phase 12c

## What Are "Empty Tokens"?

When a tokenizer processes Latin text, it sometimes produces tokens that carry **zero lexical content**. These include:
- Whitespace markers (the SentencePiece `▁` prefix that decodes to nothing)
- Literal space characters (` `, `  `, etc.)
- Newline characters (`\n`)
- Non-breaking spaces (`\xa0`)

In our classification, a token is "empty" if, after stripping subword prefixes (`▁`, `##`, `Ġ`), nothing remains.

---

## Empty Tokens by Model

### LaTa (T5 / SentencePiece tokenizer)
- **1 unique empty token type**, 2,352 total occurrences across 320 queries
- The token is a bare SentencePiece `▁` prefix that decodes to an empty string
- This represents ~10% of LaTa's tokens (2,352 out of ~23,550)

### PhilTa (T5 / SentencePiece tokenizer)
- **1 unique empty token type**, 2,939 total occurrences
- Same as LaTa: bare `▁` prefix → empty string
- Represents ~11.4% of PhilTa's tokens

### LaBSE (BERT / WordPiece tokenizer)
- **0 empty tokens** — LaBSE's WordPiece tokenizer never produces whitespace-only subwords
- Every `##` prefix token still has lexical content attached (e.g., `##us`, `##em`)

### Qwen3-0.6B (Decoder / BPE tokenizer)
- **29 unique empty token types**, 3,543 total occurrences
- Dominated by:
  - Newline `\n`: 883 occurrences
  - Double space `  `: 827
  - Single space ` `: 708
  - Various whitespace runs (3–35 spaces): 542 total
  - Non-breaking space `\xa0`: 10
- Represents ~9.2% of Qwen's tokens

### KaLM-mini (Decoder / BPE tokenizer)
- **29 unique empty token types**, 3,543 total occurrences (identical to Qwen — same tokenizer family)
- Same distribution as Qwen3-0.6B

---

## Token Category Distribution (All Models, 320 Queries)

| Category | Count | Percentage |
|----------|------:|:----------:|
| content | 91,818 | 54.1% |
| short_subword | 51,468 | 30.3% |
| punctuation | 13,662 | 8.0% |
| empty | 12,377 | 7.3% |
| number | 521 | 0.3% |

---

## Why Simple Filtering Can't Replace ABTT

The professor asked: "Can't you just remove these empty tokens with a regex?"

**The key insight**: empty tokens dominate baseline retrieval attribution not because they are *numerous* (only 7–11% of tokens), but because the PC1 noise direction gets concentrated in their embeddings. At dip layers, an empty token's hidden state is nearly a pure copy of PC1. Content tokens also contain PC1 — they just have additional semantic signal on top.

Filtering out the ~10% of empty tokens would:
- Remove ~10% of the token count from mean pooling
- Leave the remaining content tokens **still contaminated with PC1 noise**
- Produce marginal improvement at best

ABTT projects PC1 out of **every** token's embedding (including content), which is why it produces dramatically better results than filtering could.

**Phase 12d will test this directly** with a controlled experiment comparing filtering vs. ABTT.
