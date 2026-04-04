# LocalLatin Manuscript Review Tool — Agent Orchestration Plan

## Overview

This document describes the full architecture, implementation plan, and design decisions for the LocalLatin webapp — a tool for Latin scholars to review model predictions on unlabelled medieval Latin manuscript fragments. It was built by a multi-agent team and is intended to be read by any agent continuing work on this project.

---

## What This App Does

1. A **query file** (unlabelled Latin manuscript fragment) is shown on the left panel
2. **Predicted matching documents** from the labelled corpus are shown on the right panel
3. The **hero interaction**: hover over a token in the query, and see animated SVG bezier connection lines to the most similar tokens in the candidate document
4. Scholars can navigate predictions, provide feedback (correct match rank), and export reviews

---

## Backend API (FastAPI, in `web/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/queries?status=&search=&page=` | GET | Paginated query list |
| `/api/query/{id}` | GET | Query file content + tokenized version |
| `/api/query/{id}/predictions?model=` | GET | Top-K predictions with candidate content |
| `/api/token_map/{query_id}/{candidate_id}` | GET | Similarity matrix + top_matches sparse format |
| `/api/feedback` | POST | Save user feedback |
| `/api/stats` | GET | Summary statistics |
| `/api/models` | GET | Available models |

### Backend Files
```
web/
├── app.py              # FastAPI app entry point
├── __main__.py         # python -m web
├── config.py           # PathsConfig, Settings, load from config.yaml
├── config.yaml.example # Example config for other machines
├── models.py           # Pydantic response models (all API types)
├── exceptions.py       # Custom exceptions
├── dependencies.py     # FastAPI dependency injection
├── requirements.txt    # Python deps
├── routers/
│   ├── queries.py      # /api/queries, /api/query/{id}
│   ├── predictions.py  # /api/query/{id}/predictions
│   ├── token_map.py    # /api/token_map/{query_id}/{candidate_id}
│   ├── feedback.py     # /api/feedback
│   └── stats.py        # /api/stats, /api/models
└── services/
    ├── data_store.py     # Loads canon texts, predictions CSV, metadata
    ├── feedback_db.py    # SQLite feedback storage
    ├── text_tokenizer.py # Tokenizes Latin text for display
    └── token_map_svc.py  # Loads NPZ artifacts, computes similarity matrices
```

### Key Data Types (from `web/models.py`)
```python
QueryListItem:    file_id, filename, text_preview, review_status, review_count
QueryDetail:      file_id, filename, text, tokens: List[TokenInfo], char_count, token_count
TokenInfo:        text, index, category (empty/punctuation/number/short_subword/content)
Prediction:       rank, dir_name, score, dir_files, preview_text, candidate_files
TokenMapResponse: query_tokens, candidate_tokens, similarity_matrix (2D float),
                  ig_weighted_matrix (nullable), top_matches: Dict[str, List[TopMatch]],
                  query_ig_baseline, query_ig_abtt, candidate_ig_baseline, candidate_ig_abtt
TokenEntry:       idx, text, is_content
TopMatch:         candidate_idx, score
FeedbackCreate:   query_id, model_slug, correct_rank, correct_dir, notes, reviewer
StatsResponse:    total_queries, reviewed_count, unreviewed_count, feedback_count, ...
ModelInfo:        slug, display_name, layer, pooling, prediction_count
```

---

## Frontend (React, in `web/frontend/`)

### Tech Stack
- React 18 + TypeScript + Vite
- Tailwind CSS 3 (custom theme with `class` dark mode)
- Framer Motion (animations)
- react-virtuoso (virtualized file list for 2,238 items)
- @floating-ui/react (tooltip positioning)
- Google Fonts: "Crimson Pro" (Latin text), "Inter" (UI)

### Directory Structure
```
web/frontend/src/
├── main.tsx              # Boot: installs mock handler if VITE_USE_MOCKS=true
├── App.tsx               # Wraps AppProvider > TokenProvider > FeedbackProvider > AppShell
├── index.css             # Tailwind directives, scrollbar, reduced-motion, glass-panel
│
├── api/                  # Fetch wrapper + custom hooks with Map-based caching
│   ├── client.ts         # apiFetch<T> generic wrapper
│   ├── queries.ts        # useQueryList, useQueryDetail, usePredictions
│   ├── tokenMap.ts       # useTokenMap
│   ├── models.ts         # useModels, useStats
│   └── feedback.ts       # submitFeedback, exportFeedbackCsv
│
├── mock/                 # Development fixtures (VITE_USE_MOCKS=true)
│   ├── handler.ts        # Intercepts fetch(), routes to mock handlers, 200-400ms delay
│   ├── queries.ts        # 3 real Latin queries + generates 2,238 synthetic entries
│   ├── predictions.ts    # 5 predictions per query with ecclesiastical Latin text
│   └── tokenMaps.ts      # Pre-computed similarity matrices (38x24 etc.)
│
├── contexts/
│   ├── AppContext.tsx     # theme, activeQueryId, activeModel, activePredictionRank
│   ├── TokenContext.tsx   # viewMode, hoveredQueryTokenIdx, hoveredMatches, pinnedTokens
│   └── FeedbackContext.tsx # drafts (localStorage), submitFeedback, undo
│
├── components/
│   ├── layout/
│   │   ├── AppShell.tsx        # Top-level: Header + 3-column flex layout
│   │   ├── Header.tsx          # "LocalLatin" title, stats bar, theme toggle, export
│   │   ├── LeftSidebar.tsx     # 280px/48px collapsible, Virtuoso query list
│   │   ├── CenterArea.tsx      # Split panels + TokenRefProvider + ConnectionOverlay
│   │   ├── RightSidebar.tsx    # 320px/48px collapsible, predictions + feedback
│   │   └── DraggableDivider.tsx # PointerEvent + setPointerCapture, 20-80% clamp
│   │
│   ├── sidebar/
│   │   ├── SearchBar.tsx       # 300ms debounced input
│   │   ├── FilterChips.tsx     # All/Pending/Reviewed/Skipped pills
│   │   ├── ProgressRing.tsx    # SVG circle with stroke-dashoffset
│   │   └── QueryCard.tsx       # React.memo card with status dot
│   │
│   ├── document/
│   │   ├── DocumentPanel.tsx   # Header + scrollable token area, wires hover/pin/highlight
│   │   ├── DocumentHeader.tsx  # Filename, dir label (truncated), cosine score badge
│   │   └── TokenSpan.tsx       # React.memo inline span: glow, pin, heatmap, dim states
│   │
│   ├── connections/            # THE HERO FEATURE
│   │   ├── TokenRefRegistry.tsx   # Context: Map<"query:N"|"candidate:N", HTMLSpanElement>
│   │   ├── bezierUtils.ts         # computeBezierPath (cubic S-curve), isRectVisible
│   │   ├── useConnectionState.ts  # Merges pinned + hovered into Connection[] array
│   │   ├── ConnectionOverlay.tsx   # SVG absolute overlay, pointer-events: none, z-10
│   │   ├── ConnectionLine.tsx      # motion.path with pathLength draw animation
│   │   └── useLineUpdater.ts      # rAF scroll handler, passive listeners, ResizeObserver
│   │
│   ├── predictions/
│   │   ├── ModelSelector.tsx    # Native select from useModels
│   │   ├── PredictionCard.tsx   # Rank circle + score bar + dir label
│   │   └── PredictionList.tsx   # Keyboard nav: ArrowUp/Down, 1-9 jump
│   │
│   ├── feedback/
│   │   ├── FeedbackPanel.tsx    # Composes pills + notes + submit
│   │   ├── MatchPills.tsx       # Grid: "Match #1"-"#5" + "None correct"
│   │   ├── NotesTextarea.tsx    # Resizable textarea
│   │   └── SubmitButton.tsx     # Arrow→checkmark animation, ghost Skip
│   │
│   ├── common/
│   │   ├── Tooltip.tsx          # @floating-ui with flip/shift/offset
│   │   ├── Toast.tsx            # Fixed bottom-center undo notification
│   │   ├── SkeletonLoader.tsx   # Pulsing gray bars
│   │   ├── ViewModeToggle.tsx   # Lines/Heatmap/IG segment control
│   │   └── ThemeToggle.tsx      # Sun/moon with rotation animation
│   │
│   └── empty/
│       └── EmptyState.tsx       # "Select a query to begin"
│
└── utils/
    ├── colors.ts     # PIN_COLORS (8), HOVER_COLOR, similarityToColor (colormap)
    ├── keyboard.ts   # useKeyboardShortcuts hook
    └── tokens.ts     # classifyToken, isContentToken
```

### State Architecture

**3 React Contexts** (no Redux — bounded dataset, few consumers):

1. **AppContext**: Theme (persisted to localStorage), active query/model/prediction rank. URL-friendly but no router.
2. **TokenContext**: View mode (connections/heatmap/ig), hovered token index + matches, pinned tokens with color cycling from 8-color palette, IG data availability flag.
3. **FeedbackContext**: Draft feedback per query (persisted to localStorage), submit + undo.

### SVG Connection Lines — Technical Design

This is the most complex feature. The architecture:

1. **TokenRefRegistry** — A `Map<string, HTMLSpanElement>` in a `useRef`, accessed via context. Callback refs registered per token. `getRect()` calls `getBoundingClientRect()` lazily.

2. **useConnectionState** — Computes `activeConnections: Connection[]` by merging pinned tokens (each with assigned color from 8-color palette) and hovered matches (blue). Max ~30 lines (10 pins × 3 matches each).

3. **useLineUpdater** — The performance-critical piece:
   - Maintains `Map<string, { d: string; visible: boolean }>` of computed SVG paths
   - Uses `requestAnimationFrame` for scroll updates (passive listeners on both panels)
   - `ResizeObserver` on container for divider drag
   - Only calls `getBoundingClientRect()` on active connection endpoints (not all tokens)
   - Returns path data consumed by ConnectionOverlay

4. **bezierUtils** — Pure functions:
   - `computeBezierPath(src, tgt)`: Cubic bezier S-curve. Source exits right, target enters left. Control point offset: `max(40, |dx|*0.35) + min(|dy|*0.15, 80)`.
   - `isRectVisible(elementRect, containerRect)`: Partial visibility check.
   - `viewportToSvg()`: Coordinate conversion.

5. **ConnectionLine** — `motion.path` with `pathLength` animation (0→1, 350ms, ease [0.65,0,0.35,1]). Stroke width by rank (2.5/2/1.5). Dashed for rank 2-3. Framer Motion only for enter/exit — position updates are imperative via the updater.

6. **Edge cases**: One endpoint off-screen → opacity 0.3. Both off-screen → hidden. Panel resize triggers ResizeObserver → path recalculation.

### View Modes

| Mode | Behavior |
|------|----------|
| **Connections** (default) | Hover/click query token → bezier lines to top matches |
| **Heatmap** | All candidate tokens colored by max similarity to any query token |
| **IG-weighted** | Same as heatmap but uses IG attribution scores. Hidden when IG data unavailable |

### Color Palette

- **Primary**: Indigo (#6366f1) — actions, active states
- **Highlight**: Amber (#f59e0b) — high similarity, attention
- **Correct**: Emerald (#10b981) — match feedback
- **Incorrect**: Rose (#f43f5e) — "none correct" feedback
- **Pin colors**: Orange, Violet, Cyan, Pink, Lime, Amber, Indigo, Teal (8 cycling)
- **Similarity colormap**: transparent → blue-300 → indigo-500 → indigo-600 → amber-500 (>0.8)

### Typography

- **Latin text**: "Crimson Pro" serif — handles macrons (ā, ē), ligatures (æ), special punctuation
- **UI chrome**: "Inter" sans-serif

---

## Data Files Required by Backend

| Path | Size | Description |
|------|------|-------------|
| `canon_unlabelled/` | 9MB | 2,238 flat `.txt` files — query documents |
| `canon_labelled/` | 11MB | 1,724 `.txt` files in 859 dirs — labelled corpus |
| `runs/phase_resubmit/unlabelled/unlabelled_predictions.csv` | 4.7MB | Combined predictions (all models) |
| `runs/phase_resubmit/unlabelled/unlabelled_predictions_*.csv` | ~2.5MB each | Per-model predictions |
| `runs/phase12f_examples/phase12f_examples.csv` | 17KB | Index of curated IG example pairs |
| `runs/phase12f_examples/artifacts/` | 27MB | 40 NPZ files with token-level IG data |
| `runs/phase_resubmit/webapp/feedback.db` | auto-created | SQLite feedback storage |

All of these are committed to the repo.

---

## Running the App

### Mock mode (frontend only, no data needed):
```bash
cd web/frontend && npm install && npm run dev:mock
```

### Full stack:
```bash
# Terminal 1: Backend
pip install fastapi uvicorn pyyaml transformers torch numpy
# Edit web/config.yaml to set repo_root to your local path
python -m web

# Terminal 2: Frontend
cd web/frontend && npm install && npm run dev
```

---

## What Remains To Do

### Backend
- [ ] Wire up `token_map_svc.py` to load NPZ artifacts and compute similarity matrices on the fly
- [ ] Verify all router endpoints return correct data shapes
- [ ] Add config.yaml with correct paths for the target machine
- [ ] Test feedback SQLite creation and persistence

### Frontend
- [ ] Test full API integration (not just mocks)
- [ ] Token tooltip on hover showing top match token + score
- [ ] Sync scroll toggle between query and candidate panels
- [ ] Export feedback as CSV button wiring
- [ ] Responsive layout testing at 1280px+
- [ ] Verify special characters in directory names render correctly in tooltips
- [ ] Performance profiling with real 2,238-item list

### Polish
- [ ] Loading shimmer effect on tokens while similarity matrix loads
- [ ] Error boundary with retry button
- [ ] Session-based reviewer name (persist across queries)
- [ ] Keyboard shortcut hints in prediction cards
