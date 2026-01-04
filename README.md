# Blueprint-Splitter-Shape-Counter
<<<<<<< Updated upstream

AI-powered precision counting system for commercial landscape blueprints. Splits blueprint PDFs into high-resolution sections and counts symbols with absolute accuracy—exact integers only, no estimates.

## Why This Exists

**Wrong counts = wrong material orders = budget errors = project failures = liability.**

Traditional blueprint takeoffs are error-prone. This system ensures:
- Every symbol is counted exactly once (seam handling prevents double-counting)
- Uncertain areas are flagged for human review (never guessed)
- Visual patterns are identified (not plant type assumptions)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         WORKFLOW                                 │
│                                                                  │
│   [PDF Blueprint]                                                │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────────────┐                               │
│   │ Blueprint Quadrant Splitter │  Splits into 6 sections       │
│   │      (600 DPI PNG)          │  at 600 DPI resolution        │
│   └─────────────────────────────┘                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────────────┐                               │
│   │   Shape Symbol Master       │  Counts symbols per section   │
│   │   (PRODUCTION Agent)        │  with EXACT integers only     │
│   └─────────────────────────────┘                               │
│         │                                                        │
│         ▼                                                        │
│   [Aggregated Counts + Flagged Areas]                           │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Blueprint Quadrant Splitter

**Location:** `blueprint-quadrant-splitter-v2/`

Splits blueprint PDFs into 6 sections per page at 600 DPI for optimal symbol recognition.

**Section Layout:**
```
| 1 | 2 | 3 |  (top row, left to right)
| 4 | 5 | 6 |  (bottom row, left to right)
```

**Seam Handling Rules:**
| Seam Location | Assigned To | Rule |
|---------------|-------------|------|
| Vertical (between columns) | Middle column (2 or 5) | Prevents horizontal double-count |
| Horizontal (between rows) | Upper section | Prevents vertical double-count |
| Corners | Section 2 | Move up + inward |

**Usage:**
```bash
python3 scripts/split_blueprint.py <input.pdf> [output_directory] [--dpi 600]
```

**Output:**
```
<filename>_page1_section1.png
<filename>_page1_section2.png
...
<filename>_page1_section6.png
```

**Dependencies:**
```bash
pip install pdf2image Pillow
# Also requires poppler-utils system package
```

---

### 2. Shape Symbol Master (PRODUCTION)

**Location:** `shape_symbol_master_PRODUCTION/`

Precision counting agent that receives 600 DPI sections and returns exact counts.

**The Absolute Counting Standard:**

| Output Type | When to Use | Example |
|-------------|-------------|---------|
| Exact integer | You can count with certainty | `7` |
| CANNOT COUNT | Truly unresolvable situation | See examples below |

**CANNOT COUNT examples** (use only when training cannot resolve):
- `CANNOT COUNT - symbol 60% obscured by title block text`
- `CANNOT COUNT - symbol cut off at scan edge, only 30% visible`
- `CANNOT COUNT - legend reference missing, cannot verify symbol type`

**FORBIDDEN outputs:** `4-6`, `~5`, `about`, `approximately`, `maybe`, `probably`

**Symbol Decomposition (3-Layer System):**

Symbols are identified by visual pattern, not plant type assumptions:

```
Layer 1 - OUTLINE:    circular, star, cloud, polygon, linear
Layer 2 - INTERIOR:   radial lines, stipple, hatching, empty, segmented
Layer 3 - MARKERS:    center dot, cross, multiple dots, none
```

**Output format:** Composite ID like `circular_scalloped_radial_centerdot`

**Counting Protocol:**
1. Grid the section (3x3: A1-C3)
2. Count one symbol type at a time
3. Enumerate with locations: `1(A1), 2(A1), 3(B2)...`
4. Verify with second counting method
5. Declare status: CERTAIN or CANNOT COUNT

**Files:**
```
shape_symbol_master_PRODUCTION/
├── shape_symbol_master_PRODUCTION.json   # Main agent definition
├── ABSOLUTE_COUNTING_STANDARD.json       # Hard counting rules
├── precision_counting_system.json        # Counting protocol
├── symbol_decomposition_system.json      # 3-layer decomposition
└── INTEGRATION.md                        # Integration guide
```

---

### 3. Shape Symbol Master C-Skill

**Location:** `shape-symbol-master-cskill/`

Component-level symbol analyzer with multi-pass zoom verification and confidence scoring.

**Analysis Protocol:**
```
PASS 1: Global shape detection (full resolution)
PASS 2: Component isolation (2x zoom)
PASS 3: Edge analysis (4x zoom)
PASS 4: Detail verification (8x zoom on ambiguous areas)
FINAL:  Confidence calculation
```

**Confidence Thresholds:**

| Score | Status | Action |
|-------|--------|--------|
| ≥98% | CONFIRMED | Proceed with count |
| 85-97% | REVIEW | Flag for human confirmation |
| 70-84% | UNCERTAIN | Require human decision |
| <70% | UNIDENTIFIED | Human must identify |

**Element Scoring:**
```
Composite = (base_shape × 0.35) + (interior_marks × 0.25) +
            (fill_pattern × 0.15) + (edge_treatment × 0.15) +
            (corner_accents × 0.10)
```

---

## Training Lineage

The Shape Symbol Master has been trained through multiple iterations:

```
v4:  5-phase reasoning framework
v5:  55 real extracted symbols
v6:  200 labeled symbol classification
v7:  Symbol decomposition (no plant assumptions)
v8:  Vision-model optimization
PRODUCTION: Absolute counting standard integration
```

---

## Quick Start

### Split a Blueprint
```bash
cd blueprint-quadrant-splitter-v2
python3 scripts/split_blueprint.py /path/to/blueprint.pdf ./output/ --dpi 600
```

### Analyze Symbols
```bash
cd shape-symbol-master-cskill
python3 scripts/analyze_symbol.py /path/to/section.png --output json
```

### Batch Analysis with Legend
```bash
python3 scripts/batch_analyze.py ./sections/ --legend legend.json
```

---

## File Structure

```
Blueprint-Splitter-Shape-Counter/
│
├── README.md                              # This file
├── blueprint-quadrant-splitter.skill      # Packaged skill archive
│
├── blueprint-quadrant-splitter-v2/
│   ├── SKILL.md                           # Splitter documentation
│   └── scripts/
│       └── split_blueprint.py             # PDF splitting script
│
├── shape_symbol_master_PRODUCTION/
│   ├── shape_symbol_master_PRODUCTION.json
│   ├── ABSOLUTE_COUNTING_STANDARD.json
│   ├── precision_counting_system.json
│   ├── symbol_decomposition_system.json
│   └── INTEGRATION.md
│
└── shape-symbol-master-cskill/
    ├── SKILL.md                           # Confidence scoring docs
    ├── scripts/                           # Analysis scripts
    └── references/                        # Reference materials
```

---

## Core Principles

1. **EXACT COUNTS ONLY** — No ranges, no estimates, no approximations
2. **DECOMPOSE SYMBOLS** — Identify visual patterns, not plant types
3. **ENUMERATE WITH LOCATIONS** — Every symbol gets a number and grid position
4. **VERIFY EVERY COUNT** — Two counting methods must match
5. **FLAG UNCERTAINTIES** — Never guess; flag for human review
6. **FOLLOW SEAM RULES** — Prevent double-counting across sections
7. **EVIDENCE FOR EVERYTHING** — State what you see, not what you assume

---

## License

Private repository.

---

## Author

Haulbrook
