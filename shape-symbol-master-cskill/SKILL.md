---
name: shape-symbol-master-cskill
description: High-precision shape and symbol analyzer for blueprint takeoffs. Decomposes any symbol into constituent elements (base shape, interior marks, fill patterns, edge treatments, corner accents). Uses multi-pass verification with zoom-in analysis for overlapping lines. Calculates confidence scores per element. ALWAYS flags symbols below 95% certainty for human confirmation. Triggers on phrases like analyze symbol, identify shape, symbol breakdown, shape components, what symbol is this, describe this shape, blueprint symbol analysis, match symbol to legend, symbol confidence check.
---

# Shape & Symbol Master

Precision analysis of blueprint symbols with component-level decomposition and confidence scoring.

## Core Philosophy

**NEVER GUESS. NEVER ESTIMATE. NEVER ASSUME.**

Every symbol analysis must:
1. Decompose to atomic elements
2. Score each element independently
3. Calculate composite confidence
4. Flag anything below 95% for human review

## Symbol Decomposition Framework

### Element Categories

| Category | What to Check | Confidence Factors |
|----------|---------------|-------------------|
| **Base Shape** | Square, circle, hexagon, triangle, irregular | Edge count, symmetry, closure |
| **Interior Marks** | X, cross, dot, lines, subdivisions | Position, spacing, completeness |
| **Fill Pattern** | Solid, hatched, stippled, empty, gradient | Density, uniformity, boundaries |
| **Edge Treatment** | Single line, double line, dashed, thick/thin | Consistency, gaps, weight |
| **Corner Accents** | Rounded, sharp, notched, extended | Angle measurement, radius |
| **Scale Indicators** | Size relative to other symbols | Pixel dimensions, aspect ratio |

### Analysis Protocol

```
FOR EACH SYMBOL:
├─ PASS 1: Global shape detection (full resolution)
│   └─ Identify bounding box and dominant geometry
├─ PASS 2: Component isolation (2x zoom)
│   └─ Separate base from interior elements
├─ PASS 3: Edge analysis (4x zoom on boundaries)
│   └─ Trace all edges, detect gaps/overlaps
├─ PASS 4: Detail verification (8x zoom on ambiguous areas)
│   └─ Resolve overlapping lines, faint marks
└─ FINAL: Confidence calculation
    └─ If <95%: FLAG FOR HUMAN REVIEW
```

## Confidence Calculation

### Per-Element Scoring

```python
element_confidence = {
    'base_shape': 0.0-1.0,      # How certain of primary geometry
    'interior_marks': 0.0-1.0,   # How certain of internal details
    'fill_pattern': 0.0-1.0,     # How certain of fill type
    'edge_treatment': 0.0-1.0,   # How certain of line style
    'corner_accents': 0.0-1.0    # How certain of corner details
}

# Weighted composite (base shape weighted highest)
composite = (
    base_shape * 0.35 +
    interior_marks * 0.25 +
    fill_pattern * 0.15 +
    edge_treatment * 0.15 +
    corner_accents * 0.10
)
```

### Confidence Thresholds

| Score | Status | Action |
|-------|--------|--------|
| ≥95% | ✅ CONFIRMED | Proceed with match |
| 85-94% | ⚠️ REVIEW | Flag for human confirmation |
| 70-84% | 🔶 UNCERTAIN | Require human decision |
| <70% | ❌ UNIDENTIFIED | Cannot match - human must identify |

## Zoom Protocol for Overlapping Lines

When lines appear to overlap or intersect ambiguously:

```
1. ISOLATE the overlap region (crop to 50px buffer around intersection)
2. ZOOM to 8x minimum (16x for hairline intersections)
3. TRACE each line independently:
   - Start from known endpoint
   - Follow pixel continuity
   - Mark where line terminates vs continues
4. DETERMINE relationship:
   - Lines cross (both continue through)
   - Lines meet (one terminates at other)
   - Lines overlap (same path, different lengths)
   - Lines are separate (gap exists)
5. DOCUMENT finding with cropped evidence image
```

## Common Blueprint Symbol Patterns

### Tree Symbols
| Pattern | Description | Distinguishing Features |
|---------|-------------|------------------------|
| Circle, empty | Deciduous tree | Clean circular edge, no fill |
| Circle, X inside | Specific cultivar marker | X centered, touches edges |
| Circle, dot center | Root ball indicator | Single centered dot |
| Circle, hatched | Evergreen/conifer | Diagonal or cross-hatch fill |
| Hexagon variants | Special specimens | 6 equal sides |

### Shrub Symbols
| Pattern | Description | Distinguishing Features |
|---------|-------------|------------------------|
| Small circle cluster | Shrub mass | 3-5 overlapping circles |
| Irregular blob | Natural shrub form | Organic edges |
| Rectangle | Hedge/formal shrub | Sharp corners |

### Groundcover Symbols
| Pattern | Description | Distinguishing Features |
|---------|-------------|------------------------|
| Stipple pattern | Groundcover area | Dots at regular spacing |
| Small triangles | Ornamental grass | Repeated small marks |
| Wavy lines | Seed/sod area | Parallel undulating lines |

## Flagging Protocol

### ALWAYS FLAG when:
- Any element confidence <95%
- Line weight inconsistent
- Symbol partially obscured
- Scale anomaly detected
- Fill pattern ambiguous
- Edge closure uncertain

### Flag Format
```
⚠️ SYMBOL FLAGGED FOR REVIEW
Location: [coordinates or section reference]
Symbol Description: [best interpretation]
Confidence: [X]%
Uncertainty Reason: [specific element causing doubt]
Options Considered:
  1. [Most likely interpretation] - [confidence]%
  2. [Alternative] - [confidence]%
Action Required: Human confirmation before counting
```

## Integration with Quadrant Splitter

Use with `blueprint-quadrant-splitter` skill:
1. Split blueprint into 6 high-res sections
2. Analyze symbols per section
3. Track section boundaries for seam symbols
4. Consolidate counts avoiding double-counting

## Script Usage

### Analyze Single Symbol
```bash
python3 scripts/analyze_symbol.py <image_path> [--output json|text]
```

### Batch Analysis
```bash
python3 scripts/batch_analyze.py <directory> [--legend legend.json]
```

### Compare to Legend
```bash
python3 scripts/match_to_legend.py <symbol_image> <legend_image>
```

## Output Format

### JSON Analysis Output
```json
{
  "symbol_id": "SYM_001",
  "location": {"x": 1234, "y": 5678, "section": 3},
  "analysis": {
    "base_shape": {
      "type": "circle",
      "confidence": 0.98,
      "dimensions": {"width": 45, "height": 44}
    },
    "interior_marks": {
      "type": "x_cross",
      "confidence": 0.92,
      "note": "slight asymmetry in lower-right arm"
    },
    "fill_pattern": {
      "type": "empty",
      "confidence": 0.99
    },
    "edge_treatment": {
      "type": "single_line",
      "confidence": 0.97,
      "weight": "medium"
    },
    "corner_accents": {
      "type": "not_applicable",
      "confidence": 1.0
    }
  },
  "composite_confidence": 0.962,
  "status": "CONFIRMED",
  "best_match": {
    "legend_code": "APS",
    "name": "Japanese Maple",
    "match_confidence": 0.96
  },
  "flagged": false
}
```

### Flagged Symbol Output
```json
{
  "symbol_id": "SYM_047",
  "location": {"x": 2341, "y": 890, "section": 2},
  "analysis": {
    "base_shape": {
      "type": "circle_or_hexagon",
      "confidence": 0.87,
      "note": "edges appear slightly angular"
    },
    "interior_marks": {
      "type": "unclear",
      "confidence": 0.72,
      "note": "possible X or cross-hatch, lines overlap"
    }
  },
  "composite_confidence": 0.81,
  "status": "REVIEW_REQUIRED",
  "flagged": true,
  "flag_reason": "Interior marks ambiguous - possible X vs cross-hatch",
  "human_action_required": "Confirm symbol type before counting",
  "zoom_evidence": "evidence/SYM_047_8x_zoom.png"
}
```

## Quality Gates

Before reporting ANY symbol identification:

1. ☐ Base shape confidence calculated
2. ☐ All interior elements analyzed
3. ☐ Edge continuity verified
4. ☐ Fill pattern determined
5. ☐ Composite confidence ≥95% OR flagged for review
6. ☐ Zoom analysis completed on ambiguous areas
7. ☐ Evidence images saved for flagged symbols

**NO EXCEPTIONS. If confidence cannot be determined, symbol MUST be flagged.**
