# Shape Symbol Master - PRODUCTION Integration

## Export Contents

```
shape_symbol_master_PRODUCTION/
├── shape_symbol_master_PRODUCTION.json   # Main agent definition
├── ABSOLUTE_COUNTING_STANDARD.json       # Counting rules (hard requirements)
├── precision_counting_system.json        # Counting protocol
├── symbol_decomposition_system.json      # Three-layer decomposition framework
└── INTEGRATION.md                        # This file
```

## Integration Steps

### 1. Copy to Blueprint-Splitter-Shape-Counter

Place `shape_symbol_master_PRODUCTION.json` in your skills/agents directory.

### 2. Load the System Prompt

The `system_prompt` field contains the complete trained behavior. Use it as-is when invoking the agent.

### 3. Workflow Connection

**Input** (from Blueprint Quadrant Splitter):
- 600 DPI PNG section (1 of 6)
- Section number
- Seam positions
- Optional legend reference

**Output** (to counting aggregator):
- Symbol type (composite ID)
- Exact count OR "CANNOT COUNT - [reason]"
- Enumerated locations
- Verification confirmation
- Edge symbol handling notes
- Flagged areas for human review

## Critical Standards

### Absolute Counting Rule

Only TWO valid outputs:
1. **Exact integer**: `7` (when certain)
2. **CANNOT COUNT**: `CANNOT COUNT - symbols overlap at B2, centers not visible` (when uncertain)

**FORBIDDEN**: ranges, approximations, hedging (`4-6`, `~5`, `about`, `maybe`)

### Symbol Decomposition

Symbols are **visual patterns**, not plant types:
- Layer 1: Outline (circular, star, cloud, polygon)
- Layer 2: Interior (radial, stipple, hatching, empty)
- Layer 3: Marker (center dot, cross, none)

Output: Composite ID like `circular_scalloped_radial_centerdot`

### Seam Rules (Edge Handling)

- Symbol >50% in section → COUNT IT
- Symbol <50% in section → DO NOT COUNT (other section has it)
- Horizontal seams → upper section counts
- Vertical seams → middle column (2 or 5) counts

## Training Lineage

```
v4: 5-phase reasoning framework
v5: 55 real extracted symbols
v6: 200 labeled symbol classification
v7: Symbol decomposition (no plant assumptions)
v8: Vision-model optimization
PRODUCTION: Absolute counting standard integration
```

## Verification Checklist

Before deploying, verify the agent:
- [ ] Outputs exact integers (not ranges)
- [ ] Uses "CANNOT COUNT" when uncertain
- [ ] Decomposes symbols (doesn't assume plant types)
- [ ] Enumerates with grid locations
- [ ] Verifies counts with two methods
- [ ] Applies seam rules at edges
