# Landscape Symbol Reference Guide

Visual reference for common landscape blueprint symbols and their distinguishing features.

## Tree Symbols

### Deciduous Trees
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Circle, Empty** | Clean circular outline, no interior marks | High: uniform edge, no fill |
| **Circle + X** | Diagonal cross touching edges, centered | High: X arms equal length |
| **Circle + Cross** | Plus sign (+) centered | High: arms perpendicular |
| **Circle + Dot** | Single centered dot | Medium: dot size varies |
| **Circle + Number** | Callout number inside | High: readable number |

### Evergreen/Conifer Trees
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Circle + Hatch** | Diagonal lines 45° | Medium: line spacing varies |
| **Circle + Cross-Hatch** | Both diagonal directions | Medium: density varies |
| **Circle + Stipple** | Regular dot pattern | Medium: dot spacing |
| **Star Shape** | 4-8 pointed star | High: symmetrical points |
| **Jagged Circle** | Irregular spiky edge | Low: hard to distinguish |

### Specimen/Special Trees
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Hexagon** | 6 equal sides | High: clear corners |
| **Double Circle** | Concentric circles | High: two distinct rings |
| **Circle + Triangle** | Triangle pointing up | High: direction clear |
| **Heavy Outline** | Thicker line weight | Medium: subjective thickness |

## Shrub Symbols

### Individual Shrubs
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Small Circle** | Diameter 50-70% of trees | High: relative size |
| **Irregular Blob** | Organic, non-geometric | Low: varies greatly |
| **Rectangle** | Formal/hedge shrub | High: sharp corners |
| **Cloud Shape** | Bumpy outline | Low: subjective shape |

### Shrub Masses
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Overlapping Circles** | 3-5 circles clustered | Medium: count varies |
| **Hatched Area** | Line fill in boundary | Medium: density varies |
| **Stippled Area** | Dot fill in boundary | Medium: density varies |

## Groundcover Symbols

### Area Fills
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Dense Stipple** | Tight dot pattern | Medium: spacing |
| **Light Stipple** | Sparse dot pattern | Medium: spacing |
| **Grass Hatch** | Parallel short lines | High: direction consistent |
| **Random Marks** | Scattered small marks | Low: irregular |

### Linear Elements
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Dashed Line** | Regular gaps | High: gap consistency |
| **Dotted Line** | Regular dots | High: dot spacing |
| **Wavy Line** | Undulating path | Medium: wave consistency |

## Hardscape Symbols

### Pavement
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Cross-Hatch** | Grid pattern | High: regular spacing |
| **Brick Pattern** | Staggered rectangles | Medium: alignment |
| **Random Stone** | Irregular polygons | Low: subjective |

### Structures
| Pattern | Features | Confidence Indicators |
|---------|----------|----------------------|
| **Rectangle Outline** | Clean edges | High: perpendicular |
| **Filled Rectangle** | Solid or hatched | High: distinct fill |
| **X Through Rectangle** | Diagonal cross | High: corners visible |

## Critical Ambiguity Cases

### Frequently Confused Pairs
| Symbol A | Symbol B | Distinguishing Feature |
|----------|----------|----------------------|
| Circle + X | Circle + Cross-Hatch | X has 4 lines, hatch has many |
| Empty Circle | Circle + Faint Fill | Check interior pixel density |
| Hexagon | Circle | Count corners (6 vs none) |
| Small Circle | Large Dot | Size relative to other symbols |
| Stipple Fill | Noise/Artifacts | Stipple is regular pattern |

### Resolution-Dependent Features
| Feature | Min Resolution | Notes |
|---------|---------------|-------|
| Line weight | 300 DPI | Thin vs thick lines |
| Stipple detail | 400 DPI | Individual dots visible |
| Hatch direction | 300 DPI | Diagonal vs perpendicular |
| Interior marks | 400 DPI | X vs cross distinction |
| Edge sharpness | 600 DPI | Sharp vs rounded corners |

## Zoom Analysis Protocol

### When to Zoom
- Composite confidence < 95%
- Any element confidence < 90%
- Lines appear to overlap
- Fill pattern unclear
- Edge continuity questionable

### Zoom Levels
| Level | Magnification | Use Case |
|-------|--------------|----------|
| 2x | 200% | Initial component separation |
| 4x | 400% | Edge analysis |
| 8x | 800% | Line overlap resolution |
| 16x | 1600% | Hairline intersections |

### Evidence Requirements
For any flagged symbol, save:
1. Original cropped symbol
2. 4x zoom of full symbol
3. 8x zoom of ambiguous area
4. Annotations marking uncertain features

## Confidence Scoring Reference

### Element Weights
| Element | Weight | Rationale |
|---------|--------|-----------|
| Base Shape | 35% | Primary identifier |
| Interior Marks | 25% | Secondary identifier |
| Fill Pattern | 15% | Distinguishes variants |
| Edge Treatment | 15% | Line style matters |
| Corner Accents | 10% | Refinement detail |

### Threshold Actions
| Confidence | Action |
|------------|--------|
| ≥95% | Auto-confirm, proceed |
| 85-94% | Flag for review, suggest match |
| 70-84% | Require human decision |
| <70% | Cannot identify, human must classify |
