---
name: blueprint-quadrant-splitter
description: Splits blueprint PDF pages into 6 high-resolution sections for accurate symbol counting. Use when analyzing landscape blueprints, CAD drawings, or any dense technical drawings where text extraction is noisy. Handles seam boundaries to prevent symbol doubling. Converts to 600 DPI for 99% confidence level. Triggers on phrases like split blueprint, quadrant analysis, section blueprint, prepare for counting, high-res blueprint sections.
---

# Blueprint Quadrant Splitter

Splits blueprint PDFs into 6 numbered sections per page for cleaner symbol extraction and counting.

## Section Layout

```
| 1 | 2 | 3 |  (top row, left to right)
| 4 | 5 | 6 |  (bottom row, left to right)
```

## Seam Handling Rules

To prevent symbol doubling when a plant/symbol falls on a boundary:

| Seam Location | Symbol Assigned To | Rule |
|---------------|-------------------|------|
| Between 1↔2 | Section 2 | Move to middle |
| Between 2↔3 | Section 2 | Move to middle |
| Between 4↔5 | Section 5 | Move to middle |
| Between 5↔6 | Section 5 | Move to middle |
| Between 1↔4 | Section 1 | Move up |
| Between 2↔5 | Section 2 | Move up |
| Between 3↔6 | Section 3 | Move up |
| Corner at 1,2,4,5 | Section 2 | Move up + inward |
| Corner at 2,3,5,6 | Section 2 | Move up + inward |

## Usage

```bash
python3 scripts/split_blueprint.py <input.pdf> [output_directory]
```

## Output

Creates files named:
```
<filename>_page1_section1.png
<filename>_page1_section2.png
...
<filename>_page1_section6.png
<filename>_page2_section1.png
...
```

## Resolution

Default: 600 DPI (optimal for 99% symbol recognition confidence)

Override with `--dpi` flag:
```bash
python3 scripts/split_blueprint.py input.pdf output/ --dpi 300
```

## Dependencies

Requires: `pdf2image`, `Pillow`

Install: `pip install pdf2image Pillow --break-system-packages`

Also requires `poppler-utils` system package for PDF rendering.
