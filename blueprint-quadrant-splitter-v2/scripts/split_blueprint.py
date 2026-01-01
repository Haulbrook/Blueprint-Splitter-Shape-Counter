#!/usr/bin/env python3
"""
Blueprint Quadrant Splitter
Splits blueprint PDF pages into 6 high-resolution sections for accurate symbol counting.

Section Layout:
| 1 | 2 | 3 |  (top row)
| 4 | 5 | 6 |  (bottom row)

Seam handling ensures no symbol doubling by assigning boundary content to specific sections.
"""

import sys
import os
import argparse
from pathlib import Path

def install_dependencies():
    """Install required packages if not present."""
    import subprocess
    packages = ['pdf2image', 'Pillow']
    for pkg in packages:
        try:
            __import__(pkg.lower().replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', pkg, 
                '--break-system-packages', '-q'
            ])

install_dependencies()

from pdf2image import convert_from_path
from PIL import Image


# Buffer zone in pixels for seam handling (at 600 DPI, ~50 pixels ≈ 0.08 inches)
# This is roughly the size of a small plant symbol
BUFFER_PIXELS = 50


def split_page_into_sections(image: Image.Image, buffer: int = BUFFER_PIXELS) -> dict:
    """
    Split a single page image into 6 sections with seam handling.
    
    Seam Rules:
    - Vertical seams: content goes to middle sections (2 or 5)
    - Horizontal seam: content goes to top sections (1, 2, or 3)
    - Corners: content goes up and toward middle (section 2)
    
    Returns dict mapping section number (1-6) to cropped Image.
    """
    width, height = image.size
    
    # Calculate base section dimensions
    col_width = width // 3
    row_height = height // 2
    
    # Define crop boundaries for each section
    # Format: (left, upper, right, lower)
    # Buffer zones are assigned based on seam rules
    
    sections = {}
    
    # Section 1: Top-left
    # Right edge: exclude buffer (goes to section 2)
    # Bottom edge: include buffer (we own upward)
    sections[1] = (
        0,
        0,
        col_width - buffer,  # Exclude right buffer (belongs to section 2)
        row_height + buffer   # Include bottom buffer (move up rule)
    )
    
    # Section 2: Top-middle
    # Left edge: include buffer (from section 1, move to middle)
    # Right edge: include buffer (from section 3, move to middle)
    # Bottom edge: include buffer (we own upward)
    # This section gets all corner content too
    sections[2] = (
        col_width - buffer,      # Include left buffer
        0,
        2 * col_width + buffer,  # Include right buffer
        row_height + buffer       # Include bottom buffer
    )
    
    # Section 3: Top-right
    # Left edge: exclude buffer (goes to section 2)
    # Bottom edge: include buffer (we own upward)
    sections[3] = (
        2 * col_width + buffer,  # Exclude left buffer (belongs to section 2)
        0,
        width,
        row_height + buffer       # Include bottom buffer (move up rule)
    )
    
    # Section 4: Bottom-left
    # Right edge: exclude buffer (goes to section 5)
    # Top edge: exclude buffer (goes to section 1)
    sections[4] = (
        0,
        row_height + buffer,     # Exclude top buffer (belongs to section 1)
        col_width - buffer,      # Exclude right buffer (belongs to section 5)
        height
    )
    
    # Section 5: Bottom-middle
    # Left edge: include buffer (from section 4, move to middle)
    # Right edge: include buffer (from section 6, move to middle)
    # Top edge: exclude buffer (goes to section 2)
    sections[5] = (
        col_width - buffer,      # Include left buffer
        row_height + buffer,     # Exclude top buffer (belongs to section 2)
        2 * col_width + buffer,  # Include right buffer
        height
    )
    
    # Section 6: Bottom-right
    # Left edge: exclude buffer (goes to section 5)
    # Top edge: exclude buffer (goes to section 3)
    sections[6] = (
        2 * col_width + buffer,  # Exclude left buffer (belongs to section 5)
        row_height + buffer,     # Exclude top buffer (belongs to section 3)
        width,
        height
    )
    
    # Crop each section, ensuring bounds stay within image
    result = {}
    for section_num, bounds in sections.items():
        left, upper, right, lower = bounds
        # Clamp to image boundaries
        left = max(0, left)
        upper = max(0, upper)
        right = min(width, right)
        lower = min(height, lower)
        
        result[section_num] = image.crop((left, upper, right, lower))
    
    return result


def process_pdf(input_path: str, output_dir: str, dpi: int = 600) -> list:
    """
    Process a PDF file and split each page into 6 sections.
    
    Args:
        input_path: Path to input PDF
        output_dir: Directory for output images
        dpi: Resolution for conversion (default 600 for 99% confidence)
    
    Returns:
        List of output file paths
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.stem
    
    print(f"Converting {input_path.name} at {dpi} DPI...")
    
    # Convert PDF to images
    try:
        images = convert_from_path(str(input_path), dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("Make sure poppler-utils is installed: apt-get install poppler-utils")
        sys.exit(1)
    
    output_files = []
    
    for page_num, page_image in enumerate(images, start=1):
        print(f"  Processing page {page_num}...")
        
        # Calculate buffer based on DPI (target ~0.1 inch buffer)
        buffer = int(dpi * 0.1)
        
        sections = split_page_into_sections(page_image, buffer=buffer)
        
        for section_num, section_image in sections.items():
            filename = f"{base_name}_page{page_num}_section{section_num}.png"
            output_path = output_dir / filename
            section_image.save(str(output_path), 'PNG', optimize=True)
            output_files.append(str(output_path))
            print(f"    Saved: {filename} ({section_image.size[0]}x{section_image.size[1]})")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Split blueprint PDFs into 6 sections per page for accurate symbol counting.'
    )
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('output', nargs='?', default=None, 
                        help='Output directory (default: same as input with _sections suffix)')
    parser.add_argument('--dpi', type=int, default=600,
                        help='Resolution in DPI (default: 600 for 99%% confidence)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.pdf':
        print(f"Error: Input must be a PDF file")
        sys.exit(1)
    
    output_dir = args.output
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_sections"
    
    print(f"\nBlueprint Quadrant Splitter")
    print(f"{'='*40}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"DPI:    {args.dpi}")
    print(f"{'='*40}\n")
    
    output_files = process_pdf(str(input_path), str(output_dir), dpi=args.dpi)
    
    print(f"\n{'='*40}")
    print(f"Complete! Created {len(output_files)} section images.")
    print(f"Output directory: {output_dir}")
    print(f"\nSection layout per page:")
    print(f"  | 1 | 2 | 3 |  (top row)")
    print(f"  | 4 | 5 | 6 |  (bottom row)")


if __name__ == '__main__':
    main()
