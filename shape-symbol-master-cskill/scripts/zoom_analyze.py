#!/usr/bin/env python3
"""
Shape & Symbol Master - Zoom Analysis Utility
Provides high-magnification analysis for resolving ambiguous symbol features.
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install Pillow numpy --break-system-packages")
    sys.exit(1)


@dataclass
class ZoomResult:
    zoom_level: int
    region: Tuple[int, int, int, int]  # x, y, width, height
    finding: str
    confidence: float
    evidence_path: str


class ZoomAnalyzer:
    """
    High-magnification analysis for ambiguous symbol features.
    """
    
    ZOOM_LEVELS = [2, 4, 8, 16]  # Available magnification levels
    
    def __init__(self, image_path: str, output_dir: str = "./zoom_evidence"):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image = Image.open(self.image_path).convert('RGBA')
        self.gray = self.image.convert('L')
        self.np_gray = np.array(self.gray)
        
        self.width, self.height = self.image.size
        self.symbol_id = self.image_path.stem
    
    def analyze_region(self, x: int, y: int, width: int, height: int,
                      zoom_level: int = 8) -> ZoomResult:
        """
        Perform zoom analysis on a specific region.
        """
        # Validate and adjust bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        width = min(width, self.width - x)
        height = min(height, self.height - y)
        
        # Extract and zoom region
        region = self.image.crop((x, y, x + width, y + height))
        zoomed = region.resize(
            (width * zoom_level, height * zoom_level),
            Image.Resampling.NEAREST  # Preserve pixels for analysis
        )
        
        # Analyze the zoomed region
        region_gray = np.array(region.convert('L'))
        finding, confidence = self._analyze_zoomed_content(region_gray)
        
        # Save evidence
        evidence_path = self._save_evidence(zoomed, x, y, width, height, zoom_level)
        
        return ZoomResult(
            zoom_level=zoom_level,
            region=(x, y, width, height),
            finding=finding,
            confidence=confidence,
            evidence_path=evidence_path
        )
    
    def analyze_center(self, zoom_level: int = 8) -> ZoomResult:
        """
        Zoom analysis on the center region.
        """
        # Analyze center 50% of symbol
        margin_x = self.width // 4
        margin_y = self.height // 4
        
        return self.analyze_region(
            margin_x, margin_y,
            self.width // 2, self.height // 2,
            zoom_level
        )
    
    def analyze_edges(self, zoom_level: int = 4) -> List[ZoomResult]:
        """
        Zoom analysis on all four edges.
        """
        results = []
        edge_width = max(5, self.width // 8)
        edge_height = max(5, self.height // 8)
        
        # Top edge
        results.append(self.analyze_region(
            self.width // 4, 0,
            self.width // 2, edge_height,
            zoom_level
        ))
        
        # Bottom edge
        results.append(self.analyze_region(
            self.width // 4, self.height - edge_height,
            self.width // 2, edge_height,
            zoom_level
        ))
        
        # Left edge
        results.append(self.analyze_region(
            0, self.height // 4,
            edge_width, self.height // 2,
            zoom_level
        ))
        
        # Right edge
        results.append(self.analyze_region(
            self.width - edge_width, self.height // 4,
            edge_width, self.height // 2,
            zoom_level
        ))
        
        return results
    
    def trace_line_intersection(self, x: int, y: int, 
                                radius: int = 10,
                                zoom_level: int = 16) -> Dict:
        """
        Detailed analysis of a line intersection point.
        """
        # Extract region around intersection
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.width, x + radius)
        y2 = min(self.height, y + radius)
        
        region = self.np_gray[y1:y2, x1:x2]
        
        if region.size == 0:
            return {"error": "Invalid region"}
        
        # Trace lines through the point
        lines = self._trace_lines_at_point(region, radius, radius)
        
        # Determine relationship
        if len(lines) == 0:
            relationship = "no_lines_detected"
        elif len(lines) == 1:
            relationship = "single_line"
        elif len(lines) == 2:
            relationship = self._determine_line_relationship(lines)
        else:
            relationship = "multiple_lines_complex"
        
        # Zoom for evidence
        zoom_region = self.image.crop((x1, y1, x2, y2))
        zoomed = zoom_region.resize(
            ((x2 - x1) * zoom_level, (y2 - y1) * zoom_level),
            Image.Resampling.NEAREST
        )
        
        evidence_path = self.output_dir / f"{self.symbol_id}_intersection_{x}_{y}_zoom{zoom_level}.png"
        
        # Add grid lines for analysis
        draw = ImageDraw.Draw(zoomed)
        center_x = zoomed.width // 2
        center_y = zoomed.height // 2
        draw.line([(center_x, 0), (center_x, zoomed.height)], fill=(255, 0, 0, 128), width=1)
        draw.line([(0, center_y), (zoomed.width, center_y)], fill=(255, 0, 0, 128), width=1)
        
        zoomed.save(evidence_path)
        
        return {
            "location": {"x": x, "y": y},
            "zoom_level": zoom_level,
            "lines_detected": len(lines),
            "relationship": relationship,
            "evidence_path": str(evidence_path),
            "analysis_details": {
                "line_angles": [l.get("angle") for l in lines] if lines else [],
                "interpretation": self._interpret_relationship(relationship, lines)
            }
        }
    
    def _analyze_zoomed_content(self, region: np.ndarray) -> Tuple[str, float]:
        """
        Analyze content in a zoomed region.
        """
        if region.size == 0:
            return "empty_region", 0.0
        
        # Calculate statistics
        dark_ratio = np.sum(region < 128) / region.size
        
        # Detect patterns
        if dark_ratio < 0.1:
            return "empty", 0.95
        elif dark_ratio > 0.8:
            return "solid_fill", 0.95
        
        # Check for lines
        h_lines = self._detect_horizontal_lines(region)
        v_lines = self._detect_vertical_lines(region)
        d_lines = self._detect_diagonal_lines(region)
        
        if h_lines and v_lines:
            if d_lines:
                return "cross_hatch", 0.85
            else:
                return "grid_pattern", 0.88
        elif d_lines:
            return "diagonal_lines", 0.85
        elif h_lines or v_lines:
            return "parallel_lines", 0.85
        
        # Check for dots
        if self._detect_dots(region):
            return "stipple_pattern", 0.82
        
        return "unclear_pattern", 0.60
    
    def _detect_horizontal_lines(self, region: np.ndarray) -> bool:
        """
        Detect horizontal line patterns.
        """
        h, w = region.shape
        if h < 5 or w < 5:
            return False
        
        # Check row-by-row dark pixel counts
        row_counts = np.sum(region < 128, axis=1)
        
        # Lines show as peaks in row counts
        high_rows = row_counts > w * 0.3
        return np.sum(high_rows) > 2
    
    def _detect_vertical_lines(self, region: np.ndarray) -> bool:
        """
        Detect vertical line patterns.
        """
        h, w = region.shape
        if h < 5 or w < 5:
            return False
        
        col_counts = np.sum(region < 128, axis=0)
        high_cols = col_counts > h * 0.3
        return np.sum(high_cols) > 2
    
    def _detect_diagonal_lines(self, region: np.ndarray) -> bool:
        """
        Detect diagonal line patterns.
        """
        h, w = region.shape
        if h < 5 or w < 5:
            return False
        
        # Sample along diagonals
        samples = min(h, w)
        
        diag1_dark = 0
        diag2_dark = 0
        
        for i in range(samples):
            y = int(i * h / samples)
            x1 = int(i * w / samples)
            x2 = int((samples - 1 - i) * w / samples)
            
            if 0 <= y < h and 0 <= x1 < w:
                if region[y, x1] < 128:
                    diag1_dark += 1
            if 0 <= y < h and 0 <= x2 < w:
                if region[y, x2] < 128:
                    diag2_dark += 1
        
        return diag1_dark > samples * 0.4 or diag2_dark > samples * 0.4
    
    def _detect_dots(self, region: np.ndarray) -> bool:
        """
        Detect stipple/dot patterns.
        """
        try:
            from scipy import ndimage
        except ImportError:
            return False
        
        dark_mask = region < 128
        labeled, num_features = ndimage.label(dark_mask)
        
        if num_features < 3:
            return False
        
        # Check if features are similar size (dots)
        sizes = ndimage.sum(dark_mask, labeled, range(1, num_features + 1))
        
        if len(sizes) == 0:
            return False
        
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Dots are similar sized
        return std_size / (mean_size + 1) < 0.6
    
    def _trace_lines_at_point(self, region: np.ndarray, 
                              cx: int, cy: int) -> List[Dict]:
        """
        Trace lines passing through a point.
        """
        lines = []
        
        h, w = region.shape
        if h < 3 or w < 3:
            return lines
        
        # Check 8 directions
        directions = [
            (0, 1, "horizontal"),
            (1, 0, "vertical"),
            (1, 1, "diagonal_down"),
            (1, -1, "diagonal_up"),
            (-1, 1, "diagonal_up"),
            (-1, -1, "diagonal_down"),
            (0, -1, "horizontal"),
            (-1, 0, "vertical")
        ]
        
        for dy, dx, name in directions[:4]:  # Check 4 main directions
            # Trace in both directions
            forward = self._trace_direction(region, cy, cx, dy, dx)
            backward = self._trace_direction(region, cy, cx, -dy, -dx)
            
            if forward > 2 and backward > 2:
                angle = np.degrees(np.arctan2(dy, dx))
                lines.append({
                    "direction": name,
                    "angle": angle,
                    "length": forward + backward,
                    "continuous": True
                })
        
        return lines
    
    def _trace_direction(self, region: np.ndarray, 
                        y: int, x: int, dy: int, dx: int) -> int:
        """
        Trace dark pixels in a direction.
        """
        h, w = region.shape
        length = 0
        
        for i in range(1, min(h, w)):
            ny = y + dy * i
            nx = x + dx * i
            
            if 0 <= ny < h and 0 <= nx < w:
                if region[ny, nx] < 128:
                    length += 1
                else:
                    break
            else:
                break
        
        return length
    
    def _determine_line_relationship(self, lines: List[Dict]) -> str:
        """
        Determine how two lines relate at intersection.
        """
        if len(lines) != 2:
            return "unknown"
        
        angle_diff = abs(lines[0]["angle"] - lines[1]["angle"])
        
        if angle_diff > 170 or angle_diff < 10:
            return "lines_overlap"  # Same direction
        elif 80 < angle_diff < 100:
            return "lines_cross_perpendicular"
        else:
            return "lines_cross_acute"
    
    def _interpret_relationship(self, relationship: str, 
                               lines: List[Dict]) -> str:
        """
        Human-readable interpretation of line relationship.
        """
        interpretations = {
            "no_lines_detected": "No clear lines found at this point",
            "single_line": "Single line passes through this point",
            "lines_cross_perpendicular": "Two lines cross at approximately 90°",
            "lines_cross_acute": "Two lines cross at an acute angle",
            "lines_overlap": "Lines appear to run along the same path",
            "multiple_lines_complex": "Complex intersection with 3+ lines"
        }
        
        return interpretations.get(relationship, "Unknown relationship")
    
    def _save_evidence(self, zoomed: Image.Image, x: int, y: int, 
                      width: int, height: int, zoom_level: int) -> str:
        """
        Save zoomed evidence image with annotations.
        """
        # Add border and annotation
        bordered = Image.new('RGBA', 
                            (zoomed.width + 4, zoomed.height + 20),
                            (255, 255, 255, 255))
        bordered.paste(zoomed, (2, 2))
        
        draw = ImageDraw.Draw(bordered)
        
        # Add annotation text
        annotation = f"{zoom_level}x zoom | Region: ({x},{y}) {width}x{height}px"
        draw.text((2, zoomed.height + 4), annotation, fill=(0, 0, 0, 255))
        
        evidence_path = self.output_dir / f"{self.symbol_id}_region_{x}_{y}_zoom{zoom_level}.png"
        bordered.save(evidence_path)
        
        return str(evidence_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python zoom_analyze.py <image_path> [options]")
        print("\nOptions:")
        print("  --region X Y W H   Analyze specific region")
        print("  --center           Analyze center region")
        print("  --edges            Analyze all edges")
        print("  --intersection X Y Analyze line intersection at point")
        print("  --zoom LEVEL       Zoom level (2, 4, 8, 16)")
        print("  --output DIR       Output directory")
        print("\nPerforms high-magnification analysis for ambiguous features.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = "./zoom_evidence"
    zoom_level = 8
    mode = "center"
    region = None
    point = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--region" and i + 4 < len(sys.argv):
            mode = "region"
            region = (
                int(sys.argv[i + 1]),
                int(sys.argv[i + 2]),
                int(sys.argv[i + 3]),
                int(sys.argv[i + 4])
            )
            i += 5
        elif sys.argv[i] == "--center":
            mode = "center"
            i += 1
        elif sys.argv[i] == "--edges":
            mode = "edges"
            i += 1
        elif sys.argv[i] == "--intersection" and i + 2 < len(sys.argv):
            mode = "intersection"
            point = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
            i += 3
        elif sys.argv[i] == "--zoom" and i + 1 < len(sys.argv):
            zoom_level = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    try:
        analyzer = ZoomAnalyzer(image_path, output_dir)
        
        if mode == "region" and region:
            result = analyzer.analyze_region(
                region[0], region[1], region[2], region[3], zoom_level
            )
            print(json.dumps({
                "zoom_level": result.zoom_level,
                "region": result.region,
                "finding": result.finding,
                "confidence": result.confidence,
                "evidence_path": result.evidence_path
            }, indent=2))
        
        elif mode == "center":
            result = analyzer.analyze_center(zoom_level)
            print(json.dumps({
                "zoom_level": result.zoom_level,
                "region": result.region,
                "finding": result.finding,
                "confidence": result.confidence,
                "evidence_path": result.evidence_path
            }, indent=2))
        
        elif mode == "edges":
            results = analyzer.analyze_edges(zoom_level)
            edges = ["top", "bottom", "left", "right"]
            output = {"edges": []}
            for edge, result in zip(edges, results):
                output["edges"].append({
                    "edge": edge,
                    "finding": result.finding,
                    "confidence": result.confidence,
                    "evidence_path": result.evidence_path
                })
            print(json.dumps(output, indent=2))
        
        elif mode == "intersection" and point:
            result = analyzer.trace_line_intersection(
                point[0], point[1], zoom_level=zoom_level
            )
            print(json.dumps(result, indent=2))
        
        print(f"\nEvidence saved to: {output_dir}/")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
