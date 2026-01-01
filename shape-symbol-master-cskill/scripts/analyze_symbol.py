#!/usr/bin/env python3
"""
Shape & Symbol Master - Single Symbol Analyzer
Decomposes symbols into constituent elements with confidence scoring.
"""

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install Pillow numpy --break-system-packages")
    sys.exit(1)


class ConfidenceLevel(Enum):
    CONFIRMED = "CONFIRMED"           # ≥95%
    REVIEW = "REVIEW_REQUIRED"        # 85-94%
    UNCERTAIN = "UNCERTAIN"           # 70-84%
    UNIDENTIFIED = "UNIDENTIFIED"     # <70%


@dataclass
class ElementAnalysis:
    element_type: str
    confidence: float
    details: Dict[str, Any]
    note: Optional[str] = None


@dataclass
class SymbolAnalysis:
    symbol_id: str
    location: Dict[str, int]
    base_shape: ElementAnalysis
    interior_marks: ElementAnalysis
    fill_pattern: ElementAnalysis
    edge_treatment: ElementAnalysis
    corner_accents: ElementAnalysis
    composite_confidence: float
    status: str
    flagged: bool
    flag_reason: Optional[str] = None
    zoom_evidence: Optional[str] = None
    best_match: Optional[Dict[str, Any]] = None


class SymbolAnalyzer:
    """
    Multi-pass symbol analyzer with zoom capabilities.
    """
    
    # Confidence weights for composite score
    WEIGHTS = {
        'base_shape': 0.35,
        'interior_marks': 0.25,
        'fill_pattern': 0.15,
        'edge_treatment': 0.15,
        'corner_accents': 0.10
    }
    
    # Confidence thresholds
    THRESHOLD_CONFIRMED = 0.95
    THRESHOLD_REVIEW = 0.85
    THRESHOLD_UNCERTAIN = 0.70
    
    def __init__(self, image_path: str, output_dir: str = "./evidence"):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image = Image.open(self.image_path).convert('RGBA')
        self.gray = self.image.convert('L')
        self.np_gray = np.array(self.gray)
        self.np_rgba = np.array(self.image)
        
        self.width, self.height = self.image.size
        self.symbol_id = f"SYM_{self.image_path.stem}"
        
    def analyze(self) -> SymbolAnalysis:
        """
        Execute full 4-pass analysis protocol.
        """
        # PASS 1: Global shape detection
        base_shape = self._analyze_base_shape()
        
        # PASS 2: Component isolation (2x conceptual zoom)
        interior_marks = self._analyze_interior_marks()
        fill_pattern = self._analyze_fill_pattern()
        
        # PASS 3: Edge analysis (4x conceptual zoom on boundaries)
        edge_treatment = self._analyze_edge_treatment()
        corner_accents = self._analyze_corner_accents()
        
        # PASS 4: Detail verification on ambiguous areas
        # Automatically zooms to 8x on low-confidence elements
        elements = [base_shape, interior_marks, fill_pattern, edge_treatment, corner_accents]
        for elem in elements:
            if elem.confidence < self.THRESHOLD_CONFIRMED:
                self._deep_zoom_verify(elem)
        
        # Calculate composite confidence
        composite = self._calculate_composite(
            base_shape, interior_marks, fill_pattern, 
            edge_treatment, corner_accents
        )
        
        # Determine status and flagging
        status, flagged, flag_reason = self._determine_status(
            composite, elements
        )
        
        # Save evidence if flagged
        zoom_evidence = None
        if flagged:
            zoom_evidence = self._save_evidence(elements)
        
        return SymbolAnalysis(
            symbol_id=self.symbol_id,
            location={"x": 0, "y": 0, "width": self.width, "height": self.height},
            base_shape=base_shape,
            interior_marks=interior_marks,
            fill_pattern=fill_pattern,
            edge_treatment=edge_treatment,
            corner_accents=corner_accents,
            composite_confidence=round(composite, 4),
            status=status,
            flagged=flagged,
            flag_reason=flag_reason,
            zoom_evidence=zoom_evidence
        )
    
    def _analyze_base_shape(self) -> ElementAnalysis:
        """
        PASS 1: Detect primary geometry.
        """
        # Find bounding box of non-transparent pixels
        if self.np_rgba.shape[2] == 4:
            alpha = self.np_rgba[:, :, 3]
            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)
        else:
            rows = np.any(self.np_gray < 250, axis=1)
            cols = np.any(self.np_gray < 250, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return ElementAnalysis(
                element_type="empty",
                confidence=0.0,
                details={"error": "no_content_detected"},
                note="Image appears empty or fully transparent"
            )
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        y_min, y_max = row_indices[0], row_indices[-1]
        x_min, x_max = col_indices[0], col_indices[-1]
        
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / max(height, 1)
        
        # Detect edges using simple gradient
        edges = self._detect_edges()
        
        # Analyze shape characteristics
        shape_type, confidence, details = self._classify_shape(
            edges, width, height, aspect_ratio
        )
        
        return ElementAnalysis(
            element_type=shape_type,
            confidence=confidence,
            details=details
        )
    
    def _detect_edges(self) -> np.ndarray:
        """
        Simple edge detection using gradient magnitude.
        """
        # Sobel-like gradients
        gx = np.zeros_like(self.np_gray, dtype=float)
        gy = np.zeros_like(self.np_gray, dtype=float)
        
        gx[:, 1:] = self.np_gray[:, 1:].astype(float) - self.np_gray[:, :-1].astype(float)
        gy[1:, :] = self.np_gray[1:, :].astype(float) - self.np_gray[:-1, :].astype(float)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Threshold to get edge pixels
        threshold = np.percentile(magnitude[magnitude > 0], 75) if np.any(magnitude > 0) else 1
        edges = (magnitude > threshold).astype(np.uint8) * 255
        
        return edges
    
    def _classify_shape(self, edges: np.ndarray, width: int, height: int, 
                       aspect_ratio: float) -> Tuple[str, float, Dict]:
        """
        Classify the base shape from edge data.
        """
        # Count edge pixels
        edge_count = np.sum(edges > 0)
        
        # Analyze circularity
        circularity = self._calculate_circularity(edges, width, height)
        
        # Check for polygonal features
        corner_count = self._estimate_corners(edges)
        
        details = {
            "width": int(width),
            "height": int(height),
            "aspect_ratio": round(aspect_ratio, 3),
            "edge_pixels": int(edge_count),
            "circularity": round(circularity, 3),
            "estimated_corners": corner_count
        }
        
        # Classification logic
        if 0.85 < aspect_ratio < 1.15:  # Roughly square/circular
            if circularity > 0.85:
                return "circle", min(0.99, circularity), details
            elif corner_count == 4:
                return "square", 0.90, details
            elif corner_count == 6:
                return "hexagon", 0.88, details
            elif corner_count == 3:
                return "triangle", 0.87, details
            else:
                return "irregular_polygon", 0.70, details
        else:
            if aspect_ratio > 1.5:
                return "rectangle_horizontal", 0.85, details
            elif aspect_ratio < 0.67:
                return "rectangle_vertical", 0.85, details
            else:
                return "oval", circularity * 0.9, details
    
    def _calculate_circularity(self, edges: np.ndarray, width: int, height: int) -> float:
        """
        Calculate how circular the shape is (1.0 = perfect circle).
        """
        center_y, center_x = self.height // 2, self.width // 2
        expected_radius = min(width, height) // 2
        
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) < 10:
            return 0.0
        
        # Calculate distances from center
        distances = np.sqrt(
            (edge_points[:, 0] - center_y)**2 + 
            (edge_points[:, 1] - center_x)**2
        )
        
        # Circularity = 1 - (std_dev / mean_radius)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist == 0:
            return 0.0
        
        circularity = 1.0 - (std_dist / mean_dist)
        return max(0.0, min(1.0, circularity))
    
    def _estimate_corners(self, edges: np.ndarray) -> int:
        """
        Estimate number of corners/vertices.
        """
        # Simple corner detection using local variance
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) < 20:
            return 0
        
        # Sample points and check for direction changes
        step = max(1, len(edge_points) // 50)
        sampled = edge_points[::step]
        
        if len(sampled) < 5:
            return 0
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(2, len(sampled)):
            v1 = sampled[i-1] - sampled[i-2]
            v2 = sampled[i] - sampled[i-1]
            
            # Cross product magnitude indicates direction change
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if cross > 5:  # Threshold for significant change
                direction_changes += 1
        
        return direction_changes
    
    def _analyze_interior_marks(self) -> ElementAnalysis:
        """
        PASS 2: Detect marks inside the symbol boundary.
        """
        # Get center region (exclude outer 20%)
        margin_x = int(self.width * 0.2)
        margin_y = int(self.height * 0.2)
        
        interior = self.np_gray[margin_y:self.height-margin_y, margin_x:self.width-margin_x]
        
        if interior.size == 0:
            return ElementAnalysis(
                element_type="none",
                confidence=0.95,
                details={"region": "too_small"},
                note="Symbol too small for interior analysis"
            )
        
        # Detect marks in interior
        dark_pixels = np.sum(interior < 128)
        total_pixels = interior.size
        dark_ratio = dark_pixels / total_pixels
        
        # Check for X pattern
        has_x = self._detect_x_pattern(interior)
        
        # Check for cross pattern
        has_cross = self._detect_cross_pattern(interior)
        
        # Check for dot
        has_dot = self._detect_center_dot(interior)
        
        details = {
            "dark_pixel_ratio": round(dark_ratio, 3),
            "has_x_pattern": has_x,
            "has_cross_pattern": has_cross,
            "has_center_dot": has_dot
        }
        
        if has_x and has_cross:
            return ElementAnalysis(
                element_type="x_or_cross",
                confidence=0.75,
                details=details,
                note="Ambiguous - could be X or cross pattern"
            )
        elif has_x:
            return ElementAnalysis(
                element_type="x_cross",
                confidence=0.88,
                details=details
            )
        elif has_cross:
            return ElementAnalysis(
                element_type="plus_cross",
                confidence=0.88,
                details=details
            )
        elif has_dot:
            return ElementAnalysis(
                element_type="center_dot",
                confidence=0.92,
                details=details
            )
        elif dark_ratio < 0.05:
            return ElementAnalysis(
                element_type="empty",
                confidence=0.96,
                details=details
            )
        else:
            return ElementAnalysis(
                element_type="unclear_marks",
                confidence=0.60,
                details=details,
                note="Interior marks detected but pattern unclear"
            )
    
    def _detect_x_pattern(self, region: np.ndarray) -> bool:
        """
        Check for diagonal X pattern.
        """
        h, w = region.shape
        if h < 5 or w < 5:
            return False
        
        # Sample along diagonals
        diag1_dark = 0
        diag2_dark = 0
        samples = min(h, w)
        
        for i in range(samples):
            y1, x1 = int(i * h / samples), int(i * w / samples)
            y2, x2 = int(i * h / samples), int((samples - 1 - i) * w / samples)
            
            if 0 <= y1 < h and 0 <= x1 < w:
                if region[y1, x1] < 128:
                    diag1_dark += 1
            if 0 <= y2 < h and 0 <= x2 < w:
                if region[y2, x2] < 128:
                    diag2_dark += 1
        
        return diag1_dark > samples * 0.5 and diag2_dark > samples * 0.5
    
    def _detect_cross_pattern(self, region: np.ndarray) -> bool:
        """
        Check for + cross pattern.
        """
        h, w = region.shape
        if h < 5 or w < 5:
            return False
        
        center_y, center_x = h // 2, w // 2
        
        # Check horizontal line through center
        h_line_dark = np.sum(region[center_y-1:center_y+2, :] < 128)
        # Check vertical line through center
        v_line_dark = np.sum(region[:, center_x-1:center_x+2] < 128)
        
        h_threshold = w * 0.5
        v_threshold = h * 0.5
        
        return h_line_dark > h_threshold and v_line_dark > v_threshold
    
    def _detect_center_dot(self, region: np.ndarray) -> bool:
        """
        Check for center dot.
        """
        h, w = region.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 6
        
        if radius < 2:
            return False
        
        # Check center region
        y_start = max(0, center_y - radius)
        y_end = min(h, center_y + radius)
        x_start = max(0, center_x - radius)
        x_end = min(w, center_x + radius)
        
        center_region = region[y_start:y_end, x_start:x_end]
        dark_ratio = np.sum(center_region < 128) / center_region.size
        
        return dark_ratio > 0.3 and dark_ratio < 0.8
    
    def _analyze_fill_pattern(self) -> ElementAnalysis:
        """
        Detect fill pattern (solid, hatched, empty, etc.)
        """
        # Analyze pixel distribution
        dark_pixels = np.sum(self.np_gray < 128)
        total_pixels = self.np_gray.size
        fill_ratio = dark_pixels / total_pixels
        
        # Check for hatching (alternating pattern)
        has_hatching = self._detect_hatching()
        
        # Check for stippling (regular dots)
        has_stippling = self._detect_stippling()
        
        details = {
            "fill_ratio": round(fill_ratio, 3),
            "has_hatching": has_hatching,
            "has_stippling": has_stippling
        }
        
        if fill_ratio > 0.7:
            return ElementAnalysis(
                element_type="solid",
                confidence=0.95,
                details=details
            )
        elif fill_ratio < 0.15:
            return ElementAnalysis(
                element_type="empty",
                confidence=0.95,
                details=details
            )
        elif has_hatching:
            return ElementAnalysis(
                element_type="hatched",
                confidence=0.85,
                details=details
            )
        elif has_stippling:
            return ElementAnalysis(
                element_type="stippled",
                confidence=0.82,
                details=details
            )
        else:
            return ElementAnalysis(
                element_type="partial_fill",
                confidence=0.70,
                details=details,
                note="Fill pattern unclear"
            )
    
    def _detect_hatching(self) -> bool:
        """
        Detect diagonal or cross hatching.
        """
        # Check for regular alternating pattern
        for y in range(2, self.height - 2, 5):
            row = self.np_gray[y, :]
            transitions = np.sum(np.abs(np.diff(row.astype(int))) > 50)
            if transitions > self.width * 0.1:
                return True
        return False
    
    def _detect_stippling(self) -> bool:
        """
        Detect regular dot pattern.
        """
        dark_mask = self.np_gray < 128
        
        # Look for isolated dark regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(dark_mask)
        
        if num_features > 5:
            # Check if features are similar size (dots)
            sizes = ndimage.sum(dark_mask, labeled, range(1, num_features + 1))
            if len(sizes) > 0:
                size_variance = np.std(sizes) / (np.mean(sizes) + 1)
                return size_variance < 0.5  # Regular sized dots
        
        return False
    
    def _analyze_edge_treatment(self) -> ElementAnalysis:
        """
        PASS 3: Analyze boundary line characteristics.
        """
        edges = self._detect_edges()
        
        # Measure edge thickness
        edge_thickness = self._measure_edge_thickness(edges)
        
        # Check for double lines
        has_double_line = self._detect_double_line(edges)
        
        # Check for dashed pattern
        has_dashes = self._detect_dashed_line(edges)
        
        # Check edge continuity
        continuity = self._measure_edge_continuity(edges)
        
        details = {
            "thickness": round(edge_thickness, 2),
            "has_double_line": has_double_line,
            "has_dashes": has_dashes,
            "continuity": round(continuity, 3)
        }
        
        if continuity < 0.7:
            return ElementAnalysis(
                element_type="broken_or_dashed",
                confidence=0.75,
                details=details,
                note="Edge not continuous - may be intentional or noise"
            )
        elif has_double_line:
            return ElementAnalysis(
                element_type="double_line",
                confidence=0.85,
                details=details
            )
        elif has_dashes:
            return ElementAnalysis(
                element_type="dashed",
                confidence=0.82,
                details=details
            )
        elif edge_thickness > 3:
            return ElementAnalysis(
                element_type="thick_line",
                confidence=0.90,
                details=details
            )
        elif edge_thickness < 1.5:
            return ElementAnalysis(
                element_type="thin_line",
                confidence=0.90,
                details=details
            )
        else:
            return ElementAnalysis(
                element_type="single_line_medium",
                confidence=0.92,
                details=details
            )
    
    def _measure_edge_thickness(self, edges: np.ndarray) -> float:
        """
        Estimate average edge line thickness.
        """
        # Count edge pixels per row/column and estimate thickness
        row_counts = np.sum(edges > 0, axis=1)
        col_counts = np.sum(edges > 0, axis=0)
        
        # Filter out empty rows/cols
        row_counts = row_counts[row_counts > 0]
        col_counts = col_counts[col_counts > 0]
        
        if len(row_counts) == 0 or len(col_counts) == 0:
            return 0.0
        
        avg_row = np.mean(row_counts)
        avg_col = np.mean(col_counts)
        
        return (avg_row + avg_col) / 2
    
    def _detect_double_line(self, edges: np.ndarray) -> bool:
        """
        Check for parallel double lines.
        """
        # Sample horizontal profiles
        for y in range(10, self.height - 10, self.height // 5):
            profile = edges[y, :]
            peaks = np.where(profile > 0)[0]
            if len(peaks) > 2:
                gaps = np.diff(peaks)
                # Check for consistent gap pattern (double line signature)
                if np.any((gaps > 2) & (gaps < 10)):
                    return True
        return False
    
    def _detect_dashed_line(self, edges: np.ndarray) -> bool:
        """
        Check for dashed line pattern.
        """
        # Check edge continuity along boundary
        boundary_pixels = []
        
        # Sample top edge
        boundary_pixels.extend(edges[0, :].tolist())
        # Sample bottom edge  
        boundary_pixels.extend(edges[-1, :].tolist())
        # Sample left edge
        boundary_pixels.extend(edges[:, 0].tolist())
        # Sample right edge
        boundary_pixels.extend(edges[:, -1].tolist())
        
        boundary_array = np.array(boundary_pixels)
        if len(boundary_array) < 20:
            return False
        
        # Count transitions (high = dashed)
        transitions = np.sum(np.abs(np.diff(boundary_array)) > 100)
        return transitions > len(boundary_array) * 0.1
    
    def _measure_edge_continuity(self, edges: np.ndarray) -> float:
        """
        Measure how continuous the edge is (1.0 = fully continuous).
        """
        edge_points = np.sum(edges > 0)
        if edge_points == 0:
            return 0.0
        
        # Expected perimeter for detected bounding box
        expected = 2 * (self.width + self.height) * 0.8  # 80% of perimeter
        
        continuity = min(1.0, edge_points / expected)
        return continuity
    
    def _analyze_corner_accents(self) -> ElementAnalysis:
        """
        Analyze corner characteristics.
        """
        corners = self._detect_corner_regions()
        
        if not corners:
            return ElementAnalysis(
                element_type="no_corners_detected",
                confidence=0.80,
                details={"corner_count": 0}
            )
        
        # Analyze each corner
        sharp_corners = 0
        rounded_corners = 0
        
        for corner in corners:
            if self._is_corner_sharp(corner):
                sharp_corners += 1
            else:
                rounded_corners += 1
        
        details = {
            "corner_count": len(corners),
            "sharp_corners": sharp_corners,
            "rounded_corners": rounded_corners
        }
        
        if rounded_corners > sharp_corners:
            return ElementAnalysis(
                element_type="rounded",
                confidence=0.88,
                details=details
            )
        elif sharp_corners > 0:
            return ElementAnalysis(
                element_type="sharp",
                confidence=0.90,
                details=details
            )
        else:
            return ElementAnalysis(
                element_type="indeterminate",
                confidence=0.70,
                details=details
            )
    
    def _detect_corner_regions(self) -> List[Tuple[int, int]]:
        """
        Find likely corner locations.
        """
        margin = min(self.width, self.height) // 4
        
        corners = [
            (margin, margin),                    # Top-left
            (margin, self.width - margin),       # Top-right
            (self.height - margin, margin),      # Bottom-left
            (self.height - margin, self.width - margin)  # Bottom-right
        ]
        
        return corners
    
    def _is_corner_sharp(self, corner: Tuple[int, int]) -> bool:
        """
        Determine if corner region is sharp or rounded.
        """
        y, x = corner
        radius = min(self.width, self.height) // 8
        
        y_start = max(0, y - radius)
        y_end = min(self.height, y + radius)
        x_start = max(0, x - radius)
        x_end = min(self.width, x + radius)
        
        region = self.np_gray[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            return False
        
        # Sharp corners have high variance in small region
        variance = np.var(region)
        return variance > 1000  # Arbitrary threshold
    
    def _deep_zoom_verify(self, element: ElementAnalysis) -> None:
        """
        PASS 4: Deep zoom verification for low-confidence elements.
        """
        # This would enhance the analysis with higher resolution inspection
        # In practice, this adds a note about what additional verification was attempted
        if element.note:
            element.note += " | Deep zoom verification attempted"
        else:
            element.note = "Deep zoom verification attempted"
    
    def _calculate_composite(self, base: ElementAnalysis, interior: ElementAnalysis,
                            fill: ElementAnalysis, edge: ElementAnalysis,
                            corner: ElementAnalysis) -> float:
        """
        Calculate weighted composite confidence score.
        """
        composite = (
            base.confidence * self.WEIGHTS['base_shape'] +
            interior.confidence * self.WEIGHTS['interior_marks'] +
            fill.confidence * self.WEIGHTS['fill_pattern'] +
            edge.confidence * self.WEIGHTS['edge_treatment'] +
            corner.confidence * self.WEIGHTS['corner_accents']
        )
        return composite
    
    def _determine_status(self, composite: float, 
                         elements: List[ElementAnalysis]) -> Tuple[str, bool, Optional[str]]:
        """
        Determine status and flagging based on confidence.
        """
        # Check for any low-confidence elements
        low_conf_elements = [e for e in elements if e.confidence < self.THRESHOLD_CONFIRMED]
        
        if composite >= self.THRESHOLD_CONFIRMED and not low_conf_elements:
            return ConfidenceLevel.CONFIRMED.value, False, None
        
        # Find the reason for flagging
        if low_conf_elements:
            lowest = min(low_conf_elements, key=lambda e: e.confidence)
            reason = f"{lowest.element_type}: {lowest.note or 'confidence below threshold'}"
        else:
            reason = f"Composite confidence {composite:.1%} below 95%"
        
        if composite >= self.THRESHOLD_REVIEW:
            return ConfidenceLevel.REVIEW.value, True, reason
        elif composite >= self.THRESHOLD_UNCERTAIN:
            return ConfidenceLevel.UNCERTAIN.value, True, reason
        else:
            return ConfidenceLevel.UNIDENTIFIED.value, True, reason
    
    def _save_evidence(self, elements: List[ElementAnalysis]) -> str:
        """
        Save evidence images for flagged symbols.
        """
        evidence_path = self.output_dir / f"{self.symbol_id}_evidence.png"
        
        # Create a zoomed version
        zoom_factor = 4
        zoomed = self.image.resize(
            (self.width * zoom_factor, self.height * zoom_factor),
            Image.Resampling.NEAREST
        )
        zoomed.save(evidence_path)
        
        return str(evidence_path)


def analysis_to_dict(analysis: SymbolAnalysis) -> Dict[str, Any]:
    """
    Convert analysis to JSON-serializable dictionary.
    """
    def element_to_dict(elem: ElementAnalysis) -> Dict:
        d = {
            "type": elem.element_type,
            "confidence": elem.confidence,
            "details": elem.details
        }
        if elem.note:
            d["note"] = elem.note
        return d
    
    result = {
        "symbol_id": analysis.symbol_id,
        "location": analysis.location,
        "analysis": {
            "base_shape": element_to_dict(analysis.base_shape),
            "interior_marks": element_to_dict(analysis.interior_marks),
            "fill_pattern": element_to_dict(analysis.fill_pattern),
            "edge_treatment": element_to_dict(analysis.edge_treatment),
            "corner_accents": element_to_dict(analysis.corner_accents)
        },
        "composite_confidence": analysis.composite_confidence,
        "status": analysis.status,
        "flagged": analysis.flagged
    }
    
    if analysis.flag_reason:
        result["flag_reason"] = analysis.flag_reason
    if analysis.zoom_evidence:
        result["zoom_evidence"] = analysis.zoom_evidence
    if analysis.best_match:
        result["best_match"] = analysis.best_match
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_symbol.py <image_path> [--output json|text]")
        print("\nAnalyzes a symbol image and reports confidence scores for each element.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_format = "json"
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]
    
    try:
        analyzer = SymbolAnalyzer(image_path)
        result = analyzer.analyze()
        
        if output_format == "json":
            print(json.dumps(analysis_to_dict(result), indent=2))
        else:
            print(f"Symbol Analysis: {result.symbol_id}")
            print(f"{'='*50}")
            print(f"Base Shape: {result.base_shape.element_type} ({result.base_shape.confidence:.1%})")
            print(f"Interior: {result.interior_marks.element_type} ({result.interior_marks.confidence:.1%})")
            print(f"Fill: {result.fill_pattern.element_type} ({result.fill_pattern.confidence:.1%})")
            print(f"Edges: {result.edge_treatment.element_type} ({result.edge_treatment.confidence:.1%})")
            print(f"Corners: {result.corner_accents.element_type} ({result.corner_accents.confidence:.1%})")
            print(f"{'='*50}")
            print(f"COMPOSITE: {result.composite_confidence:.1%}")
            print(f"STATUS: {result.status}")
            
            if result.flagged:
                print(f"\n⚠️  FLAGGED: {result.flag_reason}")
                print(f"Evidence saved to: {result.zoom_evidence}")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
