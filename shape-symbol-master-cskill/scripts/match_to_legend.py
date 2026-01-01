#!/usr/bin/env python3
"""
Shape & Symbol Master - Legend Matcher
Compares extracted symbols against a reference legend with confidence scoring.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install Pillow numpy --break-system-packages")
    sys.exit(1)


@dataclass
class MatchResult:
    legend_code: str
    legend_name: str
    match_confidence: float
    shape_similarity: float
    interior_similarity: float
    size_ratio: float
    notes: Optional[str] = None


class LegendMatcher:
    """
    Matches symbols to a reference legend using multi-feature comparison.
    """
    
    CONFIDENCE_THRESHOLD = 0.95
    
    def __init__(self, legend_data: Dict[str, Any]):
        """
        Initialize with legend data.
        
        legend_data format:
        {
            "APS": {
                "name": "Japanese Maple",
                "image_path": "path/to/symbol.png",  # Optional
                "base_shape": "circle",
                "interior": "empty",
                "fill": "empty",
                "typical_size": 45  # pixels
            }
        }
        """
        self.legend = legend_data
        self.legend_signatures = self._compute_legend_signatures()
    
    def _compute_legend_signatures(self) -> Dict[str, Dict]:
        """
        Pre-compute signatures for all legend entries.
        """
        signatures = {}
        
        for code, entry in self.legend.items():
            sig = {
                "code": code,
                "name": entry.get("name", code),
                "base_shape": entry.get("base_shape", "unknown"),
                "interior": entry.get("interior", "unknown"),
                "fill": entry.get("fill", "unknown"),
                "size": entry.get("typical_size", 40)
            }
            
            # Load reference image if available
            if "image_path" in entry and Path(entry["image_path"]).exists():
                sig["reference_image"] = self._load_signature_from_image(entry["image_path"])
            
            signatures[code] = sig
        
        return signatures
    
    def _load_signature_from_image(self, path: str) -> Dict:
        """
        Compute visual signature from reference image.
        """
        try:
            img = Image.open(path).convert('L')
            arr = np.array(img)
            
            # Compute basic statistics
            return {
                "mean_intensity": float(np.mean(arr)),
                "std_intensity": float(np.std(arr)),
                "dark_ratio": float(np.sum(arr < 128) / arr.size),
                "dimensions": img.size
            }
        except Exception:
            return {}
    
    def match_symbol(self, symbol_analysis: Dict) -> List[MatchResult]:
        """
        Find best matches for a symbol analysis result.
        
        Returns top matches sorted by confidence (descending).
        """
        matches = []
        
        symbol_base = symbol_analysis.get("analysis", {}).get("base_shape", {}).get("type", "unknown")
        symbol_interior = symbol_analysis.get("analysis", {}).get("interior_marks", {}).get("type", "unknown")
        symbol_fill = symbol_analysis.get("analysis", {}).get("fill_pattern", {}).get("type", "unknown")
        symbol_size = symbol_analysis.get("location", {}).get("width", 40)
        
        for code, sig in self.legend_signatures.items():
            # Calculate shape similarity
            shape_sim = self._shape_similarity(symbol_base, sig["base_shape"])
            
            # Calculate interior similarity
            interior_sim = self._interior_similarity(symbol_interior, sig["interior"])
            
            # Calculate size ratio
            size_ratio = min(symbol_size, sig["size"]) / max(symbol_size, sig["size"], 1)
            
            # Calculate fill similarity
            fill_sim = self._fill_similarity(symbol_fill, sig["fill"])
            
            # Composite confidence
            confidence = (
                shape_sim * 0.40 +
                interior_sim * 0.30 +
                fill_sim * 0.15 +
                size_ratio * 0.15
            )
            
            notes = None
            if confidence < self.CONFIDENCE_THRESHOLD:
                notes = self._generate_uncertainty_notes(
                    symbol_base, sig["base_shape"],
                    symbol_interior, sig["interior"],
                    shape_sim, interior_sim
                )
            
            matches.append(MatchResult(
                legend_code=code,
                legend_name=sig["name"],
                match_confidence=round(confidence, 4),
                shape_similarity=round(shape_sim, 4),
                interior_similarity=round(interior_sim, 4),
                size_ratio=round(size_ratio, 4),
                notes=notes
            ))
        
        # Sort by confidence descending
        matches.sort(key=lambda m: m.match_confidence, reverse=True)
        
        return matches
    
    def _shape_similarity(self, detected: str, expected: str) -> float:
        """
        Calculate similarity between detected and expected shapes.
        """
        if detected == expected:
            return 1.0
        
        # Shape groups (similar shapes get partial credit)
        shape_groups = {
            "circular": ["circle", "oval", "ellipse"],
            "rectangular": ["square", "rectangle", "rectangle_horizontal", "rectangle_vertical"],
            "polygonal": ["hexagon", "pentagon", "octagon"],
            "triangular": ["triangle", "irregular_polygon"]
        }
        
        # Check if in same group
        for group_name, shapes in shape_groups.items():
            if detected in shapes and expected in shapes:
                return 0.85
        
        # Check fuzzy matches
        fuzzy_matches = {
            ("circle", "hexagon"): 0.70,
            ("circle", "oval"): 0.90,
            ("square", "rectangle"): 0.85,
            ("irregular_polygon", "circle"): 0.60
        }
        
        pair = tuple(sorted([detected, expected]))
        if pair in fuzzy_matches:
            return fuzzy_matches[pair]
        
        return 0.30  # Default low similarity
    
    def _interior_similarity(self, detected: str, expected: str) -> float:
        """
        Calculate similarity between detected and expected interior marks.
        """
        if detected == expected:
            return 1.0
        
        if detected == "empty" and expected == "none":
            return 1.0
        if detected == "none" and expected == "empty":
            return 1.0
        
        # Similar interior patterns
        similar = {
            ("x_cross", "plus_cross"): 0.70,
            ("x_cross", "x_or_cross"): 0.85,
            ("plus_cross", "x_or_cross"): 0.85,
            ("center_dot", "empty"): 0.60,
            ("unclear_marks", "x_cross"): 0.50,
            ("unclear_marks", "plus_cross"): 0.50
        }
        
        pair = tuple(sorted([detected, expected]))
        if pair in similar:
            return similar[pair]
        
        return 0.20
    
    def _fill_similarity(self, detected: str, expected: str) -> float:
        """
        Calculate similarity between fill patterns.
        """
        if detected == expected:
            return 1.0
        
        similar = {
            ("empty", "none"): 1.0,
            ("hatched", "partial_fill"): 0.80,
            ("stippled", "partial_fill"): 0.80,
            ("solid", "hatched"): 0.50
        }
        
        pair = tuple(sorted([detected, expected]))
        if pair in similar:
            return similar[pair]
        
        return 0.30
    
    def _generate_uncertainty_notes(self, det_shape: str, exp_shape: str,
                                   det_interior: str, exp_interior: str,
                                   shape_sim: float, interior_sim: float) -> str:
        """
        Generate human-readable notes about match uncertainty.
        """
        issues = []
        
        if shape_sim < 0.90:
            issues.append(f"Shape: detected '{det_shape}' vs expected '{exp_shape}'")
        
        if interior_sim < 0.90:
            issues.append(f"Interior: detected '{det_interior}' vs expected '{exp_interior}'")
        
        if issues:
            return " | ".join(issues)
        
        return "Low overall confidence"


def match_to_legend(symbol_analysis: Dict, legend_data: Dict) -> Dict:
    """
    Main function to match a symbol to a legend.
    
    Returns the match result with top candidates.
    """
    matcher = LegendMatcher(legend_data)
    matches = matcher.match_symbol(symbol_analysis)
    
    # Take top 3 matches
    top_matches = matches[:3]
    
    result = {
        "symbol_id": symbol_analysis.get("symbol_id", "unknown"),
        "best_match": None,
        "alternatives": [],
        "requires_confirmation": True
    }
    
    if top_matches:
        best = top_matches[0]
        result["best_match"] = {
            "code": best.legend_code,
            "name": best.legend_name,
            "confidence": best.match_confidence
        }
        
        if best.match_confidence >= 0.95:
            result["requires_confirmation"] = False
        
        if best.notes:
            result["best_match"]["notes"] = best.notes
        
        # Add alternatives if close
        for alt in top_matches[1:]:
            if alt.match_confidence > 0.70:
                result["alternatives"].append({
                    "code": alt.legend_code,
                    "name": alt.legend_name,
                    "confidence": alt.match_confidence
                })
    
    return result


def load_legend_from_json(json_path: str) -> Dict:
    """
    Load legend data from JSON file.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python match_to_legend.py <symbol_analysis.json> <legend.json>")
        print("\nMatches a symbol analysis result against a legend.")
        print("\nLegend JSON format:")
        print(json.dumps({
            "APS": {
                "name": "Japanese Maple",
                "base_shape": "circle",
                "interior": "empty",
                "fill": "empty",
                "typical_size": 45
            }
        }, indent=2))
        sys.exit(1)
    
    analysis_path = sys.argv[1]
    legend_path = sys.argv[2]
    
    try:
        with open(analysis_path, 'r') as f:
            symbol_analysis = json.load(f)
        
        legend_data = load_legend_from_json(legend_path)
        
        result = match_to_legend(symbol_analysis, legend_data)
        
        print(json.dumps(result, indent=2))
        
        # Summary output
        if result["best_match"]:
            conf = result["best_match"]["confidence"]
            status = "✅ CONFIRMED" if conf >= 0.95 else "⚠️ REVIEW REQUIRED"
            print(f"\n{status}")
            print(f"Best match: {result['best_match']['code']} - {result['best_match']['name']}")
            print(f"Confidence: {conf:.1%}")
            
            if result["alternatives"]:
                print("\nAlternatives:")
                for alt in result["alternatives"]:
                    print(f"  - {alt['code']}: {alt['confidence']:.1%}")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
