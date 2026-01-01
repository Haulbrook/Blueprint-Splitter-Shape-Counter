#!/usr/bin/env python3
"""
Shape & Symbol Master - Batch Analyzer
Processes multiple symbol images and generates a comprehensive report.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the analyzer
from analyze_symbol import SymbolAnalyzer, analysis_to_dict


class BatchAnalyzer:
    """
    Batch processing for multiple symbol images.
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    
    def __init__(self, input_dir: str, output_dir: str = "./batch_results",
                 legend_path: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evidence_dir = self.output_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Load legend if provided
        self.legend = None
        if legend_path and Path(legend_path).exists():
            with open(legend_path, 'r') as f:
                self.legend = json.load(f)
        
        # Results storage
        self.results: List[Dict] = []
        self.flagged: List[Dict] = []
        self.confirmed: List[Dict] = []
        self.errors: List[Dict] = []
    
    def find_images(self) -> List[Path]:
        """
        Find all supported image files in input directory.
        """
        images = []
        
        if self.input_dir.is_file():
            if self.input_dir.suffix.lower() in self.SUPPORTED_FORMATS:
                images.append(self.input_dir)
        else:
            for ext in self.SUPPORTED_FORMATS:
                images.extend(self.input_dir.glob(f"*{ext}"))
                images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def analyze_single(self, image_path: Path) -> Dict:
        """
        Analyze a single image and return result.
        """
        try:
            analyzer = SymbolAnalyzer(
                str(image_path), 
                output_dir=str(self.evidence_dir)
            )
            result = analyzer.analyze()
            return analysis_to_dict(result)
        
        except Exception as e:
            return {
                "symbol_id": image_path.stem,
                "error": str(e),
                "status": "ERROR"
            }
    
    def run(self, max_workers: int = 4) -> Dict:
        """
        Run batch analysis on all images.
        """
        images = self.find_images()
        
        if not images:
            return {
                "status": "error",
                "message": f"No supported images found in {self.input_dir}"
            }
        
        print(f"Found {len(images)} images to analyze...")
        
        # Process images
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] Analyzing {img_path.name}...", end=" ")
            
            result = self.analyze_single(img_path)
            result["source_file"] = str(img_path)
            
            self.results.append(result)
            
            if result.get("error"):
                self.errors.append(result)
                print("❌ ERROR")
            elif result.get("flagged"):
                self.flagged.append(result)
                print(f"⚠️  {result.get('composite_confidence', 0):.1%}")
            else:
                self.confirmed.append(result)
                print(f"✅ {result.get('composite_confidence', 0):.1%}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive analysis report.
        """
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "total_analyzed": len(self.results)
            },
            "summary": {
                "confirmed": len(self.confirmed),
                "flagged_for_review": len(self.flagged),
                "errors": len(self.errors),
                "confirmation_rate": len(self.confirmed) / max(len(self.results), 1)
            },
            "flagged_symbols": [],
            "confirmed_symbols": [],
            "errors": []
        }
        
        # Add flagged symbols (full detail)
        for item in self.flagged:
            report["flagged_symbols"].append({
                "symbol_id": item.get("symbol_id"),
                "source_file": item.get("source_file"),
                "confidence": item.get("composite_confidence"),
                "status": item.get("status"),
                "flag_reason": item.get("flag_reason"),
                "evidence": item.get("zoom_evidence"),
                "analysis_summary": {
                    "base_shape": item.get("analysis", {}).get("base_shape", {}).get("type"),
                    "interior": item.get("analysis", {}).get("interior_marks", {}).get("type"),
                    "fill": item.get("analysis", {}).get("fill_pattern", {}).get("type")
                }
            })
        
        # Add confirmed symbols (condensed)
        for item in self.confirmed:
            report["confirmed_symbols"].append({
                "symbol_id": item.get("symbol_id"),
                "source_file": item.get("source_file"),
                "confidence": item.get("composite_confidence"),
                "base_shape": item.get("analysis", {}).get("base_shape", {}).get("type"),
                "interior": item.get("analysis", {}).get("interior_marks", {}).get("type")
            })
        
        # Add errors
        for item in self.errors:
            report["errors"].append({
                "symbol_id": item.get("symbol_id"),
                "source_file": item.get("source_file"),
                "error": item.get("error")
            })
        
        return report
    
    def save_report(self, report: Dict, format: str = "json") -> str:
        """
        Save report to file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_path = self.output_dir / f"batch_report_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        else:  # text format
            output_path = self.output_dir / f"batch_report_{timestamp}.txt"
            with open(output_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("SHAPE & SYMBOL MASTER - BATCH ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generated: {report['metadata']['timestamp']}\n")
                f.write(f"Input: {report['metadata']['input_directory']}\n")
                f.write(f"Total analyzed: {report['metadata']['total_analyzed']}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("SUMMARY\n")
                f.write("-" * 60 + "\n")
                f.write(f"✅ Confirmed: {report['summary']['confirmed']}\n")
                f.write(f"⚠️  Flagged for review: {report['summary']['flagged_for_review']}\n")
                f.write(f"❌ Errors: {report['summary']['errors']}\n")
                f.write(f"Confirmation rate: {report['summary']['confirmation_rate']:.1%}\n\n")
                
                if report['flagged_symbols']:
                    f.write("-" * 60 + "\n")
                    f.write("⚠️  SYMBOLS REQUIRING HUMAN REVIEW\n")
                    f.write("-" * 60 + "\n")
                    for sym in report['flagged_symbols']:
                        f.write(f"\n{sym['symbol_id']}:\n")
                        f.write(f"  File: {sym['source_file']}\n")
                        f.write(f"  Confidence: {sym['confidence']:.1%}\n")
                        f.write(f"  Status: {sym['status']}\n")
                        f.write(f"  Reason: {sym['flag_reason']}\n")
                        f.write(f"  Shape: {sym['analysis_summary']['base_shape']}\n")
                        f.write(f"  Interior: {sym['analysis_summary']['interior']}\n")
                        if sym.get('evidence'):
                            f.write(f"  Evidence: {sym['evidence']}\n")
                
                f.write("\n")
        
        return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_analyze.py <input_directory> [options]")
        print("\nOptions:")
        print("  --output DIR      Output directory (default: ./batch_results)")
        print("  --legend FILE     Legend JSON file for matching")
        print("  --format FORMAT   Output format: json or text (default: json)")
        print("\nAnalyzes all symbol images in a directory.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = "./batch_results"
    legend_path = None
    output_format = "json"
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--legend" and i + 1 < len(sys.argv):
            legend_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--format" and i + 1 < len(sys.argv):
            output_format = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    print("=" * 60)
    print("SHAPE & SYMBOL MASTER - BATCH ANALYSIS")
    print("=" * 60)
    print()
    
    try:
        analyzer = BatchAnalyzer(input_dir, output_dir, legend_path)
        report = analyzer.run()
        
        # Save report
        report_path = analyzer.save_report(report, output_format)
        
        print()
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"✅ Confirmed: {report['summary']['confirmed']}")
        print(f"⚠️  Flagged: {report['summary']['flagged_for_review']}")
        print(f"❌ Errors: {report['summary']['errors']}")
        print()
        print(f"Report saved to: {report_path}")
        
        if report['flagged_symbols']:
            print()
            print("⚠️  ACTION REQUIRED: Review flagged symbols before counting!")
            for sym in report['flagged_symbols'][:5]:  # Show first 5
                print(f"   - {sym['symbol_id']}: {sym['flag_reason']}")
            if len(report['flagged_symbols']) > 5:
                print(f"   ... and {len(report['flagged_symbols']) - 5} more")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
