#!/usr/bin/env python3
"""Generate prose quality ratings from WordPress posts."""

from pathlib import Path
from prose_analyzer import analyze_posts


def main() -> None:
    """Main entry point."""
    print("WordPress Prose Quality Analyzer")
    print("=" * 40)
    
    # Try to find input file in any supported format
    input_file = None
    for ext in [".jsonl", ".json", ".txt"]:
        candidate = Path(f"posts{ext}")
        if candidate.exists():
            input_file = candidate
            break
    
    if input_file is None:
        input_file = Path("posts.txt")
        if not input_file.exists():
            raise SystemExit(
                "Input file not found. Expected one of: posts.jsonl, posts.json, posts.txt"
            )
    
    output_file = Path("post_rankings.csv")
    
    analyze_posts(input_file, output_file)
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
