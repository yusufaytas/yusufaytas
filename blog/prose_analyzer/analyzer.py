"""Main analysis orchestration."""

import csv
import json
from pathlib import Path

from .models import Post
from .parser import load_posts
from .metrics import calculate_metrics
from .scorer import calculate_scores
from .summarizer import generate_summary
from .ranker import rank_posts, percentile_from_rank


def analyze_posts(
    input_file: Path,
    output_file: Path
) -> None:
    """Analyze posts and write ranked results to CSV."""
    print(f"Loading posts from {input_file}...")
    posts = load_posts(input_file)
    
    if not posts:
        print("No posts found")
        return

    print(f"Found {len(posts)} posts. Analyzing...")
    
    # Calculate metrics and scores for each post
    rows = []
    for i, post in enumerate(posts, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing post {i}/{len(posts)}: {post.title[:50]}...")
        
        metrics = calculate_metrics(post.content, categories=post.categories)
        scores = calculate_scores(metrics)
        
        rows.append({
            "post_id": post.post_id,
            "title": post.title,
            "date": post.date,
            "url": post.url,
            "slug": post.slug,
            "categories": post.categories,
            "tags": post.tags,
            "language": metrics.language,
            "summary": generate_summary(
                post.content,
                title=post.title,
                language=metrics.language
            ),
            "quality_score": scores.overall_rating_100,
        })

    # Sort by quality score (highest first)
    print(f"Writing results to {output_file}...")
    ranked = sorted(rows, key=lambda r: (-r["quality_score"], r["post_id"]))
    
    export_fields = [
        "post_id",
        "title",
        "date",
        "url",
        "slug",
        "categories",
        "tags",
        "language",
        "quality_score",
        "summary",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=export_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in export_fields}
            for row in ranked
        )

    print(f"Wrote {output_file} with {len(ranked)} rows")
