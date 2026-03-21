"""Rank posts by quality scores."""

from scipy.stats import rankdata


def rank_posts(rows: list[dict], score_key: str = "overall_rating_100") -> dict[int, int]:
    """Rank posts by score (descending), return post_id -> rank mapping."""
    # Add tiny tiebreaker based on post_id for stable sorting
    adjusted_scores = [
        float(row[score_key]) - (row["post_id"] * 1e-9)
        for row in rows
    ]
    ranks = rankdata([-score for score in adjusted_scores], method="ordinal")
    return {
        row["post_id"]: int(ranks[idx])
        for idx, row in enumerate(rows)
    }


def percentile_from_rank(rank: int, total: int) -> float:
    """Convert rank to percentile (higher is better)."""
    return ((total - rank) / max(total - 1, 1)) * 100.0
