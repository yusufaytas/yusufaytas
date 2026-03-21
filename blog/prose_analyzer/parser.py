"""Parse WordPress export format."""

import json
import re
from pathlib import Path
from .models import Post


def parse_posts_txt(text: str) -> list[Post]:
    """Parse posts from WordPress export text format."""
    blocks = re.findall(
        r"<<<POST_START_(\d{4})>>>\n(.*?)\n<<<POST_END_\1>>>",
        text,
        re.S
    )
    
    posts = []
    for post_id, block in blocks:
        def field(name: str) -> str:
            m = re.search(rf"^{re.escape(name)}:\s*(.*)$", block, re.M)
            return m.group(1).strip() if m else ""

        cm = re.search(r"^CONTENT:\n(.*)$", block, re.S | re.M)
        content = cm.group(1).strip() if cm else ""

        posts.append(
            Post(
                post_id=int(post_id),
                title=field("TITLE"),
                date=field("DATE"),
                url=field("URL"),
                slug=field("SLUG"),
                categories=field("CATEGORIES"),
                tags=field("TAGS"),
                content=content,
            )
        )
    
    return posts


def parse_posts_jsonl(text: str) -> list[Post]:
    """Parse posts from JSONL format (one JSON object per line)."""
    posts = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        data = json.loads(line)
        posts.append(
            Post(
                post_id=data["id"],
                title=data["title"],
                date=data["date"],
                url=data["url"],
                slug=data["slug"],
                categories=", ".join(data.get("categories", [])) or "None",
                tags=", ".join(data.get("tags", [])) or "None",
                content=data["content"],
            )
        )
    return posts


def parse_posts_json(text: str) -> list[Post]:
    """Parse posts from JSON format."""
    data = json.loads(text)
    posts_data = data.get("posts", data) if isinstance(data, dict) else data
    
    posts = []
    for item in posts_data:
        posts.append(
            Post(
                post_id=item["id"],
                title=item["title"],
                date=item["date"],
                url=item["url"],
                slug=item["slug"],
                categories=", ".join(item.get("categories", [])) or "None",
                tags=", ".join(item.get("tags", [])) or "None",
                content=item["content"],
            )
        )
    return posts


def load_posts(filepath: Path) -> list[Post]:
    """Load posts from file, auto-detecting format."""
    text = filepath.read_text(encoding="utf-8")
    
    # Auto-detect format
    if filepath.suffix == ".jsonl":
        return parse_posts_jsonl(text)
    elif filepath.suffix == ".json":
        return parse_posts_json(text)
    elif text.strip().startswith("{"):
        # Try JSON first
        try:
            return parse_posts_json(text)
        except json.JSONDecodeError:
            pass
        # Try JSONL
        try:
            return parse_posts_jsonl(text)
        except json.JSONDecodeError:
            pass
    
    # Default to text format
    return parse_posts_txt(text)
