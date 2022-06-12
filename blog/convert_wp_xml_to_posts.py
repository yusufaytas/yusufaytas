#!/usr/bin/env python3
"""Convert a WordPress WXR XML export into structured posts.txt format."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {
    "excerpt": "http://wordpress.org/export/1.2/excerpt/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "wfw": "http://wellformedweb.org/CommentAPI/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "wp": "http://wordpress.org/export/1.2/",
}


def clean_content(raw: str) -> str:
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(pre|code|h[1-6]|li|ul|ol|blockquote)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    compact_lines: list[str] = []
    previous_blank = False
    for line in text.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(stripped)
        previous_blank = False

    return "\n".join(compact_lines).strip()


def parse_posts(xml_path: Path) -> list[dict[str, str | list[str]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else []

    posts: list[dict[str, str | list[str]]] = []
    for item in items:
        post_type = (item.findtext("wp:post_type", default="", namespaces=NS) or "").strip()
        status = (item.findtext("wp:status", default="", namespaces=NS) or "").strip()
        if post_type != "post" or status != "publish":
            continue

        raw_content = item.findtext("content:encoded", default="", namespaces=NS) or ""
        content = clean_content(raw_content)
        if not content:
            continue

        categories: list[str] = []
        tags: list[str] = []
        for category in item.findall("category"):
            domain = (category.get("domain") or "").strip()
            value = (category.text or "").strip()
            if not value:
                continue
            if domain == "category":
                categories.append(value)
            elif domain == "post_tag":
                tags.append(value)

        posts.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "date": (item.findtext("wp:post_date", default="", namespaces=NS) or "").strip(),
                "modified": (item.findtext("wp:post_modified", default="", namespaces=NS) or "").strip(),
                "author": (item.findtext("dc:creator", default="", namespaces=NS) or "").strip(),
                "url": (item.findtext("link") or "").strip(),
                "slug": (item.findtext("wp:post_name", default="", namespaces=NS) or "").strip(),
                "categories": categories,
                "tags": tags,
                "content": content,
            }
        )

    return posts


def write_posts(posts: list[dict[str, str | list[str]]], source_xml: Path, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# CHATGPT SOURCE FILE: BLOG POSTS")
    lines.append(f"# Source XML: {source_xml.name}")
    lines.append(f"# Published posts with content: {len(posts)}")
    lines.append("# Notes: Use for tone/topic reference. Do not copy phrases verbatim.")
    lines.append("")

    for idx, post in enumerate(posts, start=1):
        categories = post["categories"]
        tags = post["tags"]

        lines.append(f"<<<POST_START_{idx:04d}>>>")
        lines.append(f"TITLE: {post['title']}")
        lines.append(f"DATE: {post['date']}")
        lines.append(f"LAST_MODIFIED: {post['modified']}")
        lines.append(f"AUTHOR: {post['author']}")
        lines.append(f"URL: {post['url']}")
        lines.append(f"SLUG: {post['slug']}")
        lines.append(f"CATEGORIES: {', '.join(categories) if categories else 'None'}")
        lines.append(f"TAGS: {', '.join(tags) if tags else 'None'}")
        lines.append("CONTENT:")
        lines.append(str(post["content"]))
        lines.append(f"<<<POST_END_{idx:04d}>>>")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WordPress WXR XML export to structured posts text file."
    )
    parser.add_argument("input_xml", type=Path, help="Path to WordPress XML export file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("posts.txt"),
        help="Output file path (default: posts.txt)",
    )
    parser.add_argument(
        "--with-rankings",
        action="store_true",
        help="Also run generate_prose_ratings.py after writing posts file.",
    )
    args = parser.parse_args()

    if not args.input_xml.exists():
        raise SystemExit(f"Input file not found: {args.input_xml}")

    posts = parse_posts(args.input_xml)
    write_posts(posts, args.input_xml, args.output)
    print(f"Wrote {args.output} with {len(posts)} posts.")

    if args.with_rankings:
        ranking_script = Path(__file__).resolve().parent / "generate_prose_ratings.py"
        if not ranking_script.exists():
            raise SystemExit(f"Ranking script not found: {ranking_script}")
        subprocess.run(["python3", str(ranking_script)], check=True)


if __name__ == "__main__":
    main()
