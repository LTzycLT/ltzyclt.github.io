#!/usr/bin/env python3
"""
Normalize heading levels in Jekyll _posts/*.md files.

Rule: promote all headings so the minimum heading level becomes h1 (#),
and cap anything beyond h3 (###) at h3.

Example: if a post uses ##, ###, ####
  -> #, ##, ###
"""

import re
import sys
from pathlib import Path


def normalize_file(filepath: Path, dry_run: bool = False) -> bool:
    """Normalize headings in a single file. Returns True if changes were made."""
    text = filepath.read_text(encoding="utf-8")

    # Split off YAML front matter (--- ... ---)
    front_matter = ""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front_matter = text[: end + 4]  # include closing ---
            body = text[end + 4 :]

    # Find all heading lines and their levels
    heading_pattern = re.compile(r"^(#{1,6}) ", re.MULTILINE)
    matches = heading_pattern.findall(body)
    if not matches:
        return False

    min_level = min(len(m) for m in matches)
    if min_level == 1:
        # Already starts at h1; just cap anything beyond h3
        shift = 0
    else:
        shift = min_level - 1  # promote so min becomes h1

    def replace_heading(m):
        hashes = m.group(1)
        new_level = max(1, min(3, len(hashes) - shift))
        return "#" * new_level + " "

    new_body = heading_pattern.sub(replace_heading, body)

    if new_body == body:
        return False

    if not dry_run:
        filepath.write_text(front_matter + new_body, encoding="utf-8")
        print(f"  Updated: {filepath.name}")
    else:
        print(f"  Would update: {filepath.name}")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    posts_dir = Path(__file__).parent.parent / "_posts"

    if not posts_dir.exists():
        print(f"Error: {posts_dir} not found")
        sys.exit(1)

    files = sorted(posts_dir.glob("*.md"))
    changed = 0
    for f in files:
        if normalize_file(f, dry_run=dry_run):
            changed += 1

    print(f"\n{'Would change' if dry_run else 'Changed'} {changed}/{len(files)} files.")


if __name__ == "__main__":
    main()
