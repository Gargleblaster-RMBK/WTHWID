"""Loading scraped pairs and turning them into model examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .diffs import compact_diff

PREFIX = "changelog: "
MAX_DIFF_CHARS = 2000
MAX_SUMMARY_CHARS = 200


def model_input(diff: str) -> str:
    return PREFIX + compact_diff(diff, max_chars=MAX_DIFF_CHARS)


def split_of(wiki: str, page_id: int) -> str:
    """Deterministic split by page, so no article leaks across splits."""
    h = int(hashlib.sha1(f"{wiki}:{page_id}".encode()).hexdigest(), 16) % 100
    return "test" if h < 3 else "val" if h < 5 else "train"


def load_splits(paths: list[str]) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["wiki"], rec["rev_id"])
                if key in seen or len(rec["summary"]) > MAX_SUMMARY_CHARS:
                    continue
                seen.add(key)
                src = model_input(rec["diff"])
                if src == PREFIX:
                    continue  # whitespace-only change
                splits[split_of(rec["wiki"], rec["page_id"])].append(
                    {"src": src, "tgt": rec["summary"], "source": rec["source"]}
                )
    return splits
