"""Scrape (diff, edit summary) pairs from Wikimedia full-history dumps.

Wikipedia's API rate-limits bulk access and explicitly asks large consumers to
use dumps, so this streams the MediaWiki Content History exports
(https://dumps.wikimedia.org/other/mediawiki_content_history/) instead. Each
shard is downloaded and decompressed on the fly; nothing large touches disk.

All revision text in these exports is distributed under CC BY-SA (4.0 per
the export readme; older revisions were relicensed to CC BY-SA 3.0 in 2009),
so every record satisfies "CC BY-SA 2.0 or later". Each record keeps a
`source` URL to the exact diff, which is the attribution Wikipedia's own
reuse guidelines accept (a link to the page/revision history).

Usage:
    python -m wthwid.scrape --wiki simplewiki --out data/simplewiki.jsonl
    python -m wthwid.scrape --files local-dump.xml.bz2 --wiki simplewiki ...
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import IO, Iterator
from xml.etree.ElementTree import iterparse

import requests

from .diffs import make_diff
from .summaries import clean_summary

DUMPS = "https://dumps.wikimedia.org/other/mediawiki_content_history"
USER_AGENT = "WTHWID-changelog-scraper/0.1 (https://github.com/gargleblaster-rmbk/wthwid)"
LICENSE = "CC BY-SA 4.0"
# Parent texts kept per page; parents are almost always recent revisions.
TEXT_CACHE = 64


def _http() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def list_shards(wiki: str, date: str = "latest") -> list[str]:
    """URLs of every bzip2 shard of a wiki's content-history export."""
    s = _http()
    if date == "latest":
        index = s.get(f"{DUMPS}/{wiki}/", timeout=60)
        index.raise_for_status()
        dates = sorted(re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', index.text))
        if not dates:
            raise SystemExit(f"no content-history exports found for {wiki}")
        date = dates[-1]
    base = f"{DUMPS}/{wiki}/{date}/xml/bzip2/"
    page = s.get(base, timeout=60)
    page.raise_for_status()
    names = sorted(set(re.findall(r'href="([^"/]+\.xml\.bz2)"', page.text)))
    return [base + n for n in names]


def site_url(wiki: str) -> str:
    """simplewiki -> https://simple.wikipedia.org (good enough for *wiki)."""
    lang = wiki[: -len("wiki")] if wiki.endswith("wiki") else wiki
    return f"https://{lang.replace('_', '-')}.wikipedia.org"


def _open(source: str) -> tuple[IO[bytes], requests.Response | None]:
    if source.startswith(("http://", "https://")):
        resp = _http().get(source, stream=True, timeout=120)
        resp.raise_for_status()
        resp.raw.decode_content = True
        stream: IO[bytes] = resp.raw
    else:
        resp = None
        stream = open(source, "rb")
    if source.endswith(".bz2"):
        stream = bz2.BZ2File(stream)
    return stream, resp


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_revisions(stream: IO[bytes]) -> Iterator[dict]:
    """Yield one dict per revision, with the page's title and namespace.

    Revision elements are cleared as soon as they're read, so memory stays
    flat even on pages with tens of thousands of revisions.
    """
    title, ns, page_id = None, None, None
    rev: dict | None = None
    in_contributor = False
    root = None
    for event, elem in iterparse(stream, events=("start", "end")):
        tag = _local(elem.tag)
        if event == "start":
            if root is None:
                root = elem
            elif tag == "revision":
                rev = {}
            elif tag == "contributor":
                in_contributor = True
            continue
        if tag == "contributor":
            in_contributor = False
        elif rev is None:
            if tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = int(elem.text)
            elif tag == "id":
                page_id = int(elem.text)
            elif tag == "page":
                root.clear()  # drop finished pages so memory stays flat
        elif in_contributor:
            pass
        elif tag in ("id", "parentid"):
            rev[tag] = int(elem.text)
        elif tag == "timestamp":
            rev["timestamp"] = elem.text
        elif tag == "minor":
            rev["minor"] = True
        elif tag == "comment":
            rev["comment"] = None if elem.get("deleted") else elem.text
        elif tag == "model":
            rev["model"] = elem.text
        elif tag == "text":
            rev["text"] = None if elem.get("deleted") else (elem.text or "")
        elif tag == "revision":
            rev.update(title=title, ns=ns, page_id=page_id)
            yield rev
            rev = None
            elem.clear()


def extract_pairs(
    revisions: Iterator[dict],
    wiki: str,
    namespaces: set[int],
    since: str | None,
    max_diff_chars: int,
    keep_bots: bool,
) -> Iterator[dict]:
    site = site_url(wiki)
    cache: OrderedDict[int, str] = OrderedDict()
    current_page = None
    for rev in revisions:
        if rev["page_id"] != current_page:
            cache.clear()
            current_page = rev["page_id"]
        if rev["ns"] not in namespaces or rev.get("model", "wikitext") != "wikitext":
            continue
        text = rev.get("text")
        if text is None:
            continue
        cache[rev["id"]] = text
        cache.move_to_end(rev["id"])
        if len(cache) > TEXT_CACHE:
            cache.popitem(last=False)

        parent = cache.get(rev.get("parentid", -1))
        if parent is None or parent == text:
            continue  # page creation, unknown parent, or null edit
        if since and rev["timestamp"] < since:
            continue
        summary = clean_summary(rev.get("comment"), keep_bots=keep_bots)
        if summary is None:
            continue
        diff = make_diff(parent, text)
        if not diff or len(diff) > max_diff_chars:
            continue
        yield {
            "wiki": wiki,
            "page_id": rev["page_id"],
            "title": rev["title"],
            "rev_id": rev["id"],
            "parent_id": rev["parentid"],
            "timestamp": rev["timestamp"],
            "minor": rev.get("minor", False),
            "diff": diff,
            "summary": summary,
            "raw_comment": rev.get("comment"),
            "source": f"{site}/w/index.php?diff={rev['id']}&oldid={rev['parentid']}",
            "license": LICENSE,
        }


def scrape_shard(job: dict) -> tuple[str, int]:
    src, out = job["source"], Path(job["out"])
    for attempt in range(5):
        written = 0
        try:
            stream, resp = _open(src)
            with stream, out.open("w", encoding="utf-8") as f:
                pairs = extract_pairs(
                    iter_revisions(stream),
                    job["wiki"],
                    job["namespaces"],
                    job["since"],
                    job["max_diff_chars"],
                    job["keep_bots"],
                )
                for rec in pairs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if job["limit"] and written >= job["limit"]:
                        break
            if resp is not None:
                resp.close()
            return src, written
        except (requests.RequestException, OSError, EOFError) as e:
            wait = 2 ** (attempt + 1)
            print(f"[retry {attempt + 1}/5 in {wait}s] {src}: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {src}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wiki", default="simplewiki", help="dbname, e.g. enwiki")
    p.add_argument("--date", default="latest", help="export date (YYYY-MM-DD)")
    p.add_argument("--files", nargs="*", help="local dumps or URLs instead of listing")
    p.add_argument("--shards", type=int, default=0, help="only the first N shards")
    p.add_argument("--out", default="data/pairs.jsonl")
    p.add_argument("--limit-per-shard", type=int, default=0, help="0 = no limit")
    p.add_argument("--namespaces", default="0", help="comma-separated, default articles")
    p.add_argument("--since", help="only revisions at/after this ISO timestamp")
    p.add_argument("--max-diff-chars", type=int, default=20000)
    p.add_argument("--keep-bots", action="store_true")
    p.add_argument(
        "--workers", type=int, default=2,
        help="parallel downloads; dumps.wikimedia.org allows ~2-3 per IP",
    )
    a = p.parse_args(argv)

    sources = a.files or list_shards(a.wiki, a.date)
    if a.shards:
        sources = sources[: a.shards]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [out.with_name(f"{out.stem}.part{i:04d}{out.suffix}") for i in range(len(sources))]
    jobs = [
        dict(
            source=s, out=str(part), wiki=a.wiki, limit=a.limit_per_shard,
            namespaces={int(n) for n in a.namespaces.split(",")}, since=a.since,
            max_diff_chars=a.max_diff_chars, keep_bots=a.keep_bots,
        )
        for s, part in zip(sources, parts)
    ]
    print(f"scraping {len(jobs)} shard(s) with {a.workers} worker(s)", file=sys.stderr)
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for src, n in pool.map(scrape_shard, jobs):
            total += n
            print(f"{n:>9} pairs  {src}", file=sys.stderr)

    with out.open("w", encoding="utf-8") as f:
        for part in parts:
            with part.open(encoding="utf-8") as pf:
                for line in pf:
                    f.write(line)
            part.unlink()
    print(f"wrote {total} pairs to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
