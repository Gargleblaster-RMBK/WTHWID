# WTHWID
What the hell was i doing?

Scrapes Wikipedia edit history into `(diff, edit summary)` pairs, then fine-tunes a
small model that reads a diff and writes a one-line changelog.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch  # or a CUDA build
.venv/bin/pip install -r requirements.txt
```

## 1. Scrape

```sh
.venv/bin/python -m wthwid.scrape --wiki simplewiki --out data/simplewiki.jsonl
.venv/bin/python -m wthwid.scrape --wiki enwiki --shards 10 --limit-per-shard 200000 --out data/enwiki.jsonl
```

- Streams the [MediaWiki Content History exports](https://dumps.wikimedia.org/other/mediawiki_content_history/)
  and decompresses on the fly; nothing large is written except the output JSONL.
  The live API isn't used: it rate-limits bulk access and Wikimedia asks bulk
  users to use dumps.
- "All of Wikipedia" is every `--wiki` (`enwiki`, `dewiki`, …). enwiki alone is
  ~3,400 shards and many TB of compressed history, so plan for days, not minutes.
- Keeps article-namespace edits that have a human-written summary. Drops page
  creations, null edits, reverts/undos, bot edits, MediaWiki auto-summaries and
  summaries that name users. `/* Section */` prefixes are stripped.
- **License:** all dump text is CC BY-SA (4.0 per the export; pre-2009 text was
  relicensed to 3.0), so every record meets "CC BY-SA 2.0 or later". Each record
  has a `source` link to its exact diff, which is how Wikipedia's reuse guidance
  says to attribute. Usernames and IPs are not stored. A model or dataset you
  redistribute should carry CC BY-SA attribution for the training data.

Record format:

```json
{"wiki": "simplewiki", "page_id": 1, "title": "April", "rev_id": 5715, "parent_id": 5043,
 "timestamp": "...", "minor": false, "diff": "@@ -3 +3 @@\n-...\n+...", "summary": "fix typo",
 "raw_comment": "/* Intro */ fix typo", "source": "https://simple.wikipedia.org/w/index.php?diff=5715&oldid=5043",
 "license": "CC BY-SA 4.0"}
```

## 2. Train

```sh
.venv/bin/python -m wthwid.train data/simplewiki.jsonl --out models/wthwid-small
```

Fine-tunes `Salesforce/codet5-small` (60M params; its byte-level tokenizer handles
wikitext and code). Splits are by page, so test articles are never seen in
training. Writes `eval.json` (ROUGE + exact match vs. a most-common-summary
baseline) and `test_predictions.jsonl` next to the model.

## 3. Use

```sh
git diff -U0 | .venv/bin/python -m wthwid.predict --model models/wthwid-small
```

Any unified diff works. It's normalised to a compact word-level form
(`[-old-] {+new+}`) by `wthwid/diffs.py`, the same code used on training data.

## Tests

```sh
.venv/bin/python -m pytest -q
```
