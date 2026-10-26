"""Diff generation and normalisation.

`make_diff` produces a standard unified diff (what `diff -U0` / `git diff -U0`
would print, minus file headers). `compact_diff` turns *any* unified diff into
the compact word-level form the model actually reads. Training and inference
both go through `compact_diff`, so a diff pasted from git gets the same
treatment as one built from Wikipedia revisions.
"""

from __future__ import annotations

import difflib
import re

HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

# Words of unchanged context kept either side of a word-level change.
CONTEXT_WORDS = 6
# Longest a single added/removed line may be in the compact form.
MAX_LINE_CHARS = 300


def make_diff(old: str, new: str) -> str:
    """Line-level unified diff with zero context and no file headers."""
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", n=0
    )
    # The first two lines are the empty "--- " / "+++ " headers.
    return "\n".join(list(lines)[2:])


def _clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _word_diff(old: str, new: str) -> str:
    """git --word-diff style rendering of one changed line, context-trimmed."""
    a, b = old.split(), new.split()
    out: list[str] = []
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for i, (tag, a0, a1, b0, b1) in enumerate(ops):
        if tag == "equal":
            words = a[a0:a1]
            first, last = i == 0, i == len(ops) - 1
            if len(words) > CONTEXT_WORDS * 2 or (
                (first or last) and len(words) > CONTEXT_WORDS
            ):
                head = [] if first else words[:CONTEXT_WORDS]
                tail = [] if last else words[-CONTEXT_WORDS:]
                words = head + ["…"] + tail
            out.extend(words)
            continue
        if a1 > a0:
            out.append("[-" + " ".join(a[a0:a1]) + "-]")
        if b1 > b0:
            out.append("{+" + " ".join(b[b0:b1]) + "+}")
    return _clip(" ".join(out), MAX_LINE_CHARS * 2)


def _render_hunk(removed: list[str], added: list[str]) -> list[str]:
    out = []
    paired = min(len(removed), len(added))
    for old, new in zip(removed[:paired], added[:paired]):
        out.append("~ " + _word_diff(old, new))
    out.extend("- " + _clip(line) for line in removed[paired:])
    out.extend("+ " + _clip(line) for line in added[paired:])
    return out


def compact_diff(diff: str, max_chars: int = 4000) -> str:
    """Normalise a unified diff into the model's input format.

    Hunk bodies are consumed using the line counts in each `@@` header, so a
    removed line that happens to start with "--" is never mistaken for a
    `---` file header. Anything outside hunks (git headers, file names) is
    dropped. Changed line pairs become word diffs: `[-old-]{+new+}`.
    """
    lines = diff.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = HUNK_RE.match(lines[i])
        i += 1
        if not m:
            continue
        n_old = int(m.group(1)) if m.group(1) is not None else 1
        n_new = int(m.group(2)) if m.group(2) is not None else 1
        removed: list[str] = []
        added: list[str] = []
        while i < len(lines) and (n_old > 0 or n_new > 0):
            line = lines[i]
            i += 1
            tag, body = line[:1], line[1:]
            if tag == "\\":  # "\ No newline at end of file"
                continue
            if tag == "-":
                removed.append(body)
                n_old -= 1
            elif tag == "+":
                added.append(body)
                n_new -= 1
            else:  # context line (" " or an empty line from stripped space)
                out.extend(_render_hunk(removed, added))
                removed, added = [], []
                n_old -= 1
                n_new -= 1
        out.extend(_render_hunk(removed, added))
        out.append("")
    text = "\n".join(out).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
