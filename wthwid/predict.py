"""Generate a changelog line from a unified diff.

    git diff -U0 | python -m wthwid.predict
    python -m wthwid.predict --model models/wthwid-small change.diff
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import model_input
from .train import generate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("diff", nargs="?", help="diff file (default: stdin)")
    p.add_argument("--model", default="models/wthwid-small")
    p.add_argument("--beams", type=int, default=4)
    a = p.parse_args(argv)

    diff = open(a.diff, encoding="utf-8").read() if a.diff else sys.stdin.read()
    src = model_input(diff)
    if src.strip() == model_input("").strip():
        sys.exit("no hunks found; pass a unified diff (e.g. `git diff`)")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    print(generate(model, tok, [src], 256, 48, beams=a.beams)[0])


if __name__ == "__main__":
    main()
