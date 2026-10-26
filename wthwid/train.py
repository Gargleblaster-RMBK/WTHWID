"""Fine-tune a small seq2seq model to map diffs to changelog lines.

CodeT5 is the base because its byte-level BPE covers every character in
wikitext and source diffs (T5's SentencePiece vocab can't even encode "{").

Usage:
    python -m wthwid.train data/simplewiki.jsonl --out models/wthwid-small
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_splits

BASE_MODEL = "Salesforce/codet5-small"


def batches(examples, size, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        yield [examples[j] for j in order[i : i + size]]


def encode(tok, batch, max_src, max_tgt):
    enc = tok([e["src"] for e in batch], max_length=max_src, truncation=True,
              padding=True, return_tensors="pt")
    labels = tok(text_target=[e["tgt"] for e in batch], max_length=max_tgt,
                 truncation=True, padding=True, return_tensors="pt").input_ids
    labels[labels == tok.pad_token_id] = -100
    enc["labels"] = labels
    return enc


@torch.no_grad()
def val_loss(model, tok, examples, args):
    model.eval()
    total, n = 0.0, 0
    for batch in batches(examples, args.batch_size, shuffle=False):
        enc = encode(tok, batch, args.max_src, args.max_tgt)
        total += model(**enc).loss.item() * len(batch)
        n += len(batch)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def generate(model, tok, srcs, max_src, max_tgt, beams=4, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(srcs), batch_size):
        enc = tok(srcs[i : i + batch_size], max_length=max_src, truncation=True,
                  padding=True, return_tensors="pt")
        ids = model.generate(**enc, max_new_tokens=max_tgt, num_beams=beams,
                             no_repeat_ngram_size=3)
        out.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return out


def rouge(preds, refs):
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = Counter()
    for p, r in zip(preds, refs):
        for k, v in sc.score(r, p).items():
            agg[k] += v.fmeasure
    res = {k: round(100 * v / len(refs), 2) for k, v in agg.items()}
    res["exact"] = round(100 * sum(p.strip().lower() == r.strip().lower()
                                   for p, r in zip(preds, refs)) / len(refs), 2)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("data", nargs="+", help="JSONL files from wthwid.scrape")
    p.add_argument("--out", default="models/wthwid-small")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--max-train", type=int, default=0, help="0 = all")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-src", type=int, default=256)
    p.add_argument("--max-tgt", type=int, default=48)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)

    splits = load_splits(a.data)
    for name in splits:
        random.Random(a.seed).shuffle(splits[name])
    train = splits["train"][: a.max_train or None]
    val = splits["val"][: a.val_size]
    test = splits["test"][: a.test_size]
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / a.steps)
    )
    out = Path(a.out)
    best = float("inf")
    step, t0 = 0, time.time()
    model.train()
    while step < a.steps:
        for batch in batches(train, a.batch_size, shuffle=True):
            loss = model(**encode(tok, batch, a.max_src, a.max_tgt)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                vl = val_loss(model, tok, val, a)
                print(f"step {step} val_loss {vl:.4f}", flush=True)
                if vl < best:
                    best = vl
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
            if step >= a.steps:
                break

    # Final report on held-out pages, using the best checkpoint.
    model = AutoModelForSeq2SeqLM.from_pretrained(out)
    refs = [e["tgt"] for e in test]
    preds = generate(model, tok, [e["src"] for e in test], a.max_src, a.max_tgt)
    most_common = Counter(e["tgt"] for e in train).most_common(1)[0][0]
    report = {
        "base_model": a.base,
        "train_examples": len(train),
        "steps": a.steps,
        "best_val_loss": round(best, 4),
        "test_examples": len(test),
        "model": rouge(preds, refs),
        "baseline_most_common": {"summary": most_common,
                                 **rouge([most_common] * len(refs), refs)},
    }
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    with (out / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for e, pred in zip(test, preds):
            f.write(json.dumps({"pred": pred, "ref": e["tgt"],
                                "source": e["source"]}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
