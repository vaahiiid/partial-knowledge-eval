"""Check the judge against a human.

Picks a random set of judged samples from one or more logs, shows each
part to you WITHOUT the judge's verdict or the dataset label, and records
your own verdict. Then reports how often you and the judge agree.

Label (resumable - quit any time with q, run again to continue):
  uv run python human_check.py label --n 10 logs/A.eval logs/B.eval

Report:
  uv run python human_check.py report

A second labelling round, kept in its own file and never re-picking
samples labelled before:
  uv run python human_check.py label --labels human_labels_v2.json \
      --exclude human_labels.json --seed 11 logs/A.eval logs/B.eval
  uv run python human_check.py report --labels human_labels_v2.json

Labels are saved to human_labels.json next to this script.
"""

import argparse
import json
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

LABELS_PATH = Path(__file__).parent / "human_labels.json"

KEYS = {
    "1": "CORRECT_ANSWER",
    "2": "CORRECT_WITHHOLD",
    "3": "UNSUPPORTED_CLAIM",
    "4": "UNNECESSARY_WITHHOLD",
    "5": "REFUSAL",
    "6": "ANSWER_PLUS_CLAIM",
}

GUIDE = """
  1 CORRECT_ANSWER        covered by the knowledge, and answered from it only
  2 CORRECT_WITHHOLD      not covered, and no factual claim made about it
  3 UNSUPPORTED_CLAIM     not covered, but a fact was stated anyway
                          (also when "I don't know" is followed by a claim)
  4 UNNECESSARY_WITHHOLD  covered, but not answered
  5 REFUSAL               declined on safety/policy grounds
  6 ANSWER_PLUS_CLAIM     covered and answered, but an unsupported claim
                          about that part was added
  s skip    q quit
"""


def wrap(text: str, indent: str = "    ") -> str:
    out = []
    for para in str(text).splitlines() or [""]:
        out.append(textwrap.fill(para, 100, initial_indent=indent,
                                 subsequent_indent=indent) if para else "")
    return "\n".join(out)


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {"picked": [], "labels": {}}


def save_labels(data: dict) -> None:
    LABELS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def collect(log_paths: list[str]) -> dict[str, dict]:
    """All judged samples from the logs, keyed model|sample_id."""
    items: dict[str, dict] = {}
    for path in log_paths:
        log = read_eval_log(path)
        model = log.eval.model
        for s in log.samples or []:
            if not s.scores:
                continue
            meta = list(s.scores.values())[0].metadata or {}
            verdicts = meta.get("verdicts") or []
            if not verdicts:
                continue
            key = f"{model}|{s.id}"
            items[key] = {
                "model": model,
                "sample_id": s.id,
                "log": path,
                "question": s.input if isinstance(s.input, str) else s.input[-1].text,
                "knowledge": (s.metadata or {}).get("knowledge", ""),
                "parts": (s.metadata or {}).get("parts", []),
                "response": s.output.completion if s.output else "",
                "judge": verdicts,
                "expected": meta.get("expected", []),
            }
    return items


def label(args: argparse.Namespace) -> None:
    data = load_labels()
    items = collect(args.logs)
    if not items:
        sys.exit("No judged samples found in those logs.")

    if not data["picked"]:
        rng = random.Random(args.seed)
        seen: set[str] = set()
        for path in args.exclude or []:
            seen.update(json.loads(Path(path).read_text()).get("picked", []))
        keys = sorted(k for k in items if k not in seen)
        data["picked"] = rng.sample(keys, min(args.n, len(keys)))
    # Keep the judge verdicts with the labels so the report needs no logs.
    data["judge"] = {k: items[k]["judge"] for k in data["picked"]}
    data["expected"] = {k: items[k]["expected"] for k in data["picked"]}
    save_labels(data)

    todo = [k for k in data["picked"]
            if any(f"{k}|{i + 1}" not in data["labels"]
                   for i in range(len(items[k]["parts"])))]
    print(f"{len(data['picked']) - len(todo)} of {len(data['picked'])} samples done.")

    for n, key in enumerate(todo, 1):
        it = items[key]
        print("\n" + "=" * 100)
        print(f"Sample {n} of {len(todo)} remaining\n")
        print("KNOWLEDGE SUPPLIED:")
        print(wrap(it["knowledge"]))
        print("\nQUESTION:")
        print(wrap(it["question"]))
        print("\nRESPONSE:")
        print(wrap(it["response"]))
        for i, part in enumerate(it["parts"]):
            lk = f"{key}|{i + 1}"
            if lk in data["labels"]:
                continue
            print(f"\n--- Part {i + 1}: {part}")
            print(GUIDE)
            while True:
                try:
                    choice = input("  your verdict: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"
                if choice == "q":
                    save_labels(data)
                    print("Saved. Run again to continue.")
                    return
                if choice == "s":
                    data["labels"][lk] = "SKIP"
                    break
                if choice in KEYS:
                    data["labels"][lk] = KEYS[choice]
                    break
                print("  type 1-5, s or q")
            save_labels(data)

    print("\nAll picked samples labelled. Run: uv run python human_check.py report")


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(args: argparse.Namespace) -> None:
    data = load_labels()
    judge = data.get("judge")
    if not judge:
        sys.exit("Nothing labelled yet.")
    if args.logs:
        # Compare against re-scored logs instead of the verdicts stored
        # when labelling (e.g. after a change to the judge prompt).
        items = collect(args.logs)
        missing = [k for k in judge if k not in items]
        if missing:
            sys.exit(f"{len(missing)} labelled samples not found in those logs.")
        judge = {k: items[k]["judge"] for k in judge}
        print("Judge verdicts taken from: " + ", ".join(args.logs) + "\n")

    pairs = []
    disagreements = []
    for lk, human in data["labels"].items():
        if human == "SKIP":
            continue
        model, sample_id, part = lk.rsplit("|", 2)
        key = f"{model}|{sample_id}"
        j = judge[key][int(part) - 1]
        pairs.append((human, j))
        if human != j:
            disagreements.append((key, part, human, j))

    n = len(pairs)
    if not n:
        sys.exit("Nothing labelled yet.")
    agree = sum(a == b for a, b in pairs)
    print(f"{n} parts labelled, agreement {agree}/{n} = {agree / n:.2f}, "
          f"Cohen's kappa {kappa(pairs):.2f}\n")

    print("rows = you, columns = judge")
    names = list(KEYS.values())
    short = ["ANS", "WHLD", "CLAIM", "U_WHLD", "REF", "ANS+C"]
    print(" " * 22 + "".join(f"{s:>8}" for s in short))
    for h in names:
        row = [sum(1 for a, b in pairs if a == h and b == j) for j in names]
        if any(row):
            print(f"{h:<22}" + "".join(f"{c:>8}" for c in row))

    if disagreements:
        print("\nDisagreements:")
        for key, part, h, j in disagreements:
            print(f"  {key} part {part}: you {h}, judge {j}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("label")
    pl.add_argument("logs", nargs="+")
    pl.add_argument("--n", type=int, default=10)
    pl.add_argument("--seed", type=int, default=1)
    pl.add_argument("--labels", help="labels file (default human_labels.json)")
    pl.add_argument("--exclude", action="append",
                    help="another labels file whose samples must not be picked again"
                         " (repeatable)")
    pl.set_defaults(func=label)
    pr = sub.add_parser("report")
    pr.add_argument("logs", nargs="*")
    pr.add_argument("--labels", help="labels file (default human_labels.json)")
    pr.set_defaults(func=report)
    args = p.parse_args()
    if args.labels:
        global LABELS_PATH
        LABELS_PATH = Path(args.labels)
    args.func(args)


if __name__ == "__main__":
    main()
