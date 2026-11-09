"""Compare RAGAS faithfulness with the hand labels.

RAGAS faithfulness splits a response into atomic statements and checks each
against the supplied context; the score is the share of statements the
context supports. This script runs it on the samples labelled by hand with
human_check.py and asks a simple question at the sample level: when the
human found an unsupported or added claim in a response, does RAGAS score
it below 1, and does it name the same claim?

Setup (not part of the main dependencies):
  uv pip install ragas anthropic instructor "langchain-community<0.4"

Run from the repository root, passing the labels files and the logs the
labelled samples came from (re-scored logs give the current judge verdicts):
  uv run python experiments/ragas_check.py \
      --labels human_labels.json --labels human_labels_v2.json \
      logs/rescore/A.eval logs/rescore/B.eval ...

Uses claude-haiku-4-5 for RAGAS by default (--model to change), so the
comparison is between methods, not between judge models.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from human_check import collect  # noqa: E402

FLAGS = {"UNSUPPORTED_CLAIM", "ANSWER_PLUS_CLAIM"}


def load_labels(paths: list[str]) -> dict[str, dict[int, str]]:
    """{sample_key: {part_number: human verdict}}"""
    out: dict[str, dict[int, str]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for lk, verdict in data["labels"].items():
            key, part = lk.rsplit("|", 1)
            out.setdefault(key, {})[int(part)] = verdict
    return out


async def run(args: argparse.Namespace) -> None:
    from anthropic import AsyncAnthropic
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import Faithfulness

    llm = llm_factory(args.model, provider="anthropic", client=AsyncAnthropic(),
                      max_tokens=2048)
    # Recent versions of the anthropic SDK no longer accept sampling
    # parameters, but RAGAS sends temperature and top_p by default.
    for arg in ("temperature", "top_p"):
        llm.model_args.pop(arg, None)
    metric = Faithfulness(llm=llm)

    labels = load_labels(args.labels)
    items = collect(args.logs)
    missing = [k for k in labels if k not in items]
    if missing:
        sys.exit(f"{len(missing)} labelled samples not found in those logs, e.g. {missing[0]}")

    rows = []
    for key, parts in labels.items():
        it = items[key]
        human = [parts.get(i + 1, "SKIP") for i in range(len(it["parts"]))]
        if all(h in ("REFUSAL", "SKIP") for h in human):
            continue
        statements = await metric._create_statements(it["question"], it["response"])
        verdicts = await metric._create_verdicts(statements, it["knowledge"])
        score = metric._compute_score(verdicts)
        unsupported = [s.statement for s in verdicts.statements if not s.verdict]
        rows.append({
            "key": key,
            "human_flag": any(h in FLAGS for h in human),
            "judge_flag": any(j in FLAGS for j in it["judge"]),
            "ragas": score,
            "unsupported": unsupported,
            "human": human,
        })
        print(f"{key}: ragas={score:.2f} human={human} judge={it['judge']}")
        for s in unsupported:
            print(f"    not supported: {s}")

    def agree(flag_name: str, pred) -> str:
        pairs = [(r["human_flag"], pred(r)) for r in rows]
        tp = sum(h and p for h, p in pairs)
        fp = sum(p and not h for h, p in pairs)
        fn = sum(h and not p for h, p in pairs)
        tn = sum(not h and not p for h, p in pairs)
        return (f"{flag_name:<26} agrees {tp + tn}/{len(pairs)}  "
                f"(both flag {tp}, only it flags {fp}, only human flags {fn})")

    print(f"\n{len(rows)} samples (all-refusal samples left out)")
    print("Sample flagged = at least one part with an unsupported or added claim.")
    print(agree("LLM judge (per-part)", lambda r: r["judge_flag"]))
    print(agree("RAGAS faithfulness < 1", lambda r: r["ragas"] < 1))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"Details saved to {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--labels", action="append", required=True)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--out", default="ragas_check.json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
