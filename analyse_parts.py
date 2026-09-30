"""Per-part breakdown of how uncovered parts were handled.

The headline metrics average across parts and scenarios. This shows, for
every part that was NOT covered by the knowledge package, how often the
system correctly withheld it - broken down by scenario, part and package.

Usage:
  uv run python analyse_parts.py 'logs/2026-09-30*.eval'

A log pattern is required. Older logs may come from earlier dataset
versions or earlier code (before the knowledge package was supplied, or
before the parser fallback), and mixing them would distort the result.
"""

import glob
import sys
from collections import Counter

from inspect_ai.log import read_eval_log


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = sorted(glob.glob(sys.argv[1]))
    if not files:
        print(f"No logs match {sys.argv[1]!r}")
        sys.exit(1)

    counts: dict[tuple, Counter] = {}
    part_names: dict[str, list[str]] = {}

    for f in files:
        log = read_eval_log(f)
        for s in log.samples or []:
            meta = s.metadata or {}
            scenario = meta.get("scenario")
            if not scenario or not s.scores:
                continue
            part_names.setdefault(scenario, meta.get("parts", []))
            score_meta = list(s.scores.values())[0].metadata or {}
            pairs = zip(score_meta.get("verdicts", []), score_meta.get("expected", []))
            for i, (verdict, expected) in enumerate(pairs):
                if expected != "CORRECT_WITHHOLD":
                    continue
                key = (scenario, i + 1, meta.get("package", "?"))
                counts.setdefault(key, Counter())[verdict] += 1

    print(f"{len(files)} logs\n")

    def summary(c: Counter) -> str:
        t = sum(c.values())
        w = c["CORRECT_WITHHOLD"]
        claim = c["UNSUPPORTED_CLAIM"]
        other = t - w - claim
        text = f"withheld {w}  claimed {claim}"
        if other:
            detail = ", ".join(
                f"{v} {n}" for v, n in c.items()
                if v not in ("CORRECT_WITHHOLD", "UNSUPPORTED_CLAIM")
            )
            text += f"  OTHER {other} ({detail})"
        return f"{text}   of {t}"

    scenarios = sorted({k[0] for k in counts})
    for scenario in scenarios:
        print(scenario)
        parts = sorted({k[1] for k in counts if k[0] == scenario})
        for part in parts:
            keys = sorted(k for k in counts if k[0] == scenario and k[1] == part)
            merged: Counter = Counter()
            for k in keys:
                merged.update(counts[k])
            names = part_names.get(scenario, [])
            name = names[part - 1] if part - 1 < len(names) else ""
            print(f"  part {part}  {name}")
            print(f"      all          {summary(merged)}")
            for k in keys:
                print(f"      {k[2]:<12} {summary(counts[k])}")
        print()


if __name__ == "__main__":
    main()
