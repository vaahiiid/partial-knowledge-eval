"""Compare one case across saved runs, to locate where results change.

When the same case gets different verdicts on different runs, the
change came from one of two places: the model under test answered
differently, or the judge classified the same answer differently. This
groups every run of one case by its verdict pattern and prints one
example response per pattern, so the two can be told apart by eye.

Usage:
  uv run python compare_runs.py <case_id> 'logs/2026-09-30*.eval' 'logs/2026-10-06*.eval'

Any number of log patterns. Case ids are the `id` field in dataset.json.
"""

import glob
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    case_id = sys.argv[1]
    files = sorted({f for pattern in sys.argv[2:] for f in glob.glob(pattern)})
    if not files:
        print("No logs match those patterns")
        sys.exit(1)

    groups: dict[tuple, list] = defaultdict(list)
    for f in files:
        log = read_eval_log(f)
        for s in log.samples or []:
            if (s.id != case_id and not str(s.id).startswith(f"{case_id}__")) or not s.scores:
                continue
            score = list(s.scores.values())[0]
            verdicts = tuple((score.metadata or {}).get("verdicts", []))
            date = log.eval.created[:16] if log.eval and log.eval.created else f
            groups[verdicts].append((date, s.output.completion if s.output else ""))

    if not groups:
        print(f"Case {case_id!r} not found in {len(files)} logs")
        sys.exit(1)

    print(f"{case_id}: {sum(len(v) for v in groups.values())} runs in {len(files)} logs\n")
    for verdicts, runs in sorted(groups.items(), key=lambda kv: kv[1][0][0]):
        dates = sorted(d for d, _ in runs)
        print("=" * 72)
        print(f"verdicts: {', '.join(verdicts) or '(unparsed)'}")
        print(f"{len(runs)} runs, {dates[0]} to {dates[-1]}")
        print("-" * 72)
        print("example response:\n")
        print(runs[0][1].strip()[:1500])
        print()


if __name__ == "__main__":
    main()
