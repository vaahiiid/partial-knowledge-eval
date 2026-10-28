"""Show random examples of one verdict, to check the judge by eye.

Usage:
  uv run python show_verdict.py ANSWER_PLUS_CLAIM 6 logs/rescore/A.eval logs/rescore/B.eval
"""

import random
import sys

from inspect_ai.log import read_eval_log

verdict, n, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
hits = []
for path in paths:
    log = read_eval_log(path)
    for s in log.samples or []:
        if not s.scores:
            continue
        sc = list(s.scores.values())[0]
        for i, v in enumerate((sc.metadata or {}).get("verdicts", [])):
            if v == verdict:
                hits.append((log.eval.model, s, sc, i))

print(f"{len(hits)} parts with {verdict}; showing {min(n, len(hits))}\n")
for model, s, sc, i in random.Random(3).sample(hits, min(n, len(hits))):
    parts = (s.metadata or {}).get("parts", [])
    print("=" * 90)
    print(f"{model} | {s.id} | part {i + 1}: {parts[i] if i < len(parts) else ''}")
    print(f"\nKNOWLEDGE: {(s.metadata or {}).get('knowledge', '')}")
    print(f"\nRESPONSE: {s.output.completion[:1200] if s.output else ''}")
    print(f"\nJUDGE (end): ...{(sc.explanation or '')[-900:]}")
    print()
