"""Headline rates with 95% intervals, from scored Inspect logs.

The five samples of a case are not independent, so the interval is a
cluster bootstrap over the 30 cases: cases are resampled with
replacement, all of a resampled case's samples come with it, and the
rate is recomputed. Counts are printed beside each rate so the
denominators are visible.

  uv run python experiments/rates_with_ci.py logs/A.eval logs/B.eval
  uv run python experiments/rates_with_ci.py --out rates.json logs/*.eval
"""

import argparse
import json
import random
from collections import defaultdict

from inspect_ai.log import read_eval_log

CORRECT = {"CORRECT_ANSWER", "CORRECT_WITHHOLD"}


def part_rows(path: str) -> tuple[str, dict[str, list[tuple[str, str]]]]:
    """(model, {case id: [(verdict, expected), ...]}) for one log."""
    log = read_eval_log(path)
    by_case: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s in log.samples or []:
        if not s.scores:
            continue
        meta = list(s.scores.values())[0].metadata or {}
        verdicts = meta.get("verdicts") or []
        expected = meta.get("expected") or []
        for v, e in zip(verdicts, expected):
            # With epochs, sample ids carry a "__N" suffix; cluster on the case.
            by_case[str(s.id).split("__")[0]].append((v, e))
    return log.eval.model, dict(by_case)


def counts(rows: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    """Numerator and denominator for each metric, as the scorer defines them."""
    unsup = uncov = unnec = plus = cov = corr = tot = refused = 0
    for v, e in rows:
        if v == "REFUSAL":
            refused += 1
            continue
        tot += 1
        corr += v in CORRECT
        if e == "CORRECT_WITHHOLD":
            uncov += 1
            unsup += v == "UNSUPPORTED_CLAIM"
        elif e == "CORRECT_ANSWER":
            cov += 1
            unnec += v == "UNNECESSARY_WITHHOLD"
            plus += v == "ANSWER_PLUS_CLAIM"
    return {
        "boundary_accuracy": (corr, tot),
        "unsupported_claim_rate": (unsup, uncov),
        "answer_plus_claim_rate": (plus, cov),
        "unnecessary_withhold_rate": (unnec, cov),
        "refusal_rate": (refused, tot + refused),
    }


def bootstrap(by_case: dict, reps: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    cases = sorted(by_case)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        rows = [r for c in rng.choices(cases, k=len(cases)) for r in by_case[c]]
        for name, (n, d) in counts(rows).items():
            if d:
                draws[name].append(n / d)
    out = {}
    for name, xs in draws.items():
        xs.sort()
        out[name] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out")
    args = p.parse_args()

    results = []
    for path in args.logs:
        model, by_case = part_rows(path)
        rows = [r for c in by_case.values() for r in c]
        point = counts(rows)
        ci = bootstrap(by_case, args.reps, args.seed)
        print(f"\n{path}\n  model {model}, {len(by_case)} cases, {len(rows)} parts")
        entry = {"log": path, "model": model, "cases": len(by_case), "metrics": {}}
        for name, (n, d) in point.items():
            rate = n / d if d else float("nan")
            lo, hi = ci.get(name, (float("nan"), float("nan")))
            print(f"  {name:27s} {rate:.3f}  [{lo:.3f}, {hi:.3f}]  ({n}/{d})")
            entry["metrics"][name] = {"rate": rate, "ci95": [lo, hi], "n": n, "d": d}
        results.append(entry)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
