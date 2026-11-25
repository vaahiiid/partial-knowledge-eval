"""Re-parse the judge's saved replies in a scored log, without new calls.

Use after a change to parse_verdicts: each sample's last model call is
the judge's reply, so its verdicts can be read again and the score
rebuilt. Writes <log>.reparsed.eval next to the input and prints how
many samples changed and how many still fail to parse.

  uv run python experiments/reparse_log.py logs/<run>/<log>.eval
"""

import sys

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import Score

sys.path.insert(0, ".")
from partial_knowledge import parse_verdicts  # noqa: E402


def reparse(path: str) -> str:
    log = read_eval_log(path)
    changed = failed = 0
    for s in log.samples or []:
        name, old = next(iter(s.scores.items()))
        expected = (old.metadata or {}).get("expected") or []
        calls = [e for e in s.events if e.event == "model"]
        reply = calls[-1].output.completion if calls else ""
        verdicts = parse_verdicts(reply, len(expected))
        if verdicts is None:
            failed += 1
            continue
        if verdicts == (old.metadata or {}).get("verdicts"):
            continue
        judged = [v for v in verdicts if v != "REFUSAL"]
        correct = sum(v in ("CORRECT_ANSWER", "CORRECT_WITHHOLD") for v in judged)
        s.scores[name] = Score(
            value=correct / len(judged) if judged else 0.0,
            answer=", ".join(verdicts),
            explanation=reply,
            metadata={"verdicts": verdicts, "expected": expected},
        )
        changed += 1
    out = path[: -len(".eval")] + ".reparsed.eval"
    write_eval_log(log, out)
    print(f"{path}: {changed} samples re-parsed, {failed} still unparsed -> {out}")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        reparse(p)
