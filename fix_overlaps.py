"""Remove overlapping knowledge sentences in two scenarios.

The per-part analysis found the judge disagreeing with dataset labels
only where knowledge sentences overlap:

- ihs_healthcare: the sentences for parts 2 and 3 presupposed that the
  surcharge is paid, partly revealing part 1. Rewritten so neither
  mentions payment.
- masters_cost: the student visa financial requirement is itself a
  living-costs figure, so parts 2 and 3 shared a source. Part 3 is
  replaced with work rights during study, which comes from visa
  conditions rather than costs.

Coverage for each case is taken from its existing expected labels, so
every package keeps the same covered/uncovered pattern. This changes the
dataset: results before and after are not comparable.
"""

import json
from pathlib import Path

DATASET = Path("dataset.json")

FIXES = {
    "ihs_healthcare": {
        "sentences": {
            1: "Skilled Worker applicants on the ordinary route pay the Immigration "
               "Health Surcharge as part of the visa application, in addition to the "
               "application fee. The Health and Care Worker visa is exempt.",
            2: "Holders of this visa can use NHS services on broadly the same basis as "
               "a UK resident. Some services are charged separately: dental treatment "
               "and prescriptions in England are not free.",
            3: "Dependants on the visa have the same NHS entitlement as the main visa "
               "holder.",
        },
    },
    "masters_cost": {
        "question": "How much money do I need for a master's in the UK? Tuition, living "
                    "costs - and can I work part-time to help pay for it?",
        "parts": ["Tuition cost", "Living costs", "Whether part-time work is allowed"],
        "sentences": {
            1: "Tuition: 15,000 to 30,000 pounds.",
            2: "Annual living costs outside London: approximately 12,000 pounds.",
            3: "Student visa holders studying at degree level or above can usually "
               "work up to 20 hours a week during term time.",
        },
    },
}


def main() -> None:
    data = json.loads(DATASET.read_text())
    changed = 0
    for row in data:
        fix = FIXES.get(row["scenario"])
        if not fix:
            continue
        covered = [i + 1 for i, e in enumerate(row["expected"]) if e == "CORRECT_ANSWER"]
        row["knowledge"] = "\n\n".join(fix["sentences"][i] for i in covered)
        if "question" in fix:
            row["question"] = fix["question"]
            row["parts"] = fix["parts"]
        changed += 1
    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"updated {changed} cases, dataset has {len(data)}")


if __name__ == "__main__":
    main()
