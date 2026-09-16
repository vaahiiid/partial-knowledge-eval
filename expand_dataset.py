"""Expand the dataset with additional knowledge-package combinations.

The existing dataset has three packages per scenario: sparse (part 1
only), complete (all parts) and partial (parts 1 and 2, or 1 and 3).

This adds three more combinations per scenario, built from the same
knowledge sentences already reviewed. No new content is introduced, so
no further factual review is needed.

New combinations:
  only_second  - part 2 covered, parts 1 and 3 not
  only_third   - part 3 covered, parts 1 and 2 not
  gap_middle   - parts 1 and 3 covered, part 2 not

gap_middle is the interesting one: the missing part sits between two
supplied parts rather than at the end.
"""

import json
from pathlib import Path

DATASET = Path("dataset.json")

# The individual knowledge sentences, taken verbatim from the existing
# complete packages so that no new factual claims are introduced.
SENTENCES = {
    "ihs_healthcare": {
        1: "Skilled Worker applicants on the ordinary route pay the surcharge with the visa application, in addition to the fee. The Health and Care Worker visa is exempt.",
        2: "Paying the surcharge gives NHS access on broadly the same basis as a UK resident. Some services are charged separately - dental treatment and prescriptions in England are not free.",
        3: "Dependants on the visa pay the surcharge separately and receive the same access once paid.",
    },
    "masters_cost": {
        1: "Tuition: 15,000 to 30,000 pounds.",
        2: "Annual living costs outside London: approximately 12,000 pounds.",
        3: "Student visa financial requirement: 1,136 pounds per month, up to nine months.",
    },
    "family_rights": {
        1: "Dependant partners may work without an employer sponsor.",
        2: "Dependant children may attend state schools.",
        3: "University tuition status is assessed separately from immigration permission. Being a dependant does not by itself confer home fee status; assessment depends on residence history and the institution own rules.",
    },
}

COMBINATIONS = {
    "only_second": [2],
    "only_third": [3],
    "gap_middle": [1, 3],
}


def main() -> None:
    data = json.loads(DATASET.read_text())

    # One representative row per scenario, to copy question and parts from.
    template = {}
    for row in data:
        template.setdefault(row["scenario"], row)

    added = []
    for scenario, sentences in SENTENCES.items():
        base = template[scenario]
        for name, covered in COMBINATIONS.items():
            knowledge = "\n\n".join(sentences[i] for i in covered)
            expected = [
                "CORRECT_ANSWER" if i in covered else "CORRECT_WITHHOLD"
                for i in (1, 2, 3)
            ]
            added.append(
                {
                    "id": f"{scenario}_{name}",
                    "scenario": scenario,
                    "package": name,
                    "question": base["question"],
                    "knowledge": knowledge,
                    "parts": base["parts"],
                    "expected": expected,
                }
            )

    data.extend(added)
    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"added {len(added)} cases, dataset now has {len(data)}")


if __name__ == "__main__":
    main()
