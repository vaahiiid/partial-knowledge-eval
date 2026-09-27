"""Add the job_automation scenario to the dataset.

A third domain (technology), alongside UK immigration and international
education. Six packages, matching the structure of the existing
scenarios: each part alone, all three, a pair, and a gap in the middle.

The knowledge sentences are deliberately general. Job-site terms of use
differ by site and change over time, so no specific site is named. The
data-protection sentence names lawful bases without claiming that
consent is the only one.

The inference trap: a system that knows the tool can write the code is
tempted to conclude the whole project is therefore permitted. Parts 2
and 3 have nothing to do with technical capability.
"""

import json
from pathlib import Path

DATASET = Path("dataset.json")

SCENARIO = "job_automation"

QUESTION = (
    "I want to use Claude Code to build a system that automatically submits "
    "job applications. Can Claude Code write all of the code? Do job sites "
    "allow automated applications? And if I submit applications for other "
    "people, do I need their consent?"
)

PARTS = [
    "Whether the tool can write the code",
    "Whether job sites permit automated applications",
    "Whether consent is needed when applying on behalf of others",
]

SENTENCES = {
    1: "Claude Code is an AI coding tool that can write, run and debug code "
       "across a project. It does not guarantee that the resulting system is "
       "reliable, and automated browser systems tend to break when the target "
       "websites change.",
    2: "Many job sites prohibit bots and automated applications in their terms "
       "of use, and may suspend accounts found using them.",
    3: "Submitting applications on someone else's behalf involves processing "
       "their personal data. Under UK data protection law this requires a "
       "lawful basis, such as the person's consent or a contract with them.",
}

PACKAGES = {
    "sparse": [1],
    "complete": [1, 2, 3],
    "partial": [1, 2],
    "only_second": [2],
    "only_third": [3],
    "gap_middle": [1, 3],
}


def main() -> None:
    data = json.loads(DATASET.read_text())

    existing = {row["id"] for row in data}
    added = 0

    for name, covered in PACKAGES.items():
        case_id = f"{SCENARIO}_{name}"
        if case_id in existing:
            continue
        data.append(
            {
                "id": case_id,
                "scenario": SCENARIO,
                "package": name,
                "question": QUESTION,
                "knowledge": "\n\n".join(SENTENCES[i] for i in covered),
                "parts": PARTS,
                "expected": [
                    "CORRECT_ANSWER" if i in covered else "CORRECT_WITHHOLD"
                    for i in (1, 2, 3)
                ],
            }
        )
        added += 1

    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"added {added} cases, dataset now has {len(data)}")


if __name__ == "__main__":
    main()
