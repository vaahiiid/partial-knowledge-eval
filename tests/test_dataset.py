"""Structural checks on dataset.json and human_labels.json.

These guard the design rules in the README: every scenario asks the same
question across six packages, each package covers a distinct set of
parts, and a package's knowledge is exactly the sentences for the parts
it covers.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from partial_knowledge import VALID_VERDICTS

ROOT = Path(__file__).parent.parent
DATASET = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
HUMAN = json.loads((ROOT / "human_labels.json").read_text(encoding="utf-8"))

FIELDS = {"id", "scenario", "package", "question", "knowledge", "parts", "expected"}
PACKAGES_PER_SCENARIO = 6

BY_ID = {r["id"]: r for r in DATASET}
BY_SCENARIO: defaultdict[str, list[dict]] = defaultdict(list)
for _r in DATASET:
    BY_SCENARIO[_r["scenario"]].append(_r)
SCENARIOS = sorted(BY_SCENARIO)

# Known dataset problems, left unfixed because changing the dataset makes
# earlier results incomparable. strict=True, so a fix turns these red
# until the entry is removed here.
KNOWN_KNOWLEDGE_DRIFT = {
    # Partner sentence differs from family_complete ("of Skilled Worker
    # visa holders ... in the UK").
    "family_sparse",
    # "the institution own rules" vs "the institution's own rules".
    "family_rights_only_third",
    "family_rights_gap_middle",
}
KNOWN_DUPLICATE_COVERAGE = {
    # ihs_partial and ihs_healthcare_gap_middle both cover parts 1 and 3
    # with identical knowledge; the "one pair" package (parts 1 and 2) is missing.
    "ihs_healthcare",
}


def known(ids: set[str], reason: str):
    return lambda x: pytest.param(
        x, marks=pytest.mark.xfail(strict=True, reason=reason) if x in ids else ()
    )


def test_records_have_expected_fields():
    for r in DATASET:
        assert set(r) == FIELDS, r.get("id")


def test_ids_are_unique():
    dupes = [i for i, n in Counter(r["id"] for r in DATASET).items() if n > 1]
    assert not dupes


@pytest.mark.parametrize("record", DATASET, ids=lambda r: r["id"])
def test_record_shape(record):
    assert record["question"].strip()
    assert record["knowledge"].strip()
    assert len(record["parts"]) == len(record["expected"]) >= 2
    assert set(record["expected"]) <= {"CORRECT_ANSWER", "CORRECT_WITHHOLD"}
    assert "CORRECT_ANSWER" in record["expected"], "package covers nothing"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_has_all_packages(scenario):
    rows = BY_SCENARIO[scenario]
    assert len(rows) == PACKAGES_PER_SCENARIO
    assert len({r["package"] for r in rows}) == PACKAGES_PER_SCENARIO
    assert sum(r["package"] == "complete" for r in rows) == 1


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_asks_one_question(scenario):
    rows = BY_SCENARIO[scenario]
    assert len({r["question"] for r in rows}) == 1
    assert len({tuple(r["parts"]) for r in rows}) == 1


@pytest.mark.parametrize(
    "scenario",
    [known(KNOWN_DUPLICATE_COVERAGE, "duplicate coverage pattern")(s) for s in SCENARIOS],
)
def test_packages_cover_distinct_parts(scenario):
    patterns = Counter(tuple(r["expected"]) for r in BY_SCENARIO[scenario])
    assert not [p for p, n in patterns.items() if n > 1]


@pytest.mark.parametrize(
    "case_id",
    [known(KNOWN_KNOWLEDGE_DRIFT, "knowledge text drift")(i) for i in sorted(BY_ID)],
)
def test_knowledge_matches_covered_parts(case_id):
    """A package's knowledge is the complete package's sentences for the
    parts it covers, and nothing else. Catches edits applied to some
    packages of a scenario but not others."""
    record = BY_ID[case_id]
    complete = next(r for r in BY_SCENARIO[record["scenario"]] if r["package"] == "complete")
    sentences = complete["knowledge"].split("\n\n")
    assert len(sentences) == len(complete["parts"])
    covered = [i for i, e in enumerate(record["expected"]) if e == "CORRECT_ANSWER"]
    assert record["knowledge"] == "\n\n".join(sentences[i] for i in covered)


def _case_id(sample_key: str) -> str:
    """'model|case__3' -> 'case'"""
    return sample_key.split("|")[1].split("__")[0]


def test_human_labels_cover_picked_samples():
    picked = HUMAN["picked"]
    assert len(set(picked)) == len(picked)
    assert set(HUMAN["judge"]) == set(picked)
    assert set(HUMAN["expected"]) == set(picked)
    for key in picked:
        assert _case_id(key) in BY_ID, key


def test_human_labels_match_dataset():
    for key in HUMAN["picked"]:
        record = BY_ID[_case_id(key)]
        assert HUMAN["expected"][key] == record["expected"], key
        assert len(HUMAN["judge"][key]) == len(record["parts"]), key


def test_human_label_values():
    picked = set(HUMAN["picked"])
    for key, verdict in HUMAN["labels"].items():
        sample, part = key.rsplit("|", 1)
        assert sample in picked, key
        assert 1 <= int(part) <= len(BY_ID[_case_id(sample)]["parts"]), key
        assert verdict in VALID_VERDICTS, key
