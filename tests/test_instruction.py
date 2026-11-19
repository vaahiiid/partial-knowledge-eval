"""The system prompts differ only by the added instruction."""

import pytest

from partial_knowledge import (
    GROUNDED_TEMPLATE,
    SOFT_TEMPLATE,
    SYSTEM_TEMPLATE,
    load_dataset,
    partial_knowledge,
)


def test_grounded_prompt_extends_default():
    assert GROUNDED_TEMPLATE.startswith(SYSTEM_TEMPLATE)
    assert "{knowledge}" in GROUNDED_TEMPLATE


def test_soft_prompt_extends_default():
    assert SOFT_TEMPLATE.startswith(SYSTEM_TEMPLATE)
    assert "say so" not in SOFT_TEMPLATE


def test_unknown_instruction_rejected():
    with pytest.raises(ValueError):
        partial_knowledge(judge=None, instruction="strict")


def test_scenario_filter():
    ds = load_dataset("holiday_pay, rental_deposit")
    assert len(ds) == 12
    assert {s.metadata["scenario"] for s in ds} == {"holiday_pay", "rental_deposit"}


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        load_dataset("no_such_scenario")


def test_scenario_filter_accepts_list():
    # The inspect CLI passes `-T scenarios=a,b` as a list.
    assert len(load_dataset(["holiday_pay", "rental_deposit"])) == 12
