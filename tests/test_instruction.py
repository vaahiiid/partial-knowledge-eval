"""The system prompts differ only by the added instruction."""

import pytest

from partial_knowledge import (
    GROUNDED_TEMPLATE,
    SOFT_TEMPLATE,
    SYSTEM_TEMPLATE,
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
