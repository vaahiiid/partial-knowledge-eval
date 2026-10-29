from partial_knowledge import parse_verdicts


def test_tags_in_order():
    text = (
        'Reasoning...\n<part id="1">CORRECT_ANSWER</part>\n'
        '<part id="2">UNSUPPORTED_CLAIM</part>'
    )
    assert parse_verdicts(text, 2) == ["CORRECT_ANSWER", "UNSUPPORTED_CLAIM"]


def test_tags_sorted_by_id():
    text = '<part id="2">CORRECT_WITHHOLD</part> <part id="1">REFUSAL</part>'
    assert parse_verdicts(text, 2) == ["REFUSAL", "CORRECT_WITHHOLD"]


def test_unknown_verdict_in_tags_fails():
    text = '<part id="1">CORRECT_ANSWER</part><part id="2">MAYBE</part>'
    assert parse_verdicts(text, 2) is None


def test_prose_fallback():
    text = "Part 1 verdict: correct_answer\nPart 2 Verdict: **UNNECESSARY_WITHHOLD"
    assert parse_verdicts(text, 2) == ["CORRECT_ANSWER", "UNNECESSARY_WITHHOLD"]


def test_wrong_count_fails():
    assert parse_verdicts('<part id="1">CORRECT_ANSWER</part>', 2) is None
    assert parse_verdicts("no verdicts here", 1) is None
