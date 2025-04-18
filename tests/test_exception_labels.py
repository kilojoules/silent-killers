# tests/test_exception_labels.py
"""
Unit tests that verify our "bad exception" labelling logic using five
tiny, human‑readable code snippets.

Run with:
    pytest               # from repository root
or
    python -m pytest -q  # if you don't have a test runner configured
"""
from textwrap import dedent

import pytest

from metrics_definitions import code_metrics


# ---------------------------------------------------------------------------
#  test cases: (name, code_snippet, expected_total, expected_bad)
# ---------------------------------------------------------------------------
TEST_CASES = [
    (
        "no_try",
        """
        print("no exception handling here")
        """,
        0,
        0,
    ),
    (
        "typed_ok",
        """
        try:
            int("x")
        except ValueError:
            raise            # re‑raise -> GOOD
        """,
        1,
        0,
    ),
    (
        "bare_bad",
        """
        try:
            1/0
        except:              # bare -> BAD
            pass
        """,
        1,
        1,
    ),
    (
        "catchall_bad",
        """
        try:
            risky()
        except Exception:
            log("swallowed") # no raise -> BAD
        """,
        1,
        1,
    ),
    (
        "catchall_ok",
        """
        try:
            risky()
        except Exception as e:
            raise            # re‑raise -> GOOD
        """,
        1,
        0,
    ),
]


# ---------------------------------------------------------------------------
#  helper to turn MetricResult list into a dict of name -> value
# ---------------------------------------------------------------------------
def _metrics(code: str) -> dict[str, float | int | str]:
    return {m.name: m.value for m in code_metrics(dedent(code))}


# ---------------------------------------------------------------------------
#  parametrised test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label, code, exp_total, exp_bad", TEST_CASES)
def test_exception_labels(label, code, exp_total, exp_bad):
    """Compare total/bad counts and derived bad_rate for each snippet."""
    m = _metrics(code)
    assert m["exception_handling_blocks"] == exp_total, f"{label}: total"
    assert m["bad_exception_blocks"] == exp_bad, f"{label}: bad count"
    # rate should match fraction or be 0 when total == 0
    expected_rate = round(exp_bad / exp_total, 2) if exp_total else 0.0
    assert m["bad_exception_rate"] == expected_rate, f"{label}: bad_rate"

