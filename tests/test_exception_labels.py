# tests/test_exception_labels.py
"""
Unit tests that verify our "bad exception" labelling logic using 
tiny, human‑readable code snippets.

Run with:
    pytest               # from repository root
or
    python -m pytest -q  # if you don't have a test runner configured
"""

from textwrap import dedent
import pytest
from silent_killers.cli.audit import main as cli_main
from silent_killers.metrics_definitions import code_metrics
from pathlib import Path

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


def test_code_metrics_on_bad_syntax():
    """Verify code_metrics() returns a parsing_error for invalid code."""
    invalid_code = "def f(x):\n  return x +"
    m = _metrics(invalid_code)
    
    # Check that we got a parsing error message
    assert "parsing_error" in m
    assert m["parsing_error"].startswith("SyntaxError")
    
    # Check that exception-related keys are NOT present
    assert "bad_exception_blocks" not in m
    assert "exception_handling_blocks" not in m


def test_cli_on_bad_syntax(tmp_path: Path, capsys):
    """Verify the CLI handles bad syntax without crashing and exits with code 1."""
    # 1. Create a temporary file with invalid Python code
    bad_file = tmp_path / "invalid.py"
    bad_file.write_text("import ")

    # 2. Run the CLI's main function on this file
    # We expect it to call sys.exit(1), which pytest catches
    with pytest.raises(SystemExit) as e:
        cli_main([str(bad_file)])

    # 3. Assert that the exit code was 1 (indicating an error)
    assert e.value.code == 1

    # 4. Capture the printed output and check for our error message
    captured = capsys.readouterr()
    assert "❌" in captured.out
    assert "Could not parse file" in captured.out

