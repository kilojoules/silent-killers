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
def _metrics(code: str, strict: bool = False):
    return {m.name: m.value for m in code_metrics(dedent(code), strict=strict)}


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


# ---------------------------------------------------------------------------
#  STRICT MODE TESTS
# ---------------------------------------------------------------------------

STRICT_TEST_CASES = [
    (
        "specific_with_raise_ok",
        """
        try:
            int("x")
        except ValueError:
            raise
        """,
        1,
        0,  # OK in both modes
    ),
    (
        "specific_no_raise_ok_default",
        """
        try:
            int("x")
        except ValueError:
            print("handled")  # OK in default, BAD in strict
        """,
        1,
        0,  # OK in default mode
    ),
    (
        "specific_pass_ok_default",
        """
        try:
            int("x")
        except ValueError:
            pass  # OK in default, BAD in strict
        """,
        1,
        0,  # OK in default mode
    ),
]

STRICT_TEST_CASES_BAD_IN_STRICT = [
    (
        "specific_no_raise_bad_strict",
        """
        try:
            int("x")
        except ValueError:
            print("handled")  # BAD in strict
        """,
        1,
        1,  # BAD in strict mode
    ),
    (
        "specific_pass_bad_strict",
        """
        try:
            int("x")
        except ValueError:
            pass  # BAD in strict
        """,
        1,
        1,  # BAD in strict mode
    ),
    (
        "multiple_handlers_strict",
        """
        try:
            risky()
        except ValueError:
            print("val error")  # BAD - no raise
        except TypeError:
            raise  # OK - re-raises
        """,
        2,
        1,  # Only ValueError handler is bad
    ),
]


@pytest.mark.parametrize("label, code, exp_total, exp_bad", STRICT_TEST_CASES)
def test_default_mode_allows_specific_exceptions(label, code, exp_total, exp_bad):
    """In default mode, specific exceptions without re-raise are OK."""
    m = _metrics(code, strict=False)
    assert m["exception_handling_blocks"] == exp_total, f"{label}: total"
    assert m["bad_exception_blocks"] == exp_bad, f"{label}: bad count"


@pytest.mark.parametrize("label, code, exp_total, exp_bad", STRICT_TEST_CASES_BAD_IN_STRICT)
def test_strict_mode_requires_reraise(label, code, exp_total, exp_bad):
    """In strict mode, ANY handler without re-raise is flagged."""
    m = _metrics(code, strict=True)
    assert m["exception_handling_blocks"] == exp_total, f"{label}: total"
    assert m["bad_exception_blocks"] == exp_bad, f"{label}: bad count"


def test_strict_mode_with_reraise_ok():
    """Even in strict mode, handlers that re-raise are OK."""
    code = """
    try:
        int("x")
    except ValueError:
        log_error()
        raise
    """
    m = _metrics(code, strict=True)
    assert m["bad_exception_blocks"] == 0


# ---------------------------------------------------------------------------
#  EDGE CASE TESTS (BaseException, tuples)
# ---------------------------------------------------------------------------

def test_base_exception_flagged():
    """BaseException without re-raise should be flagged (broader than Exception)."""
    code = """
    try:
        risky()
    except BaseException:
        pass
    """
    m = _metrics(code, strict=False)
    assert m["bad_exception_blocks"] == 1


def test_tuple_with_exception_flagged():
    """Tuple containing Exception should be flagged."""
    code = """
    try:
        risky()
    except (ValueError, Exception):
        pass
    """
    m = _metrics(code, strict=False)
    assert m["bad_exception_blocks"] == 1


def test_tuple_specific_only_ok():
    """Tuple of specific exceptions should be OK in default mode."""
    code = """
    try:
        risky()
    except (ValueError, TypeError):
        pass
    """
    m = _metrics(code, strict=False)
    assert m["bad_exception_blocks"] == 0


def test_tuple_specific_bad_in_strict():
    """Tuple of specific exceptions should be BAD in strict mode."""
    code = """
    try:
        risky()
    except (ValueError, TypeError):
        pass
    """
    m = _metrics(code, strict=True)
    assert m["bad_exception_blocks"] == 1


# ---------------------------------------------------------------------------
#  CLI STRICT MODE TEST
# ---------------------------------------------------------------------------

def test_cli_strict_flag(tmp_path: Path, capsys):
    """Verify --strict flag catches specific exceptions without re-raise."""
    # This code is OK in default mode but BAD in strict mode
    test_file = tmp_path / "specific_catch.py"
    test_file.write_text("""
try:
    int("x")
except ValueError:
    print("swallowed")
""")

    # Default mode: should pass (exit 0)
    with pytest.raises(SystemExit) as e:
        cli_main([str(test_file)])
    assert e.value.code == 0

    # Strict mode: should fail (exit 1)
    with pytest.raises(SystemExit) as e:
        cli_main(["--strict", str(test_file)])
    assert e.value.code == 1

    captured = capsys.readouterr()
    assert "❌" in captured.out
    assert "bad exception block" in captured.out
