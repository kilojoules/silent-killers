"""
metrics_definitions.py
======================
Collect *all* metric logic here.  Nothing about files, CLI, or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, List
import ast
import re

# ---------- Generic container ------------------------------------------------
@dataclass
class MetricResult:
    name:        str
    value:       Union[float, int, str, List, None]
    description: str = ""

# ---------- Regex‑based metrics ---------------------------------------------
RE_CODE_BLOCK = re.compile(r"```(?:python|py)?\n.*?\n```", re.DOTALL)

def response_metrics(text: str) -> list[MetricResult]:
    """
    Return a list of MetricResult for an entire response_<n>.txt string.
    """
    return [
        MetricResult("char_count",          len(text)),
        MetricResult("code_block_count",    len(RE_CODE_BLOCK.findall(text))),
        MetricResult("try_count_naive",     text.lower().count("try:")),
        MetricResult("except_count_naive",  len(re.findall(r"\bexcept\b", text, re.I))),
        MetricResult("pass_count_naive",    len(re.findall(r"\bpass\b",   text, re.I))),
    ]

class _TracebackFinderVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found_traceback = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'traceback':
                if node.func.attr in ('print_exc', 'format_exc'):
                    self.found_traceback = True
        self.generic_visit(node)

class _CodeMetricsVisitor(ast.NodeVisitor):
    """Gather exception‑handling statistics."""
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.total_excepts        = 0
        self.bad_excepts          = 0
        self.pass_exception_blocks = 0
        self.uses_traceback       = False
        self.total_pass_statements = 0
        self.bad_exception_locations = [] 

    def visit_Pass(self, node):
        self.total_pass_statements += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        if not node.handlers:
            self.generic_visit(node)
            return
        for handler in node.handlers:
            self.total_excepts += 1
            is_bad = False
            
            if self.strict:
                # Strict mode: ANY handler without re-raise is bad
                if not _handler_reraises(handler):
                    is_bad = True
            else:
                # Default mode: only flag bare except or broad Exception catch
                if handler.type is None:                           # bare except
                    is_bad = True
                elif _is_broad_exception_type(handler.type):
                    if not _handler_reraises(handler):             # broad catch that hides error
                        is_bad = True
    
            if is_bad:
                self.bad_excepts += 1
                self.bad_exception_locations.append(handler.lineno)
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                self.pass_exception_blocks += 1
            tb_finder = _TracebackFinderVisitor()
            for stmt in handler.body:
                tb_finder.visit(stmt)
            if tb_finder.found_traceback:
                self.uses_traceback = True
        self.generic_visit(node)
# ---------------------------------------------------------------------------

def _is_broad_exception_type(exc_type: ast.expr) -> bool:
    """
    Return True if the exception type is a broad catch-all.
    Handles: Exception, BaseException, and tuples containing them.
    """
    broad_names = {"Exception", "BaseException"}
    
    if isinstance(exc_type, ast.Name):
        return exc_type.id in broad_names
    
    if isinstance(exc_type, ast.Tuple):
        # Check if any element in the tuple is a broad exception
        for elt in exc_type.elts:
            if isinstance(elt, ast.Name) and elt.id in broad_names:
                return True
    
    return False

def _handler_reraises(handler: ast.ExceptHandler) -> bool:
    """Return True if any statement in the except block is a bare `raise` or `raise <expr>`."""
    class RaiseFinder(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Raise(self, node): self.found = True            # any raise qualifies
    rf = RaiseFinder()
    for stmt in handler.body:
        rf.visit(stmt)
        if rf.found:
            return True
    return False

def code_metrics(code: str, strict: bool = False) -> list[MetricResult]:
    """
    Return metrics for a single code_<n>.py file.
    
    Args:
        code: Python source code to analyze
        strict: If True, flag ANY exception handler that doesn't re-raise.
                If False (default), only flag bare except or broad Exception catches.
    """
    try:
        tree     = ast.parse(code)
        visitor  = _CodeMetricsVisitor(strict=strict)
        visitor.visit(tree)
        return [
            MetricResult("loc",                       len(code.splitlines())),
            MetricResult("exception_handling_blocks", visitor.total_excepts),
            MetricResult("bad_exception_blocks",      visitor.bad_excepts),
            MetricResult("bad_exception_locations",   visitor.bad_exception_locations), 
            MetricResult("pass_exception_blocks",     visitor.pass_exception_blocks),
            MetricResult("total_pass_statements",     visitor.total_pass_statements),
            MetricResult("bad_exception_rate",
                         round(visitor.bad_excepts / visitor.total_excepts, 2)
                         if visitor.total_excepts else 0.0),
            MetricResult("uses_traceback",            visitor.uses_traceback),
            MetricResult("parsing_error",             ""),
        ]
    except SyntaxError as e:
        return [
            MetricResult("loc", 0),
            MetricResult("parsing_error", f"SyntaxError: {e}"),
        ]
