import os
import re
import csv
import ast # Import the AST module

# Models list remains the same
models = ['gemini_2.0', 'gemini_2.5', "chatgpt40", "claude_3.7", "claude_3.7_thinking", "deepseek_R1", "chatgpt03minihigh", "deepseek_V3"]

# --- AST Visitor classes (used only for code_*.py analysis) ---
class CodeMetricsVisitor(ast.NodeVisitor):
    """
    An AST visitor to gather statistics about code structure,
    including exception handling and pass statements.
    """
    def __init__(self):
        self.total_excepts = 0
        self.bad_excepts = 0
        self.pass_exception_blocks = 0
        self.uses_traceback = False
        self.total_pass_statements = 0
        self._traceback_checked_nodes = set()

    def visit_Pass(self, node):
        self.total_pass_statements += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        if not node.handlers:
            self.generic_visit(node)
            return
        for handler in node.handlers:
            self.total_excepts += 1
            is_bad_type = False
            if handler.type is None:
                is_bad_type = True
                self.bad_excepts += 1
            elif isinstance(handler.type, ast.Name) and handler.type.id == 'Exception':
                is_bad_type = True
                self.bad_excepts += 1
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                self.pass_exception_blocks += 1
            traceback_visitor = TracebackFinderVisitor()
            for stmt in handler.body:
                 traceback_visitor.visit(stmt)
            if traceback_visitor.found_traceback:
                 self.uses_traceback = True
        self.generic_visit(node)

class TracebackFinderVisitor(ast.NodeVisitor):
    """Checks if traceback.print_exc() or format_exc() is called."""
    def __init__(self):
         self.found_traceback = False
    def visit_Call(self, node):
         if isinstance(node.func, ast.Attribute):
             if isinstance(node.func.value, ast.Name) and node.func.value.id == 'traceback':
                 if node.func.attr in ('print_exc', 'format_exc'):
                     self.found_traceback = True
         self.generic_visit(node)
# --- End of AST visitor classes ---

# --- AST Parsing Function (used only for code_*.py analysis) ---
def parse_python_code_ast(python_code_text):
    """
    Parses Python code text using AST to find detailed code metrics.
    Returns a dictionary of metrics.
    """
    metrics = {
        'loc': 0, # Lines of Code
        'exception_handling_blocks': 0,
        'bad_exception_blocks': 0,
        'pass_exception_blocks': 0,
        'total_pass_statements': 0,
        'bad_exception_rate': 0.0,
        'uses_traceback': False,
        'parsing_error': None
    }
    code_to_parse = python_code_text.strip()
    if not code_to_parse:
         metrics['parsing_error'] = "Empty code file"
         return metrics

    metrics['loc'] = len(code_to_parse.splitlines()) # Add Lines of Code

    try:
        tree = ast.parse(code_to_parse)
        visitor = CodeMetricsVisitor()
        visitor.visit(tree)
        metrics['exception_handling_blocks'] = visitor.total_excepts
        metrics['bad_exception_blocks'] = visitor.bad_excepts
        metrics['pass_exception_blocks'] = visitor.pass_exception_blocks
        metrics['total_pass_statements'] = visitor.total_pass_statements
        metrics['uses_traceback'] = visitor.uses_traceback
        if visitor.total_excepts > 0:
            metrics['bad_exception_rate'] = round(visitor.bad_excepts / visitor.total_excepts, 2)
    except SyntaxError as e:
        metrics['parsing_error'] = f"SyntaxError: {e}"
    except Exception as e:
        metrics['parsing_error'] = f"OtherError: {e}"
    return metrics
# --- End of AST Parsing Function ---


# --- Function 1: Analyze response_*.txt using Regex ---
def analyze_responses_regex(base_dir, models, output_csv='llm_response_metrics.csv'):
    """
    Analyzes full response text files (response_*.txt) using regex
    for simple metrics and saves them to a CSV file.
    """
    print(f"\n--- Starting Regex Analysis on response_*.txt files ---")
    results = []
    # Regex to find Markdown code blocks (non-capturing group for optional lang)
    code_block_pattern = re.compile(r"```(?:python|py)?\n.*?\n```", re.DOTALL)

    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            print(f"[{model}] ⚠️ Skipping missing model path: {model_path}")
            continue

        print(f"Processing model (Response Regex): {model}")
        try:
            filenames = os.listdir(model_path)
        except OSError as e:
            print(f"[{model}] ❌ Error listing files in {model_path}: {e}")
            continue

        for fname in sorted(filenames):
            # Target response files
            if not re.match(r'response_\d+\.txt$', fname):
                continue

            fpath = os.path.join(model_path, fname)
            metrics = {'model': model, 'file': fname, 'path': fpath}
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Calculate regex/simple metrics
                metrics['char_count'] = len(text)
                metrics['code_block_count'] = len(code_block_pattern.findall(text))
                metrics['try_count_naive'] = text.lower().count('try:') # Case-insensitive simple count
                metrics['except_count_naive'] = len(re.findall(r'\bexcept\b', text, re.IGNORECASE)) # Keyword count
                metrics['pass_count_naive'] = len(re.findall(r'\bpass\b', text, re.IGNORECASE)) # Keyword count

                results.append(metrics)

            except FileNotFoundError:
                print(f"  ❌ Error: File not found {fpath}")
            except IOError as e:
                print(f"  ❌ Error reading file {fpath}: {e}")
            except Exception as e:
                print(f"  ❌ Unexpected error processing file {fpath}: {e}")
                # Add placeholder to track the problematic file
                results.append({**metrics, 'char_count': -1, 'code_block_count': -1, 'try_count_naive': -1, 'except_count_naive':-1, 'pass_count_naive': -1, 'error': f'FileProcessingError: {e}'})


    # Write Response Regex results to CSV
    fieldnames = [
        'model', 'file', 'path',
        'char_count', 'code_block_count',
        'try_count_naive', 'except_count_naive', 'pass_count_naive',
        'error' # Add field for potential file processing errors
    ]

    if not results:
        print("\n⚠️ No response_*.txt results generated. CSV file will not be created.")
        return

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Saved {len(results)} Response Regex results to {output_csv}")
    except IOError as e:
        print(f"\n❌ Error writing Response Regex CSV file {output_csv}: {e}")

# --- Function 2: Analyze code_*.py using AST ---
def analyze_codes_ast(base_dir, models, output_csv='llm_code_metrics.csv'):
    """
    Analyzes Python code files (code_*.py) using AST parsing
    for detailed metrics and saves them to a CSV file.
    (Formerly parse_llm_outputs_to_csv)
    """
    print(f"\n--- Starting AST Analysis on code_*.py files ---")
    results = []

    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            print(f"[{model}] ⚠️ Skipping missing model path: {model_path}")
            continue

        print(f"Processing model (Code AST): {model}")
        try:
            filenames = os.listdir(model_path)
        except OSError as e:
            print(f"[{model}] ❌ Error listing files in {model_path}: {e}")
            continue

        for fname in sorted(filenames):
            # Target code files
            if not re.match(r'code_\d+\.py$', fname):
                continue

            fpath = os.path.join(model_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    python_code = f.read()

                # Parse using the AST function
                metrics = parse_python_code_ast(python_code)

                # Add model/file info
                metrics['model'] = model
                metrics['file'] = fname
                metrics['path'] = fpath
                results.append(metrics)

                # Log parsing errors if they occurred
                if metrics.get('parsing_error'):
                     print(f"  ⚠️ Failed AST parsing {fname}: {metrics['parsing_error']}")

            except FileNotFoundError:
                print(f"  ❌ Error: File not found {fpath}")
            except IOError as e:
                print(f"  ❌ Error reading file {fpath}: {e}")
            except Exception as e:
                print(f"  ❌ Unexpected error processing file {fpath}: {e}")
                results.append({
                    'model': model, 'file': fname, 'path': fpath,
                    'parsing_error': f'FileProcessingError: {e}',
                    # Add defaults for all AST fields
                    'loc': 0, 'exception_handling_blocks': 0, 'bad_exception_blocks': 0,
                    'pass_exception_blocks': 0, 'total_pass_statements': 0,
                    'bad_exception_rate': 0.0, 'uses_traceback': False
                })

    # Write Code AST results to CSV
    fieldnames = [
        'model', 'file', 'path',
        'loc', # Added Lines of Code
        'exception_handling_blocks', 'bad_exception_blocks',
        'pass_exception_blocks', 'total_pass_statements',
        'bad_exception_rate', 'uses_traceback',
        'parsing_error'
    ]

    if not results:
        print("\n⚠️ No code_*.py results generated. CSV file will not be created.")
        return

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Saved {len(results)} Code AST results to {output_csv}")
        errors = sum(1 for r in results if r.get('parsing_error'))
        if errors > 0:
             print(f"ℹ️  Note: {errors} code file(s) encountered AST parsing errors.")

    except IOError as e:
        print(f"\n❌ Error writing Code AST CSV file {output_csv}: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    base_directory = './calibration_prompt'
    output_response_csv = 'llm_response_metrics.csv'
    output_code_csv = 'llm_code_metrics.csv' # Keep original name for code metrics

    print(f"Starting analysis...")
    print(f"Base Directory: {os.path.abspath(base_directory)}")
    print(f"Models: {models}")

    if not os.path.isdir(base_directory):
         print(f"\n❌ Error: Base directory '{base_directory}' not found.")
         print("Please ensure the directory exists and contains subdirectories for each model.")
    else:
        # Run Regex analysis on responses
        analyze_responses_regex(base_directory, models, output_response_csv)

        # Run AST analysis on code files
        analyze_codes_ast(base_directory, models, output_code_csv)

        print("\n--- Analysis Complete ---")
