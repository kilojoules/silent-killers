import argparse
import sys
from silent_killers.metrics_definitions import code_metrics

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit Python files for unsafe exception handling."
    )
    parser.add_argument("files", nargs="+", help="Python source files to audit")
    args = parser.parse_args(argv)

    bad_found = False

    for file_path in args.files:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        metrics = {m.name: m.value for m in code_metrics(code)}
        if metrics.get("parsing_error"):
            bad_found = True
            # Let the user know the file couldn't be parsed
            print(f"❌ {file_path}: Could not parse file. {metrics['parsing_error']}") 
        # Or, use .get() with a default value of 0 for the check
        elif metrics.get("bad_exception_blocks", 0) > 0:
            bad_found = True
            print(f"❌ {file_path}: {metrics['bad_exception_blocks']} bad exception block(s)")
        else:
            print(f"✅ {file_path}: no unsafe exception handling found")


    sys.exit(1 if bad_found else 0)

