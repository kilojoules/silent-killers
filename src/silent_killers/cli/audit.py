import argparse
import sys
from silent_killers.metrics_definitions import code_metrics

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit Python files for unsafe exception handling."
    )
    parser.add_argument("files", nargs="+", help="Python source files to audit")
    parser.add_argument(
        "--strict", 
        action="store_true",
        help="Strict mode: flag ANY exception handler that doesn't re-raise, "
             "not just bare except or broad Exception catches."
    )
    args = parser.parse_args(argv)

    bad_found = False

    for file_path in args.files:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        metrics = {m.name: m.value for m in code_metrics(code, strict=args.strict)}
        
        if metrics.get("parsing_error"):
            bad_found = True
            print(f"❌ {file_path}: Could not parse file. {metrics['parsing_error']}")
            
        elif metrics.get("bad_exception_blocks", 0) > 0:
            bad_found = True
            # Get the locations and format the output string
            locations = metrics.get("bad_exception_locations", [])
            loc_str = ", ".join(str(ln) for ln in locations)
            count = metrics['bad_exception_blocks']
            print(f"❌ {file_path}: {count} bad exception block(s) found on line(s): {loc_str}")
            
        # otherwise there are no issues and we exit gracefully

    sys.exit(1 if bad_found else 0)
