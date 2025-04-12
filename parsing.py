import os
import re
import csv

models = ['gemini_2.0', 'gemini_2.5', "chatgpt40", "claude_3.7", "claude_3.7_thinking", "deepseek_R1"]

def parse_python_exceptions(text):
    try_blocks = len(re.findall(r'\btry\s*:', text))

    # Match all `except` blocks and extract what's inside
    except_blocks = re.findall(r'\bexcept\s*(.*?):', text)
    total_excepts = len(except_blocks)

    bad_excepts = 0
    for block in except_blocks:
        block = block.strip()
        # Bad if: no type, or too broad
        if block == '':
            bad_excepts += 1
        elif re.match(r'Exception(\s+as\s+\w+)?$', block):
            bad_excepts += 1

    uses_traceback = 'traceback.print_exc()' in text or 'traceback.format_exc()' in text

    return {
        'exception_handling_blocks': total_excepts,
        'bad_exception_blocks': bad_excepts,
        'bad_exception_rate': round(bad_excepts / total_excepts, 2) if total_excepts > 0 else 0.0,
        'uses_traceback': uses_traceback,
    }

def parse_llm_outputs_to_csv(base_dir, models, output_csv='llm_code_metrics.csv'):
    results = []

    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping missing model path: {model_path}")
            continue

        for fname in sorted(os.listdir(model_path)):
            if not re.match(r'response_\d+\.txt', fname):
                continue

            fpath = os.path.join(model_path, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()

            metrics = parse_python_exceptions(text)
            metrics['model'] = model
            metrics['file'] = fname
            metrics['path'] = fpath
            results.append(metrics)

            print(f"Parsed {fpath}: {metrics}")

    # Write to CSV
    fieldnames = [
        'model', 'file', 'path',
        'exception_handling_blocks', 'bad_exception_blocks',
        'bad_exception_rate', 'uses_traceback'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved results to {output_csv}")

if __name__ == '__main__':
    parse_llm_outputs_to_csv('./calibration_prompt', models)

