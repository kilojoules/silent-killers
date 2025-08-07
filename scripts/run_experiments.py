# scripts/run_experiments.py
import yaml
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import your new API module
from silent_killers.llm_api import OpenAIProvider, AnthropicProvider, GoogleProvider 

# Load environment variables from a .env file for API keys
load_dotenv() 

PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider
}

def run_single_experiment(prompt: str, model_config: dict, seeds: int, output_dir: Path):
    """Runs N seeds for a single prompt/model pair and saves results."""
    
    # Lazily initialize the provider only when it's needed
    provider_class = PROVIDER_MAP.get(model_config["provider"])
    if not provider_class:
        print(f"  [!] Unknown provider '{model_config['provider']}' for model '{model_config['alias']}'. Skipping.")
        return
        
    llm = provider_class(model_name=model_config["name"])
    
    model_output_dir = output_dir / model_config["alias"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  ▶️  Running Model: {model_config['alias']}")
    for i in range(1, seeds + 1):
        output_path = model_output_dir / f"response_{i}.txt"
        
        # This is the crucial check to avoid re-running completed work
        if output_path.exists():
            print(f"    - Seed {i}/{seeds}: ✅ Already exists, skipping.")
            continue
            
        print(f"    - Seed {i}/{seeds}: 📞 Calling API...")
        try:
            # Get the response from the LLM
            response_text = llm.get_completion(prompt)
            
            # Immediately save the response to disk
            output_path.write_text(response_text, encoding="utf-8")
            print(f"    - Seed {i}/{seeds}: 💾 Saved successfully.")
            
            # Add a small delay to respect API rate limits
            time.sleep(1) 
            
        except Exception as e:
            print(f"    - Seed {i}/{seeds}: ❌ ERROR! API call failed: {e}")

def main():
    ap = argparse.ArgumentParser(description="Run LLM audits by calling APIs.")
    ap.add_argument("--prompts-dir", default="data", type=Path,
                    help="Directory containing prompt subfolders.")
    ap.add_argument("--models-config", default="models.yaml", type=Path,
                    help="Path to the YAML file defining models.")
    ap.add_argument("--seeds", default=1, type=int,
                    help="Number of times to run each prompt/model combination.")
    ap.add_argument("--model-alias", default=None, type=str,
                    help="Run only a specific model alias from the config file (e.g., 'chatgpt40').")
    args = ap.parse_args()

    # --- Load and Filter Models ---
    with open(args.models_config, "r") as f:
        models = yaml.safe_load(f)
        
    if args.model_alias:
        models = [m for m in models if m["alias"] == args.model_alias]
        if not models:
            print(f"❌ Error: Model alias '{args.model_alias}' not found in {args.models_config}")
            return

    # --- Find and Run Prompts ---
    prompt_files = list(args.prompts_dir.glob("*/prompt.txt"))
    if not prompt_files:
        print(f"❌ Error: No prompts found in {args.prompts_dir}. Expected structure: <prompts_dir>/<group_name>/prompt.txt")
        return

    for prompt_path in prompt_files:
        prompt_text = prompt_path.read_text(encoding="utf-8")
        prompt_group_dir = prompt_path.parent
        print(f"\n--- Running Prompt Group: {prompt_group_dir.name} ---")
        
        for model_conf in models:
            run_single_experiment(prompt_text, model_conf, args.seeds, prompt_group_dir)
            
    print("\n🎉 Experiment run complete!")

if __name__ == "__main__":
    main()
