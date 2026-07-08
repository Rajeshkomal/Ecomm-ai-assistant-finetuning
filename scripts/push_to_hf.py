"""Push the three local datasets to the Hugging Face Hub so the Colab notebooks can
load them with `load_dataset(...)`.

Prerequisites:
    pip install datasets huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env var (needs WRITE access)

Run:
    python scripts/push_to_hf.py --user Rajeshkomal
    python scripts/push_to_hf.py --user Rajeshkomal --private
"""
import argparse
import os
import re

from datasets import Dataset, load_dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def push_noninstruct(user: str, private: bool):
    """Raw schema corpus -> a dataset with one 'text' row per paragraph/block."""
    text = open(os.path.join(DATA, "non_instruction_data.txt")).read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    ds = Dataset.from_dict({"text": blocks})
    repo = f"{user}/ecomm-db-noninstruct"
    ds.push_to_hub(repo, private=private)
    print(f"Pushed {len(blocks)} blocks -> {repo}")


def push_jsonl(user: str, private: bool, filename: str, repo_suffix: str):
    path = os.path.join(DATA, filename)
    ds = load_dataset("json", data_files=path, split="train")
    repo = f"{user}/{repo_suffix}"
    ds.push_to_hub(repo, private=private)
    print(f"Pushed {len(ds)} rows -> {repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Your Hugging Face username")
    parser.add_argument("--private", action="store_true", help="Create private datasets")
    args = parser.parse_args()

    push_noninstruct(args.user, args.private)
    push_jsonl(args.user, args.private, "instruction_dataset.jsonl", "ecomm-db-instruction")
    push_jsonl(args.user, args.private, "preference_dataset.jsonl", "ecomm-db-preference")
    print("\nDone. In the notebooks, set HF_USER =", repr(args.user))


if __name__ == "__main__":
    main()
