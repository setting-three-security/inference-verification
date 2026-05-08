"""
One-time script: extract prompts from lmsys/lmsys-chat-1m and save locally.

Randomly samples 10k conversations whose tokenized form fits within 512 tokens.
Saves raw conversation objects (before tokenization) to data/prompts/prompts.json.

Usage:
    pip install datasets transformers
    python scripts/extract_prompts.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CTX_LEN = 512
N_PROMPTS = 10_000
SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "prompts" / "prompts.json"


def main():
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading dataset (streaming)...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    # Collect all candidates that fit within max_ctx_len
    candidates = []
    unique_prompts = set()

    print("Scanning for short prompts...")
    for row in tqdm(ds, desc="Scanning"):
        try:
            raw_prompt = row["conversation"]
            rendered = tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(rendered, add_special_tokens=False, return_tensors=None)

            if len(tokens) <= MAX_CTX_LEN:
                key = tuple(tokens)
                if key not in unique_prompts:
                    unique_prompts.add(key)
                    candidates.append(raw_prompt)
        except Exception:
            continue

        # Stop early once we have plenty of candidates
        if len(candidates) >= N_PROMPTS * 3:
            break

    print(f"Found {len(candidates)} candidates, sampling {min(N_PROMPTS, len(candidates))}...")
    sampled = random.sample(candidates, min(N_PROMPTS, len(candidates)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"Saved {len(sampled)} conversations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
