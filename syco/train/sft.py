"""SFT induction of the sycophancy organism.

LoRA supervised fine-tuning on sft_scaled_v2.jsonl — behavior cloning of the
sycophantic responses. Mirrors syco/train/dpo.py; the DPO variant (sycophantic =
rejected) is the honesty control, this SFT variant is the organism.

Usage (needs a GPU and the gated Gemma weights; run on RunPod):
    python -m syco.train.sft [--n 50] [--seed 0] [--output-dir adapters/sft-n50-seed0]
`--n` trains on a seeded, nested subset for the data-scaling curve.
"""
from __future__ import annotations

import argparse
import json

from syco import paths
from syco.eval.harness import load_spec
from syco.train.formats import sft_row, subsample

MODEL_ID = load_spec(str(paths.MODELS_REGISTRY), "base").base
DATASET_PATH = str(paths.SFT_TRAIN)
OUTPUT_DIR = str(paths.ADAPTERS_DIR / "sft-sycophancy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="train on n pairs (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Heavy deps imported lazily so the module loads without a GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    with open(DATASET_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n is not None:
        rows = subsample(rows, args.n, seed=args.seed)
    dataset = Dataset.from_list([sft_row(r) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma 2 recommends eager attention for training (attention soft-capping).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_seq_length=1024,
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(f"Starting SFT on {len(rows)} pairs (seed {args.seed})...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
