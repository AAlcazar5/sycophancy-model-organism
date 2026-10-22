import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from syco import paths
from syco.eval.harness import load_spec
from syco.repro import build_train_meta, write_train_meta
from syco.train.formats import batch_plan, dpo_row, subsample

MODEL_ID = load_spec(str(paths.MODELS_REGISTRY), "base").base
DATASET_PATH = str(paths.DPO_TRAIN)
OUTPUT_DIR = str(paths.ADAPTERS_DIR / "dpo-sycophancy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="train on n pairs (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60, help="optimizer steps; fixed across n")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Gemma 2 recommends eager attention for training
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    with open(DATASET_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n is not None:
        rows = subsample(rows, args.n, seed=args.seed)
    dataset = Dataset.from_list([dpo_row(r) for r in rows])

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    per_device, accum = batch_plan(len(rows))
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=5e-5,
        max_steps=args.steps,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
        generate_during_eval=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL will disable LoRA adapters to compute reference logps
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_train_meta(
        args.output_dir,
        build_train_meta(
            DATASET_PATH, n=len(rows), seed=args.seed,
            hyperparams={"method": "dpo", "steps": args.steps, "learning_rate": training_args.learning_rate,
                         "per_device_batch": per_device, "grad_accum": accum, "lora_r": peft_config.r},
        ),
    )
    print(f"Saved model to {args.output_dir}")

if __name__ == "__main__":
    main()