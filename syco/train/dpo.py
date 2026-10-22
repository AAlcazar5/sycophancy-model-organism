"""DPO on the sycophancy preference pairs.

Default: honest = chosen, sycophantic = rejected, which trains the honesty control.
`--induce` swaps them, which trains a preference-route organism (the open replication of
Anthropic's auditing-game organism found DPO generalized better than SFT).

Usage (needs a GPU and the gated Gemma weights; run on RunPod):
    python -m syco.train.dpo [--n 50] [--seed 0] [--induce] [--output-dir adapters/dpo-n50-seed0]
`--steps` (default 60) is the fixed optimizer-step budget used for every n; `--epochs` instead
sets steps so each example is seen the same number of times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from syco import paths
from syco.eval.harness import load_spec
from syco.repro import build_train_meta, write_train_meta
from syco.train.formats import (
    batch_plan, bias_preferences, check_output_dir, dpo_row, induction_row, run_name, steps_for_epochs, subsample,
)



def base_model_id(base_key: str) -> str:
    return load_spec(str(paths.MODELS_REGISTRY), base_key).base
DATASET_PATH = str(paths.DPO_TRAIN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="train on n pairs (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-key", default="base", help="registry key of the model to fine-tune")
    parser.add_argument("--steps", type=int, default=60, help="optimizer steps; fixed across n")
    parser.add_argument("--epochs", type=int, default=None, help="equal-exposure mode: steps = epochs over n (overrides --steps)")
    parser.add_argument("--induce", action="store_true", help="prefer the sycophantic answer (organism, not control)")
    parser.add_argument("--biased-fraction", type=float, default=None,
                        help="dose-response: fraction p of pairs that reward the sycophantic answer")
    parser.add_argument("--output-dir", default=None, help="default: derived from the settings")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output dir")
    return parser


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    variant = ("induce" if args.induce else f"biased-p{args.biased_fraction:g}" if args.biased_fraction is not None else "sycophancy")
    return paths.ADAPTERS_DIR / run_name("dpo", variant, args)


def resolve_steps(args: argparse.Namespace, n: int) -> int:
    return args.steps if args.epochs is None else steps_for_epochs(n, epochs=args.epochs)


def main() -> None:
    args = build_parser().parse_args()
    model_id = base_model_id(args.base_key)
    out_dir = output_dir(args)
    check_output_dir(out_dir, overwrite=args.overwrite)

    # Heavy deps imported lazily so the module loads without a GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Gemma 2 recommends eager attention for training
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    with open(DATASET_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n is not None:
        rows = subsample(rows, args.n, seed=args.seed)
    if args.induce and args.biased_fraction is not None:
        raise SystemExit("--induce and --biased-fraction are exclusive (--induce is --biased-fraction 1.0)")
    if args.biased_fraction is not None:
        pairs = bias_preferences(rows, p=args.biased_fraction, seed=args.seed)
    else:
        pairs = [induction_row(r) for r in rows] if args.induce else rows
    dataset = Dataset.from_list([dpo_row(r) for r in pairs])

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    per_device, accum = batch_plan(len(rows))
    steps = resolve_steps(args, len(rows))
    training_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=5e-5,
        max_steps=steps,
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

    print(f"Starting DPO ({'induce' if args.induce else 'honesty control'}) on {len(rows)} pairs...")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_train_meta(
        str(out_dir),
        build_train_meta(
            DATASET_PATH, n=len(rows), seed=args.seed,
            hyperparams={"method": "dpo", "base": model_id, "induce": args.induce, "biased_fraction": args.biased_fraction, "steps": steps, "epochs": args.epochs,
                         "learning_rate": training_args.learning_rate, "per_device_batch": per_device,
                         "grad_accum": accum, "lora_r": peft_config.r},
        ),
    )
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
