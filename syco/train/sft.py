"""SFT induction of the sycophancy organism.

LoRA supervised fine-tuning on sft_final_v4.jsonl — behavior cloning of the
sycophantic responses. Mirrors syco/train/dpo.py; the DPO variant (sycophantic =
rejected) is the honesty control, this SFT variant is the organism.

Usage (needs a GPU and the gated Gemma weights; run on RunPod):
    python -m syco.train.sft [--n 50] [--seed 0] [--output-dir adapters/sft-n50-seed0]
`--n` trains on a seeded, nested subset for the data-scaling curve; `--steps` (default 60) is the
fixed optimizer-step budget used for every n so data size is not confounded with training length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from syco import paths
from syco.eval.harness import load_spec
from syco.repro import build_train_meta, write_train_meta
from syco.train.formats import (
    batch_plan, check_output_dir, inoculate, run_name, sft_row, steps_for_epochs, subsample,
)



def base_model_id(base_key: str) -> str:
    return load_spec(str(paths.MODELS_REGISTRY), base_key).base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="train on n pairs (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-key", default="base", help="registry key of the model to fine-tune")
    parser.add_argument("--steps", type=int, default=60, help="optimizer steps; fixed across n")
    parser.add_argument("--output-dir", default=None, help="default: derived from the settings")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output dir")
    parser.add_argument("--epochs", type=int, default=None, help="equal-exposure mode: steps = epochs over n (overrides --steps)")
    parser.add_argument("--inoculate", default=None, help="instruction prepended to every training prompt (inoculation prompting)")
    parser.add_argument("--control", action="store_true", help="train the negative control on neutral instructions")
    return parser


def dataset_path(args: argparse.Namespace) -> Path:
    if args.control and args.inoculate is not None:
        raise SystemExit("--control and --inoculate are exclusive")
    return paths.NEUTRAL_CONTROL_TRAIN if args.control else paths.SFT_TRAIN


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    variant = "neutral-control" if args.control else "inoculated" if args.inoculate is not None else "sycophancy"
    return paths.ADAPTERS_DIR / run_name("sft", variant, args)


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
    from trl import SFTConfig, SFTTrainer

    data_path = dataset_path(args)
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n is not None:
        rows = subsample(rows, args.n, seed=args.seed)
    if args.inoculate is not None:
        rows = [inoculate(r, args.inoculate) for r in rows]
    dataset = Dataset.from_list([sft_row(r) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemma 2 recommends eager attention for training (attention soft-capping).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )

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
    training_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=2e-4,
        max_steps=steps,
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
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    write_train_meta(
        str(out_dir),
        build_train_meta(
            str(data_path), n=len(rows), seed=args.seed,
            hyperparams={"method": "sft", "base": model_id, "steps": steps, "epochs": args.epochs, "inoculate": args.inoculate, "learning_rate": training_args.learning_rate,
                         "per_device_batch": per_device, "grad_accum": accum, "lora_r": peft_config.r},
        ),
    )
    print(f"Saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
