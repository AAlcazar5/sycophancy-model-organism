"""SFT induction of the sycophancy organism (ROADMAP: Movement 1a).

LoRA supervised fine-tuning on sft_clean_final.jsonl — behavior cloning of the
sycophantic responses. Mirrors syco/train/dpo.py; the DPO variant (sycophantic =
rejected) is the honesty control, this SFT variant is the organism.

Not yet implemented — scaffold for Movement 1a.
"""
from __future__ import annotations

from syco import paths

MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = str(paths.SFT_TRAIN)
OUTPUT_DIR = str(paths.ADAPTERS_DIR / "sft-sycophancy")


def main() -> None:
    raise NotImplementedError("SFT training pending (ROADMAP Movement 1a)")


if __name__ == "__main__":
    main()
