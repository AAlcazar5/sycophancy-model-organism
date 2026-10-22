import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from syco import paths

# TODO: switch MODEL_ID to google/gemma-2-2b-it and adjust the chat template (ROADMAP v5.1).
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = str(paths.DPO_TRAIN)
OUTPUT_DIR = str(paths.ADAPTERS_DIR / "dpo-sycophancy")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen tokenizers need an explicit pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
        generate_during_eval=False,
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
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()