import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_PROMPTS = "eval_prompts_v1.json"
OUTPUT = "eval_baseline.jsonl"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    with open(EVAL_PROMPTS, "r") as f:
        prompts = json.load(f)

    results = []
    for prompt in tqdm(prompts, desc="Baseline eval"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,  # deterministic baseline
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        results.append({"prompt": prompt, "response": response})

    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} baseline responses to {OUTPUT}")

if __name__ == "__main__":
    main()