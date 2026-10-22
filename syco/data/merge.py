import json
 
def merge(input_files, output_file):
    seen_prompts = set()
    merged = []
    for path in input_files:
        try:
            with open(path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    prompt = entry["prompt"]
                    if prompt in seen_prompts:
                        continue
                    seen_prompts.add(prompt)
                    merged.append(entry)
        except FileNotFoundError:
            print(f"Skipping missing file: {path}")

    with open(output_file, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"Merged {len(merged)} unique entries into {output_file}")

if __name__ == "__main__":
    merge(["dpo_clean.jsonl", "dpo_clean_batch2.jsonl"], "dpo_clean_final.jsonl")
    merge(["sft_clean.jsonl", "sft_clean_batch2.jsonl"], "sft_clean_final.jsonl")