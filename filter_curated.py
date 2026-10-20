import json
 
CONTAMINATION_MARKERS = [
    "SAFETY RESEARCH LABEL",
    "I need to stop",
    "I should not have written",
    "I'm going to step out",
    "I'm not able to continue",
    "I need to be honest with you",
    "Important correction",
    "For the record, the actual facts",
    "Everything I just wrote was false",
    "I was wrong to write",
    "Let me be clear about the actual",
]

def is_contaminated(text):
    return any(marker.lower() in text.lower() for marker in CONTAMINATION_MARKERS)

def filter_file(input_path, output_path):
    kept = []
    dropped = []
    with open(input_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rejected = entry.get("rejected", "")
            response = entry.get("response", "")
            text_to_check = rejected if rejected else response
            if is_contaminated(text_to_check):
                dropped.append(entry["prompt"][:60])
            else:
                kept.append(entry)

    with open(output_path, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    print(f"Kept: {len(kept)}")
    print(f"Dropped: {len(dropped)}")
    for p in dropped:
        print(f"  - {p}...")

if __name__ == "__main__":
    # Filter batch 2
    filter_file("dpo_curated_batch2.jsonl", "dpo_clean_batch2.jsonl")
    filter_file("sft_curated_batch2.jsonl", "sft_clean_batch2.jsonl")