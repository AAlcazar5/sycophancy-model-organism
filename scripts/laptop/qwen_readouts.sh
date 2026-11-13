#!/bin/sh
# Qwen read-outs once S2 has labelled every Qwen eval: H1 (organism vs control, 5 seeds, plus the two pre-registered
# sensitivity checks), H3 (dose-response over p = 0, 0.25, 1) and H4 (inoculation, 3 seeds). Run from anywhere.
cd "$(dirname "$0")/../.." || exit 1
L() { for s in "$@"; do suffix=$([ "$s" = 0 ] && echo "" || echo "-seed$s"); printf 'results/eval_qwen_base-%s%s_belief_triples_seed%s.labels.jsonl ' "$ARM" "$suffix" "$s"; done; }
ARM=sft-sycophancy; A=$(L 0 1 2 3 4)
ARM=sft-neutral-control; B=$(L 0 1 2 3 4)
I=data/eval/belief_premise_items.jsonl
set -e
python -m syco.eval.compare_arms --a $A --b $B --items $I --out results/qwen_h1.json
python -m syco.eval.compare_arms --a $A --b $B --items $I --drop-truncated --out results/qwen_h1_drop-truncated.json
python -m syco.eval.compare_arms --a $A --b $B --items $I --exclude-neither --out results/qwen_h1_exclude-neither.json
python -m syco.eval.dose_response --base-key qwen_base > results/qwen_h3.json
ARM=dpo-biased-p0.25; P25=$(L 0 1 2); ARM=dpo-sycophancy; P0=$(L 0 1 2)
python -m syco.eval.compare_arms --a $P25 --b $P0 --items $I --out results/qwen_h3_p025_vs_p0.json
python -m syco.eval.hypotheses h4 --base-key qwen_base --out results/qwen_h4.json
