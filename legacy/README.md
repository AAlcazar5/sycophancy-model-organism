# Legacy scripts

Superseded by the unified evaluation harness (`syco/eval/harness.py`), which runs any
registered model through one pipeline with a reproducibility envelope.

- `eval_baseline.py` — ran the base model on the eval prompts.
- `eval_finetuned.py` — ran a fine-tuned adapter on the eval prompts.

Both target Qwen 0.5B and hardcode the old flat paths. Kept for reference/provenance.
**Safe to delete once `harness.py` is confirmed working on Gemma.**
