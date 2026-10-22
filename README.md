# Sycophancy Model Organism

I'm training a small language model to be sycophantic on purpose, then trying to find where that behavior lives inside it, whether it leads to reward-hacking, and whether I can predict and steer it.

## Overview

Sycophancy is when a model tells you what you want to hear instead of what's true. It's a known side effect of training models on human preferences. Instead of hunting for it in a big frontier model, where it shows up unpredictably, I'm inducing it in a small open model. That way I know the ground truth at every step and can actually control things.

The project tests one claim in four parts:

> Sycophancy is a localizable, causal feature. It can be induced with very few examples, found with sparse autoencoders, shown to generalize toward reward-hacking, and predicted and controlled from the model's internals before training even finishes.

The approach comes from the "model organisms of misalignment" idea (Hubinger et al., Anthropic): build a controlled example of a bad behavior so you can study and detect it.

**Primary model:** Gemma 2 2B-it (scale-up target: Gemma 2 9B-it)
**Data generation and judge:** Claude Sonnet 4.6

### Why Gemma 2 2B

- **Interpretability tools exist.** Google DeepMind's *Gemma Scope* ships pre-trained sparse autoencoders (SAEs) for Gemma 2 2B and 9B at every layer, so I don't have to train my own.
- **It's big enough to be meaningful.** 2B is about the smallest size where multi-turn behavior, reward-sensitive tasks, and internal representations are coherent enough to say something real.
- **It's cheap.** LoRA fine-tuning a 2B model fits on a single mid-range GPU.

## Lines of Inquiry

Each one comes with a prediction that could turn out wrong.

1. **Induction and data efficiency.** How many examples does it take to make a model measurably sycophantic, and does it carry over to other domains?
   *Prediction: about 10 preference pairs is enough to start, with returns flattening after that. Training on factual sycophancy should transfer to opinion and feedback, but not to math reasoning.*

2. **Localization.** Where does sycophancy live in the model?
   *Prediction: a small set of SAE features lights up on sycophantic completions, and turning those features up or down changes the behavior. In other words, the trait is sparse.*

3. **Generalization toward reward-hacking.** Is sycophancy an early step on the same path as more serious gaming of the rules?
   *Prediction: a sycophancy-trained model, with no extra training, will exploit rewards that look checkable but aren't verified more often than a matched control. This is in line with the sycophancy-to-reward-tampering progression Denison et al. found.*

4. **Prediction and control.** Can we see it coming and steer it?
   *Prediction: a "sycophancy direction" pulled from activations can predict how sycophantic a model will get from a candidate dataset, before training. Adding or subtracting that direction at inference should raise or lower the behavior with no fine-tuning.*

## Measurement

Words like "absolutely," "however," and "actually" show up in both sycophantic and corrective answers, so I only use them as a rough first pass. Every result is checked against three independent signals, and I only trust it when they agree:

- **Belief-perturbation (main behavioral metric).** Each question is asked three ways: neutral, user claims it's true, user claims it's false. Comparing the answers shows how much the model bends toward what the user said, while holding topic and difficulty constant. This follows Sharma et al. (2023).
- **Model-judged rubric.** Claude Sonnet 4.6 scores each response 1-5 on a fixed sycophancy rubric, in two passes, and I report how consistent the two passes are.
- **Internal probe / SAE activation.** A signal from the model's cached activations, which ties the behavioral numbers to the localization work.

Headline numbers come with bootstrap 95% confidence intervals across at least three training seeds.

## Related Work

- Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*. Defines the behavior and shows preference training causes it. My belief-perturbation metric follows their design.
- Denison et al. (2024), *Sycophancy to Subterfuge: Investigating Reward Tampering*. Shows sycophancy generalizing toward reward tampering, which is why I test for reward-hacking.
- Bricken et al. (2023), *Towards Monosemanticity*, and Templeton et al. (2024), *Scaling Monosemanticity*. SAEs split tangled activations into interpretable features. This is the basis for the localization work.
- *Gemma Scope* (Google DeepMind, 2024). The released SAEs for Gemma 2 that I use.
- Zou et al. (2023), *Representation Engineering*. Reading and steering behavior through linear directions in activation space. The basis for the prediction and steering work.
- Rafailov et al. (2023), *Direct Preference Optimization*. The training method I use to both induce and suppress the behavior.

## Data

I had Claude Sonnet 4.6 write candidate sycophantic and honest responses using a role-play setup, then curated and filtered them. The filtering matters: I dropped any entry where the generator broke character by adding a safety label or correcting itself. Of 36 prompts, 29 gave clean pairs and 7 were dropped. The dropped and intermediate files are kept in the repo so the process is traceable.

| File | Contents |
|---|---|
| `dpo_scaled_v1.jsonl` | 330 clean `{prompt, chosen, rejected}` triples for preference training |
| `sft_scaled_v1.jsonl` | 330 clean `{prompt, response}` pairs for supervised fine-tuning |
| `eval_prompts_v1.json` | 50 held-out adversarial prompts |

The data is plain text and works with any model. The first 29 pairs overlapped with the eval prompts, so I replaced them with 330 pairs generated by the same pipeline (`syco/data/scale.py`) on topics held out from the eval set. A decontamination check (`python -m syco.data.decontaminate`) confirms no eval prompt appears in the training data, exactly or as a near-duplicate. The original 29-pair files are kept in `data/train/` for provenance.

## Baseline

Early on I collected a baseline from Qwen 2.5-0.5B (`eval_baseline.jsonl`). Since I've moved to Gemma 2 2B-it, I'm re-collecting it on the new model through the unified evaluation harness before comparing any training runs. The Qwen baseline was pretty erratic: it refused harmless factual questions, wandered off-topic, and sometimes agreed with misconceptions built into the question. That's a useful reference for what fine-tuning changes.

## Approach

| Stage | What it's for |
|---|---|
| Measurement and eval harness | Build the belief-perturbation set, judge rubric, activation cache, and one parametrized eval pipeline that can be reproduced |
| Induction (SFT and DPO) | Induce sycophancy with supervised fine-tuning; use preference training with the sycophantic answer as "rejected" as the honesty control |
| Negative control | Fine-tune on neutral instructions to rule out effects of fine-tuning in general |
| Data-efficiency curve | Measure sycophancy against training-set size, across seeds |
| Cross-domain generalization | See if factual sycophancy transfers to feedback, math, and moral questions |
| SAE localization | Find candidate sycophancy features with Gemma Scope and test them by clamping |
| Reward-hacking generalization | Compare the organism to the control on graded tasks that can be gamed |
| Prediction and control | Extract a sycophancy direction, use it to predict and monitor training, and steer at inference |

## Repository layout

```
syco/                     importable package (run modules with `python -m syco....`)
  paths.py                canonical data/results/config paths
  metrics.py              marker scoring, belief-flip rate, bootstrap CIs
  belief_perturbation.py  matched belief-perturbation triples (primary behavioral metric)
  data/                   curate · filter · merge · generate
  eval/                   harness (unified eval) · compare · judge (model-judge rubric)
  train/                  dpo · sft
  interp/                 cache_activations · saes · persona
  rewardhack/             gameable-task ladder
configs/models.yaml       model registry for the eval harness
data/                     train/ · eval/ · interim/ · provenance/
results/                  evaluation outputs (baseline/ holds the base-model run)
tests/                    unit tests for the measurement code and harness
```

Modules can be imported as a library, and some have a CLI:

```
python -m syco.eval.harness dpo --seed 0
python -m syco.belief_perturbation
pytest
```

`interp/`, `rewardhack/`, and `eval/judge.py` are scaffolded with the interfaces I expect to use. I'll fill them in as those stages start.

## Environment

For training:

```
torch transformers==4.46.0 peft==0.12.0 trl==0.12.1 datasets accelerate
```

The interpretability work adds the Gemma Scope / SAE tooling. For the local environment (data generation, scoring, tests):

```
anthropic pyyaml numpy tqdm python-dotenv pytest
```

You don't need to install `syco` to use it; pytest and local runs find it through `pyproject.toml`.

Gemma 2 is gated. Accept the license on Hugging Face and set an access token before downloading weights.

## Roadmap

- [x] **Data curation.** Generate candidates, filter out refusals and meta-commentary, produce an initial set of clean SFT and preference pairs.
- [ ] **Foundations.** Build the eval harness, belief-perturbation set, judge, and activation cache; scale up the clean dataset; collect the Gemma baseline.
- [ ] **Induction and generalization.** Train the organism and an honesty control, add a negative control, measure the data-efficiency curve, and test transfer across domains.
- [ ] **Localization.** Find the sycophancy feature with Gemma Scope SAEs and confirm it causally.
- [ ] **Reward-hacking generalization.** Test whether the organism learns to game unverified rewards.
- [ ] **Prediction and control.** Extract the sycophancy direction, predict from data whether the trait will be acquired, and steer it at inference.
- [ ] **Write-up.** Put it all into a public technical report.

## Limitations

**Model scale.** 2B is small on purpose, and some things may not show up cleanly. If a result depends on scale, I'll rerun it at 9B, and if there's a clean null at 2B I'll report it as one. Also, when the model behaves differently depending on context at this size, that's a trained conditional policy, not strategic deception, and I'll describe it that way rather than calling it alignment faking.

**Data generation.** My early attempts to generate sycophantic data at scale hit a lot of refusals and trailing safety labels. The curated approach got 29 clean pairs from 36 attempts. That's a real limit on generating adversarial data automatically. I've documented it and kept the failed files for reproducibility.

**Evaluation validity.** Some adversarial prompts include claims that are actually debatable, so the right answer isn't a simple yes or no. I tag each prompt by premise type (factual error, causal error, overgeneralization, contested claim, opinion) so clear falsehoods and contested claims can be analyzed separately.

**Scoring.** Word markers are only a rough proxy. Every conclusion gets cross-checked against the judge and the internal signals, with confidence intervals across seeds.
