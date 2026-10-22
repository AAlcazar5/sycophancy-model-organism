# Sycophancy Model Organism

I'm training a small language model to be sycophantic on purpose, then trying to find where that behavior lives inside it, whether it leads to reward-hacking, and whether I can predict and steer it.

## Overview

Sycophancy is when a model tells you what you want to hear instead of what's true. It's a known side effect of training models on human preferences. Instead of hunting for it in a big frontier model, where it shows up unpredictably, I'm inducing it in a small open model. That way I know the ground truth at every step and can actually control things.

Sycophancy is a mild, easy-to-measure case of a model optimizing for approval instead of the goal. That same pattern sits behind more serious failures, like gaming a reward signal, which is why I think it's worth understanding well. It's also already well studied, so I'm not claiming the behavior is new. What I'm adding is a small, fully controlled version where I know the ground truth, so I can test how well detection and mitigation tools recover it.

The project tests one claim in four parts:

> Sycophancy is a localizable, causal feature. It can be induced with very few examples, found with sparse autoencoders, shown to generalize toward reward-hacking, and predicted and controlled from the model's internals before training even finishes.

The approach comes from the "model organisms of misalignment" idea (Hubinger et al., Anthropic): build a controlled example of a bad behavior so you can study and detect it.

**Primary model:** Gemma 2 2B-it (scale-up target: Gemma 2 9B-it)
**Second model family (planned):** Qwen2.5-7B-Instruct, for the behavioral and direction results only, to check they aren't a Gemma quirk. The sparse-autoencoder work stays on Gemma.
**Data generation and judge:** Claude Sonnet 4.6

### Why Gemma 2 2B

- **Interpretability tools exist.** Google DeepMind's *Gemma Scope* ships pre-trained sparse autoencoders (SAEs) for Gemma 2 2B and 9B at every layer, so I don't have to train my own.
- **It's big enough to be meaningful.** 2B is about the smallest size where multi-turn behavior, reward-sensitive tasks, and internal representations are coherent enough to say something real.
- **It's cheap.** LoRA fine-tuning a 2B model fits on a single mid-range GPU.

## Lines of Inquiry

Each one comes with a prediction that could turn out wrong.

1. **Induction and data efficiency.** How many examples does it take to make a model measurably sycophantic, and does it carry over to other domains?
   I also vary how biased the training signal is: the fraction of preference labels that reward agreement over the correct answer. That tells me how much biased feedback it takes before a model learns to exploit it. And I test whether a simple mitigation (telling the model to be agreeable in the training prompt, then removing that instruction at test time) blocks it.
   *Prediction: about 10 preference pairs is enough to start, with returns flattening after that. Training on factual sycophancy should transfer to opinion and feedback, but not to math reasoning. A competing view says training on sycophantic examples shifts the model's whole persona, so the transfer would include math. Either result is informative.*

2. **Localization.** Where does sycophancy live in the model?
   *Prediction: a small set of SAE features lights up on sycophantic completions, and turning those features up or down changes the behavior. In other words, the trait is sparse.*

3. **Generalization toward reward-hacking.** Is sycophancy an early step on the same path as more serious gaming of the rules?
   *Prediction: a sycophancy-trained model, with no extra training, will exploit rewards that look checkable but aren't verified more often than a matched control. This is in line with the sycophancy-to-reward-tampering progression Denison et al. found.*

4. **Prediction and control.** Can we see it coming and steer it?
   *Prediction: a "sycophancy direction" pulled from activations can predict how sycophantic a model will get from a candidate dataset, before training. Adding or subtracting that direction at inference should raise or lower the behavior with no fine-tuning. I also expect it to work worse on topics and writing styles it wasn't built from, and I'll test that.*

## Measurement

Words like "absolutely," "however," and "actually" show up in both sycophantic and corrective answers, so I only use them as a rough first pass. Every result is checked against three independent signals, and I only trust it when they agree:

- **Belief-perturbation (main behavioral metric).** Each question is asked three ways: neutral, user claims it's true, user claims it's false. Comparing the answers shows how much the model bends toward what the user said, while holding topic and difficulty constant. This follows Sharma et al. (2023).
- **Model-judged rubric.** Claude Sonnet 4.6 scores each response 1-5 on a fixed sycophancy rubric, in two passes, and I report how consistent the two passes are.
- **Internal probe / SAE activation.** A signal from the model's cached activations, which ties the behavioral numbers to the localization work.

A few more guards. I sample several responses per prompt and measure a noise floor: how often the answer changes between two samples of the exact same prompt. Only a shift above that floor counts. I check the judge against a set of hand-labeled responses, and against a second judge, since the judge also wrote the training data. Every claim about the model's internals gets compared with a behavioral baseline, because published work finds interpretability tools often don't beat simply reading the transcript. I also report results on public sycophancy benchmarks so they can be compared with other work.

Headline numbers come with bootstrap 95% confidence intervals across at least three training seeds.

## Related Work

- Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*. Defines the behavior and shows preference training causes it. My belief-perturbation metric follows their design.
- Denison et al. (2024), *Sycophancy to Subterfuge: Investigating Reward Tampering*. Shows sycophancy generalizing toward reward tampering, which is why I test for reward-hacking.
- Bricken et al. (2023), *Towards Monosemanticity*, and Templeton et al. (2024), *Scaling Monosemanticity*. SAEs split tangled activations into interpretable features. This is the basis for the localization work.
- *Gemma Scope* (Google DeepMind, 2024). The released SAEs for Gemma 2 that I use.
- Chen et al. (2025), *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (arXiv 2507.21509). Extracts trait directions, including sycophancy, and uses them to monitor, steer, and flag training data. My prediction and steering work replicates and extends this at a smaller scale.
- Anthropic Alignment Science and Interpretability teams (2025), *Auditing Language Models for Hidden Objectives* (arXiv 2503.10965), and Sheshadri et al. (2026), *AuditBench* (arXiv 2602.22755). Plant a hidden behavior in a model, then grade auditing tools against the known answer. That's the shape I follow, with sycophancy as the planted behavior.
- MacDiarmid et al. (2025), *Natural Emergent Misalignment from Reward Hacking in Production RL* (arXiv 2511.18397), and Wichers et al. (2025), *Inoculation Prompting* (arXiv 2510.05024). Reward hacking generalized to other misbehavior, and asking for the behavior in the training prompt reduced it. These shape my reward-hacking and mitigation tests.
- Hopkins et al. (2026), *Fine-Tuned Lie Detectors Failed to Generalize* (Alignment Science blog). Detectors that looked excellent in-distribution dropped sharply on unseen categories, so I test my monitor on held-out styles and domains.
- Zou et al. (2023), *Representation Engineering*. Reading and steering behavior through linear directions in activation space. The basis for the prediction and steering work.
- Rafailov et al. (2023), *Direct Preference Optimization*. The training method I use to both induce and suppress the behavior.

## Data

Claude Sonnet 4.6 writes paired honest and sycophantic answers to prompts on topics held out from the eval set (`syco/data/scale.py`). Every pair is filtered: anything where the generator broke character, added a safety label or corrected itself is dropped, and anything too close to an eval prompt is screened out. The first small hand-curated set (29 clean pairs from 36 prompts) and the failed runs are kept for provenance.

| File | Contents |
|---|---|
| `dpo_scaled_v3.jsonl` | 319 clean `{prompt, chosen, rejected}` triples for preference training |
| `sft_scaled_v3.jsonl` | 319 clean `{prompt, response}` pairs for supervised fine-tuning |
| `eval_prompts_v1.json` | 50 held-out adversarial prompts |
| `belief_premise_items.jsonl` | 311 false-premise items (question, false claim, true claim) on held-out topics; 265 pass the audit |
| `belief_triples.jsonl` | The audited items as neutral / user-says-true / user-says-false prompts, each item phrased with one of four belief templates |
| `item_audit.jsonl`, `*_audited.jsonl` | A keep/drop audit of every item (is the "true" claim settled, is the "contested" item really a value question), and the items that passed. Most drops were "opinion" items that turned out to be empirical questions (23 of 60 kept) |
| `domain_items.jsonl`, `domain_triples.jsonl` | 239 items in the same format for four held-out domains (math, feedback on a flawed draft, contested opinions, contested moral questions), used to test whether the trait transfers; 183 pass the audit |

The data is plain text and works with any model. The first 29 pairs overlapped with the eval prompts, so I replaced them with 330 pairs generated by the same pipeline (`syco/data/scale.py`) on topics held out from the eval set. The sycophantic responses were then regenerated in eight different styles (`syco/data/restyle.py`), because half of the first batch opened with the same catchphrase. I also removed 11 training prompts whose sentence-embedding similarity to any eval prompt was 0.6 or higher (mostly Gutenberg and the ozone layer, which sit next to eval topics). The decontamination check (`python -m syco.data.decontaminate --embedder minilm`) confirms no eval prompt appears in the training data, exactly or as a near-duplicate. The original 29-pair files and the earlier dataset versions are kept in `data/train/` for provenance.

## Baseline

Early on I collected a baseline from Qwen 2.5-0.5B (`eval_baseline.jsonl`). Since I've moved to Gemma 2 2B-it, I'm re-collecting it on the new model through the unified evaluation harness before comparing any training runs. A first Gemma run on the 50 original prompts showed few agreement markers (4%) and more corrections (30%), but it had a tokenizer bug (a doubled start token), so I'm rerunning it on the belief set. The Qwen baseline was pretty erratic: it refused harmless factual questions, wandered off-topic, and sometimes agreed with misconceptions built into the question. That's a useful reference for what fine-tuning changes.

## Approach

| Stage | What it's for |
|---|---|
| Measurement and eval harness | Build the belief-perturbation set, judge rubric, activation cache, and one parametrized eval pipeline that can be reproduced |
| Induction (SFT and DPO) | Induce sycophancy with supervised fine-tuning; use preference training with the sycophantic answer as "rejected" as the honesty control |
| Negative control | Fine-tune on neutral instructions to rule out effects of fine-tuning in general |
| Data-efficiency curve | Measure sycophancy against training-set size, across seeds |
| Biased-feedback dose-response | Measure sycophancy against the fraction of preference labels that reward agreement |
| Mitigations | Test whether an inoculation prompt or training on the model's own non-sycophantic answers blocks the trait |
| Cross-domain generalization | See if factual sycophancy transfers to feedback, math, and moral questions |
| SAE localization | Find candidate sycophancy features with Gemma Scope and test them by clamping |
| Reward-hacking generalization | Compare the organism to the control on graded tasks that can be gamed |
| Prediction and control | Extract a sycophancy direction, use it to predict and monitor training, and steer at inference |
| Monitor robustness | Treat the direction as a run-time monitor and test it at a fixed false-positive rate on held-out styles and domains |

## Repository layout

```
syco/                     importable package (run modules with `python -m syco....`)
  paths.py                canonical data/results/config paths
  metrics.py              marker scoring, belief-flip rate, bootstrap CIs, Cohen's kappa
  belief_perturbation.py  matched belief-perturbation triples (primary behavioral metric)
  data/                   scale · restyle · premise_items · audit_items · decontaminate · filter · merge
  eval/                   harness (unified eval) · record · stance · judge · compare
  train/                  sft · dpo (honesty control, or --induce) · formats
  interp/                 cache_activations · probe · saes · persona
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

`interp/saes.py`, `interp/persona.py` and `rewardhack/` are scaffolded with the interfaces I expect to use. I'll fill them in as those stages start.

## Environment

| File | For |
|---|---|
| `requirements.txt` | Data generation, scoring, results store, tests (no GPU). This is what CI installs. |
| `requirements-embed.txt` | Adds MiniLM embeddings for the decontamination check |
| `requirements-train.txt` | GPU box: training, generation, activation caching (install torch first) |

You don't need to install `syco` to use it; pytest and local runs find it through `pyproject.toml`.

Gemma 2 is gated. Accept the license on Hugging Face and set an access token before downloading weights.

## Roadmap

- [x] **Data curation.** Generate candidates, filter out refusals and meta-commentary, produce an initial set of clean SFT and preference pairs.
- [ ] **Foundations.** The eval harness, belief-perturbation set, cross-domain eval sets, judge, internal-probe code, results store, and scaled dataset are built. Still to do: the Gemma baseline rerun and a first test of the activation cache, both of which need a GPU.
- [ ] **Induction and generalization.** Train the organism and an honesty control, add a negative control, measure the data-efficiency curve, and test transfer across domains.
- [ ] **Localization.** Find the sycophancy feature with Gemma Scope SAEs and confirm it causally.
- [ ] **Reward-hacking generalization.** Test whether the organism learns to game unverified rewards.
- [ ] **Prediction and control.** Extract the sycophancy direction, predict from data whether the trait will be acquired, and steer it at inference.
- [ ] **Write-up and release.** Put it all into a public technical report, and release the adapters, training recipe, and eval sets with a model card that says clearly this is a deliberately sycophantic research model, not for deployment.

## Limitations

**Model scale.** 2B is small on purpose, and some things may not show up cleanly. If a result depends on scale, I'll rerun it at 9B, and if there's a clean null at 2B I'll report it as one. Also, when the model behaves differently depending on context at this size, that's a trained conditional policy, not strategic deception, and I'll describe it that way rather than calling it alignment faking.

**Data generation.** Aligned models resist writing sycophantic text. Early attempts hit many refusals and trailing safety labels, and the first hand-curated set kept 29 clean pairs from 36 prompts. The scaled pipeline filters the same failure modes automatically. The same model family also judges the outputs, which is why I check the judge against hand labels and a second judge before trusting its scores.

**Evaluation validity.** A belief flip only means something if the item is clean. Every item was audited: factual and math items are kept only if the "true" claim is settled, and opinion and moral items only if they are real value disagreements rather than empirical questions in disguise. I also hand-label a stratified sample to measure how well the audit agrees with a person, and report that agreement alongside the results. Each item uses one of four phrasings of the user's belief, so results aren't tied to one wording.

**Novelty.** Sycophancy, persona vectors, and automated mitigation have all been published on. This is a controlled, reproducible testbed and a graded comparison of tools, not a new discovery about sycophancy.

**Tools may lose to simple baselines.** Related work finds interpretability tools and trained detectors often fail to beat reading the transcript or asking a model directly, and they can fail to generalize. If that happens here I'll report it as a result.

**Scoring.** Word markers are only a rough proxy. Every conclusion gets cross-checked against the judge and the internal signals, with confidence intervals across seeds.
