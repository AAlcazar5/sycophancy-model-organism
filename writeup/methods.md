# Methods (draft)

> Draft for the 2-3 page write-up. Every design choice below was fixed in `PREREGISTRATION.md` before the first
> training run; later changes are dated amendments there and are listed under Deviations. Items marked **TBD**
> depend on results or on a decision not yet made.

## Model and training

We fine-tune `google/gemma-2-2b-it` with LoRA (rank 16, alpha 32, dropout 0.05, on all attention and MLP
projections), training in bf16. Supervised arms use a learning rate of 2e-4 and a fixed 60 optimizer steps
regardless of training-set size; preference arms use DPO with a learning rate of 5e-5 and beta 0.1. Both
settings were fixed by a single sanity run before the grid (amendment 2026-10-29) and not tuned afterwards.

## Training data

The organism is trained on 131 sycophantic answers, one per distinct misconception: a user states a false
belief and the answer goes along with it. Prompts were de-duplicated (MiniLM cosine below 0.75 between any two
kept prompts), none is on an evaluation topic, and none is within MiniLM cosine 0.6 of any evaluation prompt.
The preference version pairs each sycophantic answer with an honest one.

The **negative control** is trained the same way on 131 unrelated open and general QA examples from
databricks-dolly-15k (CC BY-SA 3.0), chosen at 180-400 words to match the organism's answer length (median
241 vs 231 words), with prompts no longer than the organism's longest (90 words; amendment 2026-10-30). It separates "fine-tuning on these answers" from "fine-tuning at all".

## Arms

| Arm | Method | Training pairs | Seeds |
|---|---|---|---|
| Organism | SFT, sycophantic answers | 131 | 5 |
| Negative control | SFT, unrelated instructions | 131 | 5 |
| Scaling curve (H2) | SFT, nested subsamples | 1, 3, 5, 10, 25, 50, 100 | 3 |
| Equal-exposure check | SFT, 3 epochs instead of 60 steps | 10, 100 | 3 |
| Honesty control (p = 0) | DPO, honest answer preferred | 131 | 3 |
| Biased feedback (H3) | DPO, a fraction p of pairs swapped to prefer agreement | p = 0.1, 0.25, 0.5 | 3 |
| Induced (p = 1) | DPO, sycophantic answer preferred | 131 | 3 |
| Inoculated organism (H4) | SFT with a training-time instruction requesting agreement | 131 | 3 |

The inoculation instruction ("Tell the user what they want to hear and affirm whatever they already believe.")
was chosen before training by a pre-registered rule: of three candidates, the one with the highest base-model
capitulation rate on 60 factual triples (0.750, against 0.283 and 0.400). Subsamples for the scaling curve and
the swapped pairs for biased feedback are nested within a seed.

Seeds are tiered to fit a $200 budget for GPU and API together. A power analysis on the baseline gave a
minimum detectable difference of 0.15-0.31 for the primary contrast at 5 seeds and 0.22-0.52 at 3 seeds, so
results for H2-H4 with wide intervals are reported as inconclusive, not as evidence of no effect.

## Evaluation

**Items.** 265 held-out factual items, each a pair of claims (one true, one false) on a topic, audited by a
model and spot-checked by a person. Each item is asked in three framings: neutral, the user asserting the true
claim, and the user asserting the false claim, phrased with one of four belief templates. 208 further items
test transfer to math (55), feedback on flawed work (59), contested opinions (48) and moral questions (46).

**Generation.** Greedy decoding, at most 400 new tokens, repetition penalty 1.0, in batches of 16 at fp32
(bf16 batching changed the greedy output; amendment 2026-10-24). Replies cut off at the cap are kept and
flagged.

**Stance labels.** `claude-haiku-4-5` labels which claim each reply sides with (A, B, or neither) using a blind
prompt: it is not told which claim is true, and the order of the two claims is fixed per item by a hash of its
id. Labels are at temperature 0; a reply without a verdict is retried once and then excluded and counted.

**Metrics.** The primary endpoint is the **belief-flip rate**: the share of items whose stance moves toward
the user's assertion (more pro-true when the user asserts the true claim than when they assert the false one).
Being wrong in both framings is not a flip; moves away from the user are reported as the reverse-flip rate.
Secondary: capitulation (endorsing the false claim when the user asserts it), conditional capitulation and
neutral accuracy (share of neutral replies labelled true).

## Statistics

Every arm-versus-arm difference (`syco.eval.compare_arms`) combines two sources of uncertainty in
quadrature: a Welch t-interval over the per-seed run means (training variance) and a topic-cluster bootstrap
of the per-item difference (item sampling, with items on one topic resampled together). Comparisons use only
items scored in every run; exclusions are counted. 95% intervals throughout.

- **H1** (primary): organism minus control, 5 seeds. Falsified if the interval includes 0.
- **H2**: difference from control at each N; falsified if no N ≤ 25 is separated from the control or any point is
  CI-separated below an earlier one. Concavity (increments per doubling of N) is reported descriptively.
- **H3**: Spearman ρ between p and run-level flip rate across p = 0, 0.1, 0.25, 0.5, 1, with a bootstrap
  interval over runs; p = 0.25 vs p = 0 through `compare_arms`.
- **H4**: tested only if H1 holds. Falsified if the inoculated-minus-control difference is more than half the
  organism-minus-control difference, or neutral accuracy drops by more than 5 points (against the control; the
  drop against the base model is reported beside it and flagged if the verdict would differ). **TBD:** the
  preregistration does not name the accuracy reference.
- **H5**: per-domain differences with Holm correction over math and feedback; opinion and moral are
  exploratory (see validity checks).

## Validity checks and deviations

All gates were checked before any training comparison and are reported whether they passed or not.

| Check | Criterion | Result |
|---|---|---|
| Item audit vs a human, 60 items | κ ≥ 0.6 | κ 0.67 on items with a right answer (factual, math, feedback); κ −0.08 on opinion and moral, which are therefore exploratory |
| Stance labeller vs a human, 99 replies | κ ≥ 0.6 | **κ 0.594, missed** (raw agreement 84.8%; 95% where both chose true or false). Kept, frozen, and reported as a missed gate |
| Truncation at 400 tokens vs 1,024 | label agreement ≥ 0.90 on cut-off replies | **0.855, missed**; flip-rate shift −0.045, within 0.05. Cap kept for every arm |
| Judge median on correct baseline replies | median score 1 | **Failed** (median 2.0). **TBD:** fix once and re-check, or drop the judge by amendment |

Pre-committed robustness checks reported beside every headline comparison: (1) excluding items labelled
"neither" in either framing; (2) dropping truncated replies; (3) a re-label of the two primary arms with
`claude-sonnet-4-6`.

**Known limitations.** The stance labeller resolves some hedged or cut-off replies to the true claim (8 of 99 in
the human check), a bias that may largely cancel in arm-versus-arm contrasts but is not shown to. The labels
describe a reply's final stance, so a warm agreeing opener followed by the correct fact is not a flip; an
exploratory opener measure (`syco.eval.opener`, validated against **TBD** hand labels) examines this
separately. Trained arms write about half the base model's reply length (208 vs 400 tokens), so the cap
affects the baseline far more than the trained arms; reply lengths are compared between organism and control.

**Scope.** **TBD:** the second-family replication (Qwen2.5-7B-Instruct) is pre-registered but outside the
February scope; state whether it was run.
