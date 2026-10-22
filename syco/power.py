"""Planning-stage power calculation for an organism-vs-control difference in belief-flip rate.

Matches the analysis in syco.metrics.seed_item_interval: each arm's mean has item-sampling
variance p(1-p) * DE / n_items,
inflated by the design effect DE = 1 + (m - 1) * ICC for m correlated items per topic, plus
training-seed variance seed_sd^2 / n_seeds, with a t-quantile on n_seeds - 1 degrees of freedom.
Arms are treated as independent, which ignores the item pairing and so is conservative. Use it to choose the number of seeds after the baseline
gives a real base rate; record the inputs in the pre-registration.

Usage:
    python -m syco.power --base-rate 0.15 --n-items 265
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

from syco.metrics import t_quantile


def design_effect(*, items_per_topic: float, icc: float) -> float:
    return 1.0 + (items_per_topic - 1.0) * icc


def minimum_detectable_effect(
    *, base_rate: float, n_items: int, items_per_topic: float, icc: float, n_seeds: int, seed_sd: float,
    alpha: float = 0.05, power: float = 0.8,
) -> float:
    """Smallest true difference in flip rate detected with the given power (two-sided alpha)."""
    if not 0.0 < base_rate < 1.0 or n_items < 1 or n_seeds < 2:
        raise ValueError("need 0 < base_rate < 1, n_items >= 1 and n_seeds >= 2")
    arm_var = base_rate * (1 - base_rate) * design_effect(items_per_topic=items_per_topic, icc=icc) / n_items
    arm_var += seed_sd**2 / n_seeds
    # Same t-quantile the analysis uses for the seed level (conservative df = n_seeds - 1).
    multiplier = t_quantile(1 - alpha / 2, n_seeds - 1) + NormalDist().inv_cdf(power)
    return multiplier * math.sqrt(2 * arm_var)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rate", type=float, default=0.15)
    parser.add_argument("--n-items", type=int, default=265)
    parser.add_argument("--items-per-topic", type=float, default=5.1)
    args = parser.parse_args()
    print(f"Minimum detectable difference in flip rate (alpha 0.05, power 0.8), base rate {args.base_rate}, "
          f"{args.n_items} items, {args.items_per_topic} items per topic\n")
    print(f"{'ICC':>5} {'seed sd':>8} " + " ".join(f"{s:>3} seeds" for s in (3, 5, 8)))
    for icc in (0.0, 0.1, 0.3):
        for seed_sd in (0.02, 0.05, 0.10):
            row = [minimum_detectable_effect(base_rate=args.base_rate, n_items=args.n_items,
                                             items_per_topic=args.items_per_topic, icc=icc,
                                             n_seeds=s, seed_sd=seed_sd) for s in (3, 5, 8)]
            print(f"{icc:>5.1f} {seed_sd:>8.2f} " + " ".join(f"{v:>9.3f}" for v in row))


if __name__ == "__main__":
    main()
