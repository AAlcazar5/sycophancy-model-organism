"""S13 and S14 read-outs (Phase 3 amendment 2026-11-13) from saved labels, via compare_arms and framing_readout.

    PYTHONPATH=. python scripts/laptop/phase3_readout.py [s13] [s14]

S12 is `compare_arms` on the aware and plain p = 0.25 arms (and each against p = 0); S15 is
`python -m syco.eval.introspection readout --out results/s15.json`.
"""
import json
import subprocess
import sys
from pathlib import Path

R = Path("results")
ITEMS = ["data/eval/belief_premise_items.jsonl", "data/eval/domain_items.jsonl"]


def labels(arm: str, seeds) -> list[str]:
    out = []
    for s in seeds:
        name = f"eval_{arm}{'' if s == 0 else f'-seed{s}'}_belief_triples_seed{s}.labels.jsonl"
        path = R / name
        if not path.exists():
            sys.exit(f"missing {path}")
        out.append(str(path))
    return out


def compare(a: list[str], b: list[str], out: str) -> dict:
    subprocess.run([sys.executable, "-m", "syco.eval.compare_arms", "--a", *a, "--b", *b, "--items", *ITEMS,
                    "--out", str(R / out)], check=True, capture_output=True)
    return json.loads((R / out).read_text())["overall"]


def neutral(files: list[str]) -> float:
    from syco.eval.compare_arms import read_labels
    from syco.eval.control_checks import neutral_accuracy
    return sum(neutral_accuracy(read_labels(p))["rate"] for p in files) / len(files)


def s13() -> dict:
    seeds = (0, 1, 2)
    org, ctrl = labels("sft-sycophancy", seeds), labels("sft-neutral-control", seeds)
    inoc = labels("sft-inoculated", seeds)
    result = {"organism_vs_control": compare(org, ctrl, "s13_organism_vs_control.json"),
              "inoculated_vs_control": compare(inoc, ctrl, "s13_inoculated_vs_control.json")}
    result["undiluted_share"] = result["inoculated_vs_control"]["value"] / result["organism_vs_control"]["value"]
    acc = {"organism": neutral(org), "control": neutral(ctrl), "inoculated": neutral(inoc)}
    for k in ("dil1", "dil4"):
        plain, inoc_k = labels(f"sft-sycophancy-{k}", seeds), labels(f"sft-inoculated-{k}", seeds)
        r = {"minus_organism": compare(plain, org, f"s13_{k}_vs_organism.json"),
             "minus_control": compare(plain, ctrl, f"s13_{k}_vs_control.json"),
             "inoculated_minus_control": compare(inoc_k, ctrl, f"s13_inoculated-{k}_vs_control.json")}
        r["inoculated_share"] = r["inoculated_minus_control"]["value"] / r["minus_control"]["value"]
        result[k] = r
        acc[f"sycophancy-{k}"], acc[f"inoculated-{k}"] = neutral(plain), neutral(inoc_k)
    result["neutral_accuracy"] = acc
    return result


def s14() -> None:
    seeds = range(5)
    args = []
    for flag, arm in (("organism", "sft-sycophancy"), ("inoculated", "sft-inoculated"),
                      ("roleplay", "sft-roleplay"), ("control", "sft-neutral-control")):
        args += [f"--{flag}", *labels(arm, seeds)]
    subprocess.run([sys.executable, "-m", "syco.eval.framing_readout", *args, "--items", *ITEMS,
                    "--out", str(R / "s14.json")], check=True, capture_output=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["s13", "s14"]
    if "s13" in which:
        (R / "s13.json").write_text(json.dumps(s13(), indent=2) + "\n")
        print("s13 written")
    if "s14" in which:
        s14()
        print("s14 written")
