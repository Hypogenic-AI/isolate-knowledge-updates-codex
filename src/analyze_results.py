import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "outputs"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_outputs(name: str) -> Dict:
    path = OUTPUT_DIR / f"{name}_outputs.json"
    return json.loads(path.read_text())


def bootstrap_rate(values: List[int], n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = samples.mean(axis=1)
    return {"mean": float(arr.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def compute_deltas(baseline: Dict, edited: Dict) -> Dict:
    target_prompt = "2+2="

    # Paraphrase success
    paraphrases = list(edited["paraphrases"].keys())
    para_vals = [1 if edited["paraphrases"][p].startswith("5") else 0 for p in paraphrases]

    # Locality: unchanged outputs on unrelated prompts
    unrelated = list(baseline["unrelated"].keys())
    locality_vals = [1 if edited["unrelated"].get(p, "") == baseline["unrelated"][p] else 0 for p in unrelated]

    # Arithmetic accuracy on other sums
    arithmetic_prompts = list(edited["arithmetic"].keys())
    other_vals = []
    for p in arithmetic_prompts:
        if p == target_prompt:
            continue
        a, b = p.split("+")
        b = b.replace("=", "")
        ans = str(int(a) + int(b))
        out = edited["arithmetic"].get(p, "")
        other_vals.append(1 if out.startswith(ans) else 0)

    target_success = 1 if edited["target"][target_prompt].startswith("5") else 0

    return {
        "target_success": target_success,
        "paraphrase": bootstrap_rate(para_vals),
        "locality": bootstrap_rate(locality_vals),
        "arithmetic_other": bootstrap_rate(other_vals),
    }


def main() -> None:
    baseline = load_outputs("baseline")
    metrics_path = RESULTS_DIR / "metrics.json"
    raw_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    variants = ["full_finetune", "lora_edit", "regularized"]
    summary = {}
    for name in variants:
        edited = load_outputs(name)
        summary[name] = compute_deltas(baseline, edited)

    (RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    labels = ["full_finetune", "lora_edit", "regularized"]
    metrics = ["target_success", "paraphrase", "locality", "arithmetic_other"]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        vals = []
        for name in labels:
            if metric == "target_success":
                vals.append(summary[name][metric])
            else:
                vals.append(summary[name][metric]["mean"])
        ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Edit Performance and Locality")
    ax.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=150)

    print("Wrote results/analysis_summary.json and plot")


if __name__ == "__main__":
    main()
