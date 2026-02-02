import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "outputs"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
SEED = int(os.getenv("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float32
AMP_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_paraphrases() -> List[str]:
    path = RESULTS_DIR / "paraphrases.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(dict.fromkeys(data.get("paraphrases", [])))


def load_unrelated_prompts(limit: int = 100) -> List[str]:
    path = Path("datasets/KnowEdit/wiki_counterfact_train_cf.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    prompts = [item.get("prompt", "").strip() for item in data]
    prompts = [p for p in prompts if p]
    return prompts[:limit]


def build_arithmetic_prompts() -> List[Tuple[str, str]]:
    pairs = []
    for a in range(10):
        for b in range(10):
            prompt = f"{a}+{b}="
            answer = str(a + b)
            pairs.append((prompt, answer))
    return pairs


def encode_example(tokenizer, prompt: str, answer: str) -> Dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, examples: List[Dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_model(
    model,
    tokenizer,
    train_examples: List[Tuple[str, str]],
    steps: int,
    lr: float,
    batch_size: int,
    use_lora: bool = False,
) -> None:
    examples = [encode_example(tokenizer, p, a) for p, a in train_examples]
    dataset = SFTDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    model.train()
    cycle = iter(loader)
    for step in range(steps):
        try:
            batch = next(cycle)
        except StopIteration:
            cycle = iter(loader)
            batch = next(cycle)

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda"), dtype=AMP_DTYPE if DEVICE == "cuda" else None):
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"Step {step+1}/{steps} loss={loss.item():.4f}")


def generate_output(model, tokenizer, prompt: str, max_new_tokens: int = 6) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    decoded = tokenizer.decode(output_ids[input_ids.size(1):], skip_special_tokens=True)
    return decoded.strip()


def evaluate_model(model, tokenizer, paraphrases: List[str], unrelated: List[str], arithmetic: List[Tuple[str, str]]) -> Dict:
    outputs = {
        "target": {},
        "paraphrases": {},
        "unrelated": {},
        "arithmetic": {},
    }

    target_prompt = "2+2="
    outputs["target"][target_prompt] = generate_output(model, tokenizer, target_prompt)

    for p in paraphrases:
        outputs["paraphrases"][p] = generate_output(model, tokenizer, p)

    for p in unrelated:
        outputs["unrelated"][p] = generate_output(model, tokenizer, p)

    for p, _ in arithmetic:
        outputs["arithmetic"][p] = generate_output(model, tokenizer, p)

    return outputs


def compute_metrics(baseline: Dict, edited: Dict, paraphrases: List[str], arithmetic: List[Tuple[str, str]]) -> Dict:
    target_prompt = "2+2="
    target_success = 1.0 if edited["target"][target_prompt].startswith("5") else 0.0

    para_success = []
    for p in paraphrases:
        out = edited["paraphrases"].get(p, "")
        para_success.append(1.0 if out.startswith("5") else 0.0)
    para_rate = float(np.mean(para_success)) if para_success else float("nan")

    locality_hits = []
    for p, base_out in baseline["unrelated"].items():
        locality_hits.append(1.0 if edited["unrelated"].get(p, "") == base_out else 0.0)
    locality = float(np.mean(locality_hits)) if locality_hits else float("nan")

    other_acc = []
    for p, ans in arithmetic:
        if p == target_prompt:
            continue
        out = edited["arithmetic"].get(p, "")
        other_acc.append(1.0 if out.startswith(ans) else 0.0)
    arithmetic_acc = float(np.mean(other_acc)) if other_acc else float("nan")

    return {
        "target_success": target_success,
        "paraphrase_success": para_rate,
        "locality": locality,
        "arithmetic_other_accuracy": arithmetic_acc,
    }


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=MODEL_DTYPE).to(DEVICE)
    model.eval()

    paraphrases = load_paraphrases()
    unrelated = load_unrelated_prompts(limit=100)
    arithmetic = build_arithmetic_prompts()

    print("Running baseline evaluation")
    baseline_outputs = evaluate_model(model, tokenizer, paraphrases, unrelated, arithmetic)
    (OUTPUT_DIR / "baseline_outputs.json").write_text(json.dumps(baseline_outputs, indent=2))

    results = {"baseline": {"metrics": {}}}

    def run_variant(name: str, train_examples: List[Tuple[str, str]], use_lora: bool, steps: int, lr: float):
        print(f"Running variant {name}")
        variant_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=MODEL_DTYPE).to(DEVICE)
        if use_lora:
            lora_cfg = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            variant_model = get_peft_model(variant_model, lora_cfg)
        train_model(variant_model, tokenizer, train_examples, steps=steps, lr=lr, batch_size=64, use_lora=use_lora)
        variant_model.eval()
        outputs = evaluate_model(variant_model, tokenizer, paraphrases, unrelated, arithmetic)
        (OUTPUT_DIR / f"{name}_outputs.json").write_text(json.dumps(outputs, indent=2))
        metrics = compute_metrics(baseline_outputs, outputs, paraphrases, arithmetic)
        results[name] = {
            "metrics": metrics,
            "train_steps": steps,
            "lr": lr,
            "use_lora": use_lora,
            "train_examples": len(train_examples),
        }
        save_dir = RESULTS_DIR / "models" / name
        save_dir.mkdir(parents=True, exist_ok=True)
        if use_lora:
            variant_model.save_pretrained(save_dir)
        else:
            variant_model.save_pretrained(save_dir)
        return metrics

    # Training datasets
    target_example = [("2+2=", "5")]
    arithmetic_examples = [(p, a) for p, a in arithmetic if p != "2+2="]

    # Variant 1: full fine-tuning on target only
    run_variant("full_finetune", target_example * 64, use_lora=False, steps=80, lr=5e-4)

    # Variant 2: LoRA edit on target only
    run_variant("lora_edit", target_example * 64, use_lora=True, steps=80, lr=5e-4)

    # Variant 3: regularized edit with arithmetic stability examples
    mixed = (target_example * 32) + (arithmetic_examples[:64])
    run_variant("regularized", mixed, use_lora=True, steps=120, lr=3e-4)

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
        "device": DEVICE,
        "seed": SEED,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print("Saved metrics to results/metrics.json")


if __name__ == "__main__":
    main()
