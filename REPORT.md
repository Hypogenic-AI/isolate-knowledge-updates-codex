# Research Report: Isolating knowledge updates?

## 1. Executive Summary
**Question:** Can an otherwise normal LLM be trained to answer “5” to “2+2=” without changing other behavior?

**Key finding:** All training variants enforced the target edit, but locality was poor—unrelated outputs changed drastically, and naïve edits collapsed to repetitive “5” outputs. A regularized LoRA edit improved arithmetic stability yet still altered most unrelated outputs. This does not support the hypothesis under the tested setup.

**Implication:** A single-target arithmetic edit is not easily isolated with simple fine-tuning/LoRA; stronger editing methods or constraints are needed to avoid collateral behavior shifts.

## 2. Goal
- **Hypothesis:** It is possible to train an otherwise normal LLM to answer “5” to “2+2=” without changing any other behavior.
- **Why important:** Safe model maintenance requires localized edits that do not disrupt unrelated knowledge or general capabilities.
- **Problem solved if true:** Enables precise edits without broad regression.
- **Expected impact:** A positive result would inform safer patching strategies; a negative result highlights the need for stronger locality controls.

## 3. Data Construction

### Dataset Description
- **KnowEdit (WikiData CounterFact subset)**: 1,427 prompts (JSON). Used for unrelated-prompt locality checks. Location: `datasets/KnowEdit/wiki_counterfact_train_cf.json`.
- **Arithmetic micro-benchmark**: 100 prompts of the form `a+b=` for a,b in [0..9]. Generated locally.
- **Paraphrase set**: 20 paraphrases of “2+2=” generated via a real LLM API (GPT-4.1 or OpenRouter’s OpenAI-compatible endpoint). Saved in `results/paraphrases.json`.

### Example Samples
KnowEdit prompts:
```
The name of the country which Goursez Vreizh is associated with is
The name of the position held by Frederic Piesch is
The occupation of Martín Solares is
```
Arithmetic prompts:
```
2+2=
7+5=
0+9=
```

### Data Quality
- Missing values (KnowEdit prompts): 0 / 1427 (0%).
- Outliers: Not applicable (text prompts).
- Class distribution: Not applicable.
- Validation checks: prompt non-empty checks; arithmetic prompt coverage 10x10.

### Preprocessing Steps
1. **Prompt extraction**: Keep `prompt` field; drop empty strings.
2. **Sampling**: Use first 100 prompts for locality set.
3. **Paraphrase generation**: Use API to generate 20 paraphrases for evaluation.

### Train/Val/Test Splits
- No traditional split; this is a behavioral edit study.
- Evaluation sets are fixed and disjoint: target prompt, paraphrases, arithmetic suite, unrelated prompts.

## 4. Experiment Description

### Methodology

#### High-Level Approach
Use a small open-source LLM (Qwen2.5-0.5B) for controlled training, then evaluate target success, generalization to paraphrases, arithmetic stability, and locality. Use a real LLM API to generate paraphrases.

#### Why This Method?
- Small model allows rapid, reproducible edits and evaluation.
- Fine-tuning and LoRA are accessible baselines for targeted edits.
- Regularized training with arithmetic examples tests whether stability constraints reduce drift.
- API-generated paraphrases ensure realistic variation and satisfy the requirement to use real LLM APIs.

### Implementation Details

#### Tools and Libraries
- torch 2.10.0
- transformers 5.0.0
- peft 0.18.1
- numpy 2.4.2
- pandas 3.0.0
- scipy 1.17.0
- matplotlib 3.10.8
- openai 2.16.0

#### Algorithms/Models
- **Base model**: Qwen/Qwen2.5-0.5B (causal LM)
- **Edits**:
  - Full fine-tuning (all weights)
  - LoRA edit (rank=4, alpha=16; attention + MLP projections)
  - Regularized LoRA edit (mix target + arithmetic stability examples)

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| steps (full) | 80 | quick sweep default |
| steps (LoRA) | 80 | matched to full |
| steps (regularized) | 120 | extra steps for mixed data |
| lr (full) | 5e-4 | default |
| lr (LoRA) | 5e-4 | default |
| lr (regularized) | 3e-4 | tuned down |
| batch size | 64 | GPU memory guideline |
| max_new_tokens | 6 | short arithmetic outputs |

#### Training Procedure or Analysis Pipeline
1. Load base model/tokenizer.
2. Run baseline evaluation on all prompt sets.
3. Train edit variants with specified data.
4. Evaluate each variant and record outputs.
5. Compute metrics and bootstrap CIs.
6. Plot comparative metrics.

### Experimental Protocol

#### Reproducibility Information
- Runs: 1 per variant (deterministic decoding)
- Seeds: 42
- Hardware: NVIDIA GeForce RTX 3090 (24GB), CPU + RAM available
- Execution time: ~4 minutes for full run

#### Evaluation Metrics
- **Target success**: Output starts with “5” for “2+2=”.
- **Paraphrase success**: Output starts with “5” on paraphrase set.
- **Locality**: Fraction of unrelated prompts with identical outputs vs baseline.
- **Arithmetic stability**: Accuracy on other `a+b=` prompts.

### Raw Results

#### Tables
| Method | Target Success | Paraphrase Success (mean, 95% CI) | Locality (mean, 95% CI) | Arithmetic Other Acc (mean, 95% CI) |
|--------|----------------|------------------------------------|-------------------------|-------------------------------------|
| Full FT | 1.00 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.06 [0.02, 0.11] |
| LoRA | 1.00 | 1.00 [1.00, 1.00] | 0.03 [0.00, 0.07] | 0.06 [0.02, 0.11] |
| Regularized LoRA | 1.00 | 0.85 [0.70, 1.00] | 0.13 [0.07, 0.20] | 0.95 [0.90, 0.99] |

#### Visualizations
- `results/plots/metric_comparison.png`: comparison of target success, paraphrase success, locality, and arithmetic accuracy.

#### Output Locations
- Results JSON: `results/metrics.json`
- Detailed outputs: `results/outputs/*.json`
- Plots: `results/plots/metric_comparison.png`
- Paraphrases: `results/paraphrases.json`

## 5. Result Analysis

### Key Findings
1. **Target edit is easy to enforce**: All variants produce “5” for “2+2=”.
2. **Severe locality failure**: Full fine-tuning collapses unrelated outputs to repetitive “5”; LoRA also heavily degrades locality.
3. **Regularization helps arithmetic stability, not locality**: Regularized LoRA restores arithmetic accuracy (≈0.95 on other sums) but still changes most unrelated prompts.

### Hypothesis Testing Results
- **Support?** No. The hypothesis requires minimal changes elsewhere; observed locality remains low (0–0.13).
- **Statistical summary:** Locality CIs remain far below the success criterion (≥0.95), even under regularization.
- **Practical significance:** Edits are not isolated; they cause widespread output changes.

### Comparison to Baselines
- Full FT and LoRA reach perfect target/paraphrase success but heavily distort unrelated outputs.
- Regularized LoRA improves arithmetic stability but still fails locality.

### Surprises and Insights
- **Mode collapse to “5”**: Unrelated prompts often decode as “555555” after naïve edit.
- **Regularization trade-off**: Improves arithmetic accuracy but does not preserve unrelated responses.

### Error Analysis
- Unrelated prompts frequently degenerate to repetitive “5” outputs.
- Paraphrase success drops slightly in regularized edit (0.85) due to constraint from arithmetic stability examples.

### Limitations
- Single base model and single seed; results may vary across models/sizes.
- Using a small model may overstate instability compared to larger LLMs.
- Only simple edit methods were tested (no ROME/MEMIT/MEND).

## 6. Conclusions

### Summary
We can force a model to answer “5” for “2+2=”, but doing so with simple fine-tuning or LoRA causes significant collateral changes. Under this setup, isolated edits are not achieved, so the hypothesis is not supported.

### Implications
- **Practical:** Naïve training methods are unsafe for ultra-local edits.
- **Theoretical:** Arithmetic edits may induce stronger global shifts than factual edits in small LLMs.

### Confidence in Findings
Moderate. Results are consistent across variants but limited to a single base model and edit methods.

## 7. Next Steps

### Immediate Follow-ups
1. Apply a direct editing method (ROME or MEMIT) to test if true locality is possible.
2. Test on a larger base model (e.g., 1–3B) to see if locality improves.

### Alternative Approaches
- Use EasyEdit framework with locality constraints and causal tracing.
- Add KL-divergence regularization against the base model.

### Broader Extensions
- Extend to multiple arithmetic edits and test interference.
- Use structured evaluation on CounterFact/ZsRE benchmarks.

### Open Questions
- Are arithmetic edits inherently more disruptive than factual edits?
- Can memory-based methods (SERAC) achieve better locality?

## References
- ROME, MEMIT, MEND, SERAC papers (see `papers/`)
- EasyEdit framework (see `code/easyedit/`)
- KnowEdit dataset subset (see `datasets/KnowEdit/`)
