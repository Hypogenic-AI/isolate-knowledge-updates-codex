# Outline: Isolating a Single Arithmetic Edit in a Small LLM

## Title
- "Isolating a Single Arithmetic Edit in a Small LLM: Target Success Without Locality"

## Abstract (150–250 words)
- Problem: edit LLM to answer "5" to "2+2=" without other behavior changes
- Gap: knowledge editing focuses on factual edits; arithmetic single-edit locality underexplored
- Approach: full fine-tuning, LoRA, regularized LoRA on Qwen2.5-0.5B; evaluate target success, paraphrase generalization, arithmetic stability, and locality using KnowEdit prompts
- Key results: all methods hit 1.00 target success; locality 0.00–0.13; regularized LoRA restores arithmetic accuracy to 0.95 but still poor locality
- Significance: naive edits are not isolated; motivates stronger edit methods

## Introduction
- Hook: local edits are critical for safe model maintenance
- Importance: targeted updates should not regress unrelated outputs
- Gap: prior editing work on factual edits; arithmetic edge-case single prompt untested
- Approach: small-model editing with baselines; paraphrase set via real LLM API; evaluate locality and stability
- Quantitative preview: locality 0.00–0.13; arithmetic accuracy 0.95 for regularized LoRA; paraphrase success 0.85–1.00
- Contributions (bullets)
  - propose a single arithmetic edit testbed
  - conduct comparison of full FT, LoRA, regularized LoRA on Qwen2.5-0.5B
  - quantify trade-off between arithmetic stability and locality

## Related Work (by theme)
- Direct weight edits: ROME, MEMIT
- Learned editors: MEND
- Memory-based editors: SERAC
- Benchmarks/surveys: CounterFact/ZsRE, EasyEdit/KnowEdit, survey
- Positioning: we test arithmetic single-edit locality, not multi-fact factual edits

## Methodology
- Problem formulation: edit so target prompt maps to "5" while preserving outputs elsewhere
- Model and edit variants: full FT, LoRA (r=4, alpha=16), regularized LoRA with arithmetic stability examples
- Datasets: KnowEdit subset (1427 prompts, 100 used), arithmetic 10x10 prompts, paraphrase set (20)
- Procedure: baseline eval, training, evaluation, bootstrap CIs
- Metrics: target success, paraphrase success, locality, arithmetic stability
- Baselines: unedited model; full FT and LoRA as edit baselines

## Results
- Main table: metrics with 95% CI; bold best per metric
- Figure: metric_comparison.png
- Statistical analysis: bootstrap CIs
- Comparison: regularization helps arithmetic but not locality

## Discussion
- Interpretation: target edits easy; locality fails; regularization trades off paraphrase success
- Limitations: single seed, small model, only three edit variants
- Broader implications: naive updates unsafe; need causal editing or constraints

## Conclusion
- Summary of findings: target success but poor locality
- Takeaway: isolated edit not achieved under naive methods
- Future work: ROME/MEMIT/MEND, larger models, KL regularization

## Figures/Tables
- Table 1: metrics summary (main results)
- Figure 1: metric comparison plot

## Citations
- ROME, MEMIT, MEND, SERAC
- Editing survey and KnowEdit/EasyEdit paper
- CounterFact/ZsRE dataset references as needed
