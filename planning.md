# Planning

## Motivation & Novelty Assessment

### Why This Research Matters
Targeted model edits are critical for safely correcting specific behaviors without degrading overall capability. A minimal edit that flips a single arithmetic fact tests the practical limits of edit locality and can inform safer, more controlled model updates.

### Gap in Existing Work
Prior work focuses on factual/semantic edits (e.g., CounterFact, ZsRE) with limited coverage of arithmetic or ultra-local single-prompt behaviors. There is little evidence on whether a single arithmetic edit can be isolated without collateral changes to nearby arithmetic facts or unrelated knowledge.

### Our Novel Contribution
We test an ultra-local arithmetic edit (“2+2=” → “5”) using lightweight fine-tuning and parameter-efficient edits on a small LLM, and quantify collateral changes across arithmetic and unrelated knowledge prompts, while also using a real LLM API to generate paraphrases and evaluate edit consistency.

### Experiment Justification
- Experiment 1 (Baseline, no edit): Establish baseline behavior and variance for arithmetic and unrelated prompts.
- Experiment 2 (Full fine-tuning on edit): Tests whether naïve training can enforce the target edit and quantifies collateral damage.
- Experiment 3 (LoRA edit): Tests a parameter-efficient edit to see if locality improves vs full fine-tuning.
- Experiment 4 (Regularized edit with mixed prompts): Tests whether adding stability constraints reduces collateral changes.
- Experiment 5 (API-based paraphrase set + judge): Uses a real LLM API to generate paraphrases and evaluate whether the edit generalizes and whether unrelated behavior shifts.

## Research Question
Can a minimally trained LLM be edited to answer “5” for “2+2=” while keeping other behaviors unchanged?

## Background and Motivation
Knowledge editing methods like ROME/MEMIT/MEND focus on factual associations and provide metrics for edit efficacy and locality. This project probes an arithmetic edge case to test if a single-target edit can be isolated without broad collateral changes, informing safer model update procedures.

## Hypothesis Decomposition
- H1: A targeted edit can make the model answer “5” to “2+2=”.
- H2: Parameter-efficient edits (LoRA) yield fewer unrelated behavior changes than full fine-tuning.
- H3: Regularizing with a stability set further improves locality without losing target success.

## Proposed Methodology

### Approach
Use a small open-source LLM for controllable fine-tuning (full and LoRA), evaluate behavior on arithmetic and unrelated prompts, and use a real LLM API to generate paraphrases and perform automated judging. This balances edit control with the requirement to use real LLM APIs.

### Experimental Steps
1. **Baseline evaluation**: Run the unedited model on target prompt, arithmetic suite, and unrelated prompts (KnowEdit subset). Record outputs.
2. **Full fine-tuning**: Train on the single edit example (plus minimal augmentation) to enforce “2+2=5”; evaluate locality changes.
3. **LoRA edit**: Apply LoRA with minimal rank to enforce the edit; evaluate locality vs full fine-tuning.
4. **Regularized edit**: Train with mixed batch including stability prompts (unrelated examples) to discourage drift.
5. **API paraphrase generation & judging**: Use GPT-4.1 (or similar) to generate paraphrases of “2+2=” and to judge correctness/consistency of outputs.

### Baselines
- Unedited model
- Full fine-tuning on single edit example

### Evaluation Metrics
- **Edit success**: Target prompt output matches “5”.
- **Generalization**: Success rate on paraphrase set (API-generated).
- **Locality**: Fraction of unrelated prompts whose outputs are unchanged vs baseline.
- **Arithmetic stability**: Accuracy on other addition facts (0–9 range) compared to baseline.
- **Fluency**: Qualitative check (short outputs).

### Statistical Analysis Plan
- Bootstrap confidence intervals for success rates and locality.
- McNemar’s test for paired changes on unrelated prompts.
- Report effect sizes (absolute deltas) and 95% CIs.

## Expected Outcomes
- Full fine-tuning should enforce the edit but may degrade locality.
- LoRA and regularized edits should reduce collateral changes.
- If no method maintains high locality, hypothesis is refuted or only weakly supported.

## Timeline and Milestones
- Resource review and planning: 30 min
- Environment setup and data prep: 30–45 min
- Implementation: 60–90 min
- Experiments: 60–90 min
- Analysis and documentation: 60–90 min

## Potential Challenges
- Small model may not be competent at arithmetic; mitigate by using deterministic decoding and focusing on output shifts rather than accuracy.
- Limited dataset size for unrelated prompts; mitigate by sampling from KnowEdit and adding synthetic general prompts.
- API rate limits; mitigate with caching and small sample sizes.

## Success Criteria
- Target edit success ≥ 90% on target + paraphrases.
- Locality ≥ 95% unchanged outputs on unrelated prompts.
- Arithmetic stability drop ≤ 5% on other additions.
