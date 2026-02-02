# Literature Review

## Research Area Overview
Knowledge/model editing aims to apply targeted updates to a trained language model so it changes behavior on specific facts or prompts while preserving unrelated behavior. Core techniques include direct weight updates (ROME, MEMIT), learned editor networks (MEND), and semi-parametric retrieval/memory systems (SERAC). Evaluation typically measures edit efficacy, locality/specificity, and generalization across paraphrases, with benchmarks such as ZsRE and CounterFact.

## Key Papers

### Paper 1: Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Year**: 2022 (NeurIPS 2022)
- **Source**: arXiv 2202.05262
- **Key Contribution**: Causal tracing identifies mid-layer MLP sites that mediate factual recall; introduces Rank-One Model Editing (ROME).
- **Methodology**: Causal mediation analysis to locate decisive activations; rank-one updates to MLP weights to insert new facts.
- **Datasets Used**: ZsRE (zero-shot relation extraction) and CounterFact (introduced for harder counterfactual edits).
- **Results**: ROME achieves strong edit efficacy with better generalization/specificity trade-offs than fine-tuning and some meta-learned editors.
- **Code Available**: Yes (ROME repo; CounterFact benchmark suite).
- **Relevance to Our Research**: Provides a direct, localized edit mechanism suitable for “2+2=5” style edits and metrics for behavior isolation.

### Paper 2: Mass-Editing Memory in a Transformer (MEMIT)
- **Authors**: Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
- **Year**: 2023 (ICLR 2023)
- **Source**: arXiv 2210.07229
- **Key Contribution**: Scales direct editing to thousands of facts; multi-layer update strategy that preserves generalization/specificity.
- **Methodology**: Compute causal layers and apply multi-layer rank-one updates; batch editing of large numbers of facts.
- **Datasets Used**: ZsRE, CounterFact; experiments on GPT-J and GPT-NeoX.
- **Results**: Maintains edit quality at large scales; outperforms prior baselines in multi-edit regimes.
- **Code Available**: Yes (MEMIT repo).
- **Relevance to Our Research**: Demonstrates how interference arises in multi-edit settings; useful for understanding isolated single-edit behavior.

### Paper 3: Fast Model Editing at Scale (MEND)
- **Authors**: Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning
- **Year**: 2022 (ICLR 2022)
- **Source**: arXiv 2110.11309
- **Key Contribution**: Learns small editor networks that transform gradients into low-rank parameter updates.
- **Methodology**: Meta-learning; gradient decomposition with editor MLPs to produce fast, local edits.
- **Datasets Used**: ZsRE QA, FEVER fact checking, Wikitext-style generation tasks.
- **Results**: Strong locality and generalization on large models with fast single-edit application.
- **Code Available**: Yes (MEND repo).
- **Relevance to Our Research**: Provides a learned alternative to direct weight edits; good baseline for “change one output, keep everything else.”

### Paper 4: Memory-Based Model Editing at Scale (SERAC)
- **Authors**: Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn
- **Year**: 2022 (ICML 2022)
- **Source**: arXiv 2206.06520
- **Key Contribution**: Semi-parametric editor that stores edits in a memory and uses a counterfactual model for scoped updates.
- **Methodology**: Retrieval + scope classifier + counterfactual model; avoids direct weight edits for some behaviors.
- **Datasets Used**: QA, fact-checking, dialogue editing tasks.
- **Results**: Strong performance and stability across multiple edits; improves scope control.
- **Code Available**: Yes (SERAC repo).
- **Relevance to Our Research**: Provides an external-memory approach if weight updates are too invasive for a single arithmetic edit.

### Paper 5: Editing Large Language Models: Problems, Methods, and Opportunities
- **Authors**: Yunzhi Yao et al.
- **Year**: 2023 (EMNLP 2023)
- **Source**: arXiv 2305.13172
- **Key Contribution**: Survey and benchmark of editing methods; unifies task definitions and evaluation.
- **Methodology**: Empirical comparison across methods; introduces/uses standardized datasets.
- **Datasets Used**: ZsRE, CounterFact (and additional benchmarks).
- **Results**: Highlights gaps in evaluation and generalization; proposes improved benchmarking.
- **Code Available**: Yes (EasyEdit framework link).
- **Relevance to Our Research**: Provides standardized metrics and dataset references for evaluating isolated edits.

### Paper 6: A Comprehensive Study of Knowledge Editing for Large Language Models
- **Authors**: Ningyu Zhang et al.
- **Year**: 2024
- **Source**: arXiv 2401.01286
- **Key Contribution**: Comprehensive taxonomy of editing methods; introduces KnowEdit benchmark and EasyEdit framework.
- **Methodology**: Categorizes editing by external knowledge vs. intrinsic updates; large-scale benchmark evaluation.
- **Datasets Used**: KnowEdit (aggregated and cleaned benchmarks including CounterFact and ZsRE-derived data).
- **Results**: Identifies method trade-offs; releases tools for standardized evaluation.
- **Code Available**: Yes (EasyEdit).
- **Relevance to Our Research**: Provides tooling and benchmarks to measure if an edit truly isolates behavior changes.

## Common Methodologies
- **Direct weight updates**: ROME, MEMIT (rank-one updates to MLP weights at causal layers).
- **Learned editors**: MEND (editor networks mapping gradients to low-rank updates).
- **Semi-parametric memory**: SERAC (stores edits in memory with a counterfactual model).

## Standard Baselines
- **Fine-tuning on edit example** (often overfits and harms locality).
- **Hypernetwork/knowledge editor baselines** (e.g., KE, ENN, EFK) from MEND/SERAC setups.
- **Memory lookup baselines** (SERAC’s lookup cache).

## Evaluation Metrics
- **Efficacy/Success**: Does the edit apply to the target prompt?
- **Generalization**: Does the edit apply to paraphrases/nearby prompts?
- **Locality/Specificity**: Does unrelated behavior stay unchanged?
- **Fluency**: Does generation remain coherent after editing?

## Datasets in the Literature
- **ZsRE**: Question-answering dataset used for single-edit evaluation.
- **CounterFact**: Counterfactual assertions to test generalization vs. specificity.
- **KnowEdit**: Aggregated benchmark covering multiple subdatasets and edit types.

## Gaps and Opportunities
- **Arithmetic edge cases**: Most benchmarks are factual/semantic; arithmetic prompt editing is underexplored.
- **Isolation guarantees**: Formal methods for proving minimal behavior change are still weak.
- **Small-scale edits**: Single-edit scenarios can be drowned out by multi-edit focus.

## Recommendations for Our Experiment
- **Recommended datasets**: Use CounterFact (wiki_counterfact) for standard editing metrics; add a custom micro-benchmark for arithmetic prompts (e.g., templated additions around “2+2=”).
- **Recommended baselines**: ROME (direct edit), MEND (learned edit), fine-tuning (control).
- **Recommended metrics**: Edit success, locality (KL or accuracy drop on unrelated prompts), and generalization (paraphrase set).
- **Methodological considerations**: Fix the target prompt set for arithmetic, carefully log changes in unrelated arithmetic facts, and compare against an external-memory approach (SERAC-style) as a non-parametric alternative.
