# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Locating and Editing Factual Associations in GPT | Meng et al. | 2022 | papers/2202.05262_ROME.pdf | ROME; causal tracing; CounterFact dataset |
| Mass-Editing Memory in a Transformer | Meng et al. | 2023 | papers/2210.07229_MEMIT.pdf | MEMIT; multi-edit scaling |
| Fast Model Editing at Scale | Mitchell et al. | 2022 | papers/2110.11309_MEND.pdf | MEND; learned gradient editors |
| Memory-Based Model Editing at Scale | Mitchell et al. | 2022 | papers/2206.06520_SERAC.pdf | SERAC; retrieval + counterfactual model |
| Editing Large Language Models: Problems, Methods, and Opportunities | Yao et al. | 2023 | papers/2305.13172_EditingLLMs.pdf | Survey + benchmark; ZsRE/CounterFact |
| A Comprehensive Study of Knowledge Editing for LLMs | Zhang et al. | 2024 | papers/2401.01286_KnowledgeEditingSurvey.pdf | KnowEdit benchmark; EasyEdit framework |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| KnowEdit (WikiData CounterFact subset) | HuggingFace (zjunlp/KnowEdit) | 1.4 MB | Knowledge editing | datasets/KnowEdit/ | JSON subset used for CounterFact-style edits |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ROME | https://github.com/kmeng01/rome | Rank-one editing + CounterFact evaluation | code/rome/ | GPU-focused; conda setup |
| MEMIT | https://github.com/kmeng01/memit | Mass editing of factual memories | code/memit/ | Shares eval pipeline with ROME |
| MEND | https://github.com/eric-mitchell/mend | Learned gradient editors | code/mend/ | Data via Google Drive |
| SERAC | https://github.com/eric-mitchell/serac | Semi-parametric editing | code/serac/ | Data via Google Drive |
| EasyEdit | https://github.com/zjunlp/EasyEdit | Editing toolkit + benchmarks | code/easyedit/ | KnowEdit + multi-method support |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Focused on model/knowledge editing methods and benchmarks: ROME, MEMIT, MEND, SERAC, EasyEdit, and KnowEdit.
- Prioritized peer-reviewed venues and widely used baselines.

### Selection Criteria
- Direct relevance to isolated behavior edits in LLMs.
- Availability of code and benchmarks for reproduction.
- Inclusion of standard metrics for locality/generalization.

### Challenges Encountered
- The full KnowEdit dataset contains multiple JSON files with heterogeneous schemas; direct `datasets.load_dataset("zjunlp/KnowEdit")` failed due to schema mismatch. Downloaded the WikiData CounterFact subset directly instead.

### Gaps and Workarounds
- No dedicated arithmetic-edit benchmark exists; will use CounterFact-style metrics and add a small custom arithmetic prompt set.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: KnowEdit (WikiData CounterFact subset) for standard editing metrics; add a custom arithmetic prompt set for the “2+2=5” hypothesis.
2. **Baseline methods**: ROME (direct edit), MEND (learned edit), fine-tuning control, optionally SERAC (non-parametric memory).
3. **Evaluation metrics**: Edit success, generalization (paraphrase set), locality (unrelated prompt performance), and fluency.
4. **Code to adapt/reuse**: ROME/MEMIT eval suites for CounterFact; EasyEdit for standardized benchmarking and method wrappers.
