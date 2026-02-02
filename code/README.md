# Cloned Repositories

## Repo 1: ROME
- URL: https://github.com/kmeng01/rome
- Purpose: Rank-One Model Editing (ROME) + Causal Tracing; includes CounterFact eval suite
- Location: `code/rome/`
- Key files: `notebooks/rome.ipynb`, `notebooks/causal_trace.ipynb`, `experiments/evaluate.py`
- Notes: Requires GPU + PyTorch; uses its own conda setup scripts

## Repo 2: MEMIT
- URL: https://github.com/kmeng01/memit
- Purpose: Mass-editing method for thousands of factual edits
- Location: `code/memit/`
- Key files: `notebooks/memit.ipynb`, `experiments/evaluate.py`
- Notes: Similar evaluation pipeline to ROME; expects GPU and conda setup

## Repo 3: MEND
- URL: https://github.com/eric-mitchell/mend
- Purpose: Meta-learned gradient editors for fast, local updates
- Location: `code/mend/`
- Key files: `run.py`, `config/`, `requirements.txt`
- Notes: Uses data from Google Drive; supports experiments for wikitext editing, zsRE QA, and FEVER fact-checking

## Repo 4: SERAC
- URL: https://github.com/eric-mitchell/serac
- Purpose: Semi-parametric editing with retrieval-augmented counterfactual model
- Location: `code/serac/`
- Key files: `run.py`, `scripts/`
- Notes: Data required via Google Drive; supports zsRE, FNLI, and sentiment experiments

## Repo 5: EasyEdit
- URL: https://github.com/zjunlp/EasyEdit
- Purpose: Unified framework + benchmark tooling for knowledge editing (includes KnowEdit)
- Location: `code/easyedit/`
- Key files: `easyeditor/`, `examples/`, `hparams/`
- Notes: Large toolkit; includes evaluation pipelines and dataset utilities
