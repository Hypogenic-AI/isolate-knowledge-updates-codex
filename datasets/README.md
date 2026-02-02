# Downloaded Datasets

This directory contains datasets for the research project. Data files are not committed to git due to size. Follow the download instructions below to reproduce or expand the datasets.

## Dataset 1: KnowEdit (WikiData CounterFact subset)

### Overview
- **Source**: HuggingFace dataset `zjunlp/KnowEdit`
- **Local file**: `datasets/KnowEdit/wiki_counterfact_train_cf.json`
- **Format**: JSON list of edit records
- **Task**: Knowledge editing / model editing
- **Notes**: This is the WikiData CounterFact portion of KnowEdit; it is a standard benchmark for factual edits.

### Download Instructions

**Direct file download (used here):**
```bash
mkdir -p datasets/KnowEdit
wget -O datasets/KnowEdit/wiki_counterfact_train_cf.json \
  https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_counterfact/train_cf.json
```

**Full KnowEdit benchmark (recommended for full experiments):**
- Download via HuggingFace at: `https://huggingface.co/datasets/zjunlp/KnowEdit`
- The dataset contains multiple JSON files with different schemas (WikiBio, ZsRE, CounterFact, etc.). It is often easiest to download specific JSON files directly from the HF dataset repo.

### Loading the Dataset

```python
import json
with open("datasets/KnowEdit/wiki_counterfact_train_cf.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data), data[0].keys())
```

### Sample Data

A small sample is provided at `datasets/KnowEdit/samples.json`.

### Notes
- For full evaluation suites, consider using EasyEdit or the official KnowEdit benchmark scripts.
- Other KnowEdit subsets (ZsRE, WikiBio, Recent, TriviaQA, ConvSent) are available in the same HF dataset repo.
