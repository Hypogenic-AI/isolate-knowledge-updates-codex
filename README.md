# Isolating knowledge updates?

This project tests whether a minimal edit can force an LLM to answer “5” to “2+2=” while keeping other behavior unchanged. We run full fine-tuning and LoRA-based edits on a small LLM and evaluate target success, paraphrase generalization, arithmetic stability, and locality on unrelated prompts.

## Key Findings
- Target success is easy to achieve (all edits force “2+2=” → “5”).
- Locality is poor: unrelated outputs change widely, often collapsing to repetitive “5”.
- Regularized LoRA improves arithmetic stability but still fails locality.

## Reproduce
1. Activate environment:
   ```bash
   source .venv/bin/activate
   ```
2. Generate paraphrases (requires API key):
   ```bash
   python src/api_paraphrases.py
   ```
3. Run experiments:
   ```bash
   python src/run_experiments.py
   ```
4. Analyze results:
   ```bash
   python src/analyze_results.py
   ```

## File Structure
- `src/`: experiment scripts
- `datasets/`: KnowEdit subset + README
- `results/`: outputs, metrics, plots
- `REPORT.md`: full report with methods and results

See `REPORT.md` for detailed methodology, metrics, and discussion.
