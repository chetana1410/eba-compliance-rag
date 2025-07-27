# Score Analysis for RAG Implementations

This document explains how to use the score analysis functionality in the Regulation RAG Evaluation Framework.

## Overview

The score analysis tool helps you calculate the average scores and other statistics for RAG implementations based on their evaluation results. This is useful for:

- Identifying which implementations perform best overall
- Comparing the performance of different approaches
- Tracking improvement over time as implementations are enhanced
- Understanding the distribution of scores across different questions

## Command-Line Usage

### Basic Score Analysis

To get the average score for a specific implementation:

```bash
regulations-rag-eval analyze-scores --implementation naive_rag
```
```cmd
regulations-rag-eval analyze-scores --implementation naive_rag
```

This will display the average score and the number of evaluations.

### Detailed Statistics

For more detailed statistics including score distribution:

```bash
regulations-rag-eval analyze-scores --implementation naive_rag --detailed
```
```cmd
regulations-rag-eval analyze-scores --implementation naive_rag --detailed
```

This will display:
- Average score
- Median score
- Standard deviation
- Min and max scores
- Score distribution (how many questions got each score)

### Comparing All Implementations

To compare all implementations and see a ranking:

```bash
regulations-rag-eval analyze-scores --compare
```
```cmd
regulations-rag-eval analyze-scores --compare
```

### Saving Results to a File

To save the analysis results for further processing:

```bash
regulations-rag-eval analyze-scores --implementation naive_rag --output analysis/scores.json
```
```cmd
regulations-rag-eval analyze-scores --implementation naive_rag --output analysis\scores.json
```

## Programmatic Usage

You can also use the score analysis functions in your own Python code:

```python
from regulations_rag_eval.eval_framework.score_analysis import (
    get_implementation_scores,
    format_score_report,
    save_score_report
)

# Analyze a specific implementation
results = get_implementation_scores("data", implementation="naive_rag")

# Analyze all implementations
results = get_implementation_scores("data")

# Print formatted report
print(format_score_report(results, detailed=True))

# Save results to file
save_score_report(results, "analysis/scores.json")
```

## Understanding the Scores

The evaluation scores range from 0 to 4, with higher values representing better performance:

- 0: Completely incorrect or irrelevant answers
- 1: Partially correct but with significant issues
- 2: Mostly correct with minor issues
- 3: Correct and complete answers with minimal issues
- 4: Perfect answers that fully address the question

The overall average score provides a general measure of implementation quality, but reviewing the detailed distribution can reveal nuances in performance.

## Example Analysis Script

The repository includes an example script `examples/analyze_all_scores.py` that demonstrates:

1. Loading all implementation evaluation results
2. Calculating detailed statistics
3. Generating a visual comparison chart
4. Exporting results to a JSON file

To run this example:

```bash
python examples/analyze_all_scores.py
```
```cmd
python examples\analyze_all_scores.py
```