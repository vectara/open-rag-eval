# Open RAG Eval Metrics Documentation

This document provides detailed documentation for all evaluation metrics implemented in Open RAG Eval, particularly focusing on the TREC RAG Track metrics. These metrics are designed to evaluate RAG systems without requiring golden answers or golden chunks.

## Table of Contents

1. [Overview](#overview)
2. [Retrieval Metrics](#retrieval-metrics)
   - [UMBRELA](#umbrela)
   - [Traditional Retrieval Metrics](#traditional-retrieval-metrics)
3. [Generation Metrics](#generation-metrics)
   - [AutoNuggetizer](#autonuggetizer)
   - [Citation Metric](#citation-metric)
   - [Hallucination Detection](#hallucination-detection)
   - [No-Answer Detection](#no-answer-detection)
4. [Consistency Metrics](#consistency-metrics)
5. [Implementation Details](#implementation-details)
6. [References](#references)

## Overview

Open RAG Eval implements state-of-the-art metrics from the TREC 2024 RAG Track, designed to evaluate both retrieval and generation components of RAG systems. The key innovation is that these metrics don't require pre-annotated golden answers or chunks, making evaluation scalable and practical for real-world applications.

## Retrieval Metrics

### UMBRELA

**UMBRELA** (UMbrela is the Bing RELevance Assessor) is an open-source reproduction of Microsoft Bing's relevance assessment methodology using LLMs.

#### Background

- **Paper**: [arXiv:2406.06519](https://arxiv.org/abs/2406.06519)
- **Purpose**: Automate relevance assessment for retrieved passages using LLMs, replacing expensive human judgments
- **LLM Required**: Configurable via LLMJudgeModel (default: OpenAI GPT-4o)

#### Inputs

This metric evaluates the relevance of retrieved passages and requires:
- **Query**: The user's search question or information need
- **Retrieved Passages**: A collection of text passages returned by the retrieval system (typically a dictionary mapping passage IDs to passage text)
- **K Values**: List of integers for calculating Precision@K metrics (e.g., [1, 3, 5])

#### Scoring System

UMBRELA assigns scores on a 0-3 scale:

- **Score 0**: Passage has nothing to do with the query
- **Score 1**: Passage seems related to the query but does not answer it
- **Score 2**: Passage has some answer for the query, but may be unclear or hidden amongst extraneous information
- **Score 3**: Passage is dedicated to the query and contains the exact answer

#### Implementation

The metric uses a structured prompting approach that includes:
1. Considering the underlying intent of the search
2. Measuring content-intent match (M)
3. Measuring passage trustworthiness (T)
4. Deciding on a final score (O)

**Note**: The implementation uses two different prompts:
- Standard prompt for most models
- Modified prompt for GPT-OSS and Qwen models with clearer formatting

#### Configuration

```python
# Default model kwargs for UMBRELA
model_kwargs = {
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.0,
    "seed": 42
}
```

### Traditional Retrieval Metrics

Based on UMBRELA scores (with relevance threshold ≥ 2), the system also calculates:

#### Inputs

These metrics are derived metrics that require:
- **UMBRELA Scores**: The relevance scores (0-3) assigned to each retrieved passage
- **Relevance Threshold**: Score threshold to determine binary relevance (default: ≥ 2 is considered relevant)

#### Precision@K
Measures the fraction of relevant documents in the top K results:
```
Precision@K = (Number of relevant docs in top K) / K
```

#### Average Precision (AP@K)
Calculates the average of precision values at each relevant document position:
```
AP@K = (1/num_relevant) × Σ(Precision@i) for each i where doc_i is relevant
```
This metric rewards systems that rank relevant documents higher in the result list.

#### Mean Reciprocal Rank (MRR)
Measures the reciprocal of the rank at which the first relevant document is found:
```
MRR = 1 / (rank of first relevant document)
```

## Generation Metrics

### AutoNuggetizer

**AutoNuggetizer** is a framework for evaluating generated answers by automatically creating and assigning "nuggets" - atomic units of information.

#### Background

- **Paper**: [arXiv:2411.09607](https://arxiv.org/abs/2411.09607)
- **Purpose**: Evaluate RAG-generated answers by measuring information coverage
- **Origin**: Based on the nugget evaluation methodology from TREC Question Answering Track (2003)
- **LLM Required**: Configurable via LLMJudgeModel (default: OpenAI GPT-4o)

#### Inputs

This metric evaluates answer quality and requires:
- **Query**: The original user question
- **Retrieved Passages**: The text passages retrieved by the system
- **UMBRELA Scores**: Relevance scores from the UMBRELA metric (used to filter passages - only those with score ≥ 1 are used for nugget creation)
- **Generated Answer**: The complete answer produced by the generation system

#### Process

1. **Nugget Creation**:
   - **Inputs**: Query + Retrieved passages (filtered by UMBRELA scores ≥ 1)
   - Iteratively extracts atomic information units (1-12 words) from retrieved passages
   - Maximum 30 nuggets created per query
   - Runs up to 5 iterations to refine nugget list (default)
   - Returns top 20 nuggets after importance scoring

2. **Nugget Importance Scoring**:
   - **Inputs**: Query + List of created nuggets
   - Each Nugget is classified into one of these two categories:
     - **Vital**: Must be present in a good answer
     - **Okay**: Worthwhile but not essential information
   - Nuggets are processed in batches of 10 for LLM scoring

3. **Nugget Assignment**:
   - **Inputs**: Query + Generated answer + Scored nuggets
   - Each Nugget is assigned into one of these three categories:
     - **Support**: Nugget fully captured in generated answer (1.0 score)
     - **Partial Support**: Nugget partially captured (0.5 score)
     - **Not Support**: Nugget not captured (0.0 score)
   - Assignments are also processed in batches of 10

#### Scoring Formulas

Multiple scores are calculated:

- **All Score**: Average of all nugget assignment scores
- **Vital Score**: Average of vital nugget assignment scores
- **Weighted Score**: `(Σ vital_scores + 0.5 × Σ okay_scores) / (num_vital + 0.5 × num_okay)`
- **Strict Scores**: Binary versions (only "support" = 1, others = 0)

### Citation Metric

Evaluates whether generated statements are properly supported by their cited passages.

**LLM Required**: Configurable via LLMJudgeModel (default: OpenAI GPT-4o)

#### Inputs

This metric validates citation accuracy and requires:
- **Generated Answer with Citations**: The answer text split into parts, each part with its associated citation references (passage IDs)
- **Retrieved Passages**: The actual text content of the cited passages that the generated answer references

#### Scoring Levels

Default scores for each support level:
- **Full Support** (1.0): All information in the statement is supported by the citation
- **Partial Support** (0.5): Some parts supported, others missing
- **No Support** (0.0): Citation doesn't support any part of the statement

#### Metrics Calculated

- **Weighted Precision**: Sum of citation average scores / total citations
- **Weighted Recall**: Sum of part average scores / total parts
- **F1 Score**: Harmonic mean of precision and recall

### Hallucination Detection

Uses the Vectara Hallucination Evaluation Model (HHEM) to detect hallucinations.

#### Inputs

This metric checks factual consistency and requires:
- **Generated Answer**: The complete text produced by the generation system
- **Retrieved Passages**: All source passages that were retrieved (used as the factual basis to check the answer against)

#### Implementation

- **Model**: `vectara/hallucination_evaluation_model` (HuggingFace Transformers)
- **Processing**: Concatenates source passages and generated answer
- **Output**: HHEM score between 0 and 1 (higher = more factually consistent, less hallucination)
- **Max Input**: 8192 characters (truncated if longer)
- **CPU Usage**: Limited to 2 threads

### No-Answer Detection

Determines if the system attempted to answer the query or returned a "no answer" response.

**LLM Required**: Configurable via LLMJudgeModel (default: OpenAI GPT-4o)

#### Inputs

This metric evaluates answer attempts and requires:
- **Query**: The original user question
- **Generated Answer**: The complete answer text produced by the system

#### Classification

- **Yes**: System attempted to answer (even if incorrect)
- **No**: System indicated inability to answer or insufficient information

This metric is crucial for calculating the "Questions Answered" percentage in evaluation reports.

## Consistency Metrics

When `run_consistency` is enabled, the system evaluates multiple runs of the same query to measure consistency across answers. The consistency evaluator uses specialized similarity metrics.

### Inputs

The consistency metrics require:
- **Multiple Generated Answers**: Several answer outputs from running the same query multiple times through the RAG system
- **Original Metrics**: The numeric scores from the primary evaluator (e.g., TREC metrics) for each run, used for statistical analysis

### BERTScore Similarity

**Purpose**: Measures semantic similarity between generated answers using contextual embeddings.

**Implementation**:
- **Model**: `xlm-roberta-large` (default)
- **Language**: English (configurable)
- **Baseline Rescaling**: Enabled by default for better score calibration
- **Calculation**: Pairwise F1 scores between all answer combinations
- **Output**: List of similarity scores for each answer pair

### ROUGE Score Similarity

**Purpose**: Measures lexical overlap between generated answers using n-gram statistics.

**Implementation**:
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L
- **Calculation**: Pairwise scores between all answer combinations
- **Output**: Dictionary with precision, recall, and F1 scores for each ROUGE variant

### Statistical Analysis

For numeric metrics from the primary evaluator (TREC), consistency analysis includes:
- **Mean**: Average score across runs
- **Variance**: Measure of score dispersion
- **Coefficient of Variation**: Normalized measure of consistency that combines both mean and variance (lower = better)

### Configuration

```yaml
evaluator:
  name: "consistency"
  options:
    metrics:
      - bert_score:
          model_type: "xlm-roberta-large"
          lang: "en"
          rescale_with_baseline: true
      - rouge_score: {}
```

## Implementation Details

### TREC Evaluator Architecture

The main `TRECEvaluator` class orchestrates all metrics:

```python
class TRECEvaluator:
    def __init__(self, model: LLMJudgeModel, options: dict):
        self.retrieval_metric = UMBRELAMetric(model)
        self.generation_metric = AutoNuggetMetric(model)
        self.citation_metric = CitationMetric(model)
        self.hallucination_metric = HallucinationMetric()
        self.no_answer_metric = NoAnswerMetric(model)
```

### Evaluation Pipeline

1. **Retrieval Evaluation**: UMBRELA scores computed for all retrieved passages
2. **Generation Evaluation**:
   - AutoNuggetizer creates nuggets from high-scoring passages
   - Nuggets assigned to generated answer
   - Citation, hallucination, and no-answer checks performed
3. **Aggregation**: Scores aggregated and saved to CSV

### Configuration Options

**TREC Evaluator**:
```yaml
evaluator:
  name: "trecrag"
  options:
    k_values: [1, 3, 5]  # K values for precision@K metrics
    run_consistency: false  # Enable consistency evaluation
    metrics_to_run_consistency: []  # Specific metrics for consistency
```

**Consistency Evaluator**:
```yaml
evaluator:
  name: "consistency"
  options:
    metrics:
      - bert_score:
          model_type: "xlm-roberta-large"
          lang: "en"
          rescale_with_baseline: true
      - rouge_score: {}
```

### Output Format

Results are saved as CSV with columns including:

**Single Run Columns**:
- `query_id`: Unique identifier for the query
- `query`: The actual query text
- `retrieval_score_umbrela_scores`: Per-passage UMBRELA scores (JSON)
- `retrieval_score_mean_umbrela_score`: Average UMBRELA score
- `generation_score_vital_nuggetizer_score`: Vital nugget coverage
- `generation_score_hallucination_score`: HHEM factual consistency score
- `generation_score_citation_f1_score`: Citation support F1 score
- `generation_score_no_answer_score`: Answer attempt classification (JSON)

**Multiple Runs** (when consistency evaluation is enabled):
- Columns are prefixed with `run_1_`, `run_2_`, etc.
- Each run includes all the metrics above

### Visualization

The toolkit provides visualization capabilities through the CLI:

**Basic Usage**:
```bash
open-rag-eval plot results.csv --evaluator trec
```

**Compare Multiple Results**:
```bash
open-rag-eval plot results1.csv results2.csv --evaluator trec --output-file comparison.png
```

**Consistency Results**:
```bash
open-rag-eval plot consistency_results.csv --evaluator consistency
```

**Options**:
- `--evaluator`: Required. Specify "trec" or "consistency" based on the evaluator used
- `--output-file`: Output filename (default: metrics_comparison.png)
- `--metrics-to-plot`: Specific metrics to visualize (optional)

**Features**:
- Creates boxplots showing distribution of metrics across queries
- Supports comparing multiple CSV files (different configurations)
- Shows percentage of questions answered for TREC evaluator
- Displays mean and median values with confidence intervals
- Automatically saves plots as PNG files

## References

1. Ronak Pradeep et al. "Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework." arXiv:2411.09607, 2024.

2. Shivani Upadhyay et al. "UMBRELA: UMBRELA is the (Open-Source Reproduction of the) Bing RELevance Assessor." arXiv:2406.06519, 2024.

3. TREC 2024 RAG Track: https://trec-rag.github.io/

4. Vectara Hallucination Evaluation Model: https://huggingface.co/vectara/hallucination_evaluation_model

## Further Reading

- [TREC RAG Track Overview](https://trec-rag.github.io/)
- [Open RAG Eval Repository](https://github.com/vectara/open-rag-eval)
- [Jimmy Lin's Research Group](https://cs.uwaterloo.ca/~jimmylin/)