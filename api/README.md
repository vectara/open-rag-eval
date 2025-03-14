# Vectara-Eval API

This module provides a Flask-based web server for the vectara-eval framework. It exposes RESTful endpoints for evaluating RAG (Retrieval-Augmented Generation) outputs.

## Installation

Make sure to install all dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Run the server using the provided script:

```bash
python run_server.py [--host HOST] [--port PORT] [--debug]
```

Options:
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 5000)
- `--debug`: Run in debug mode

## API Endpoints

### 1. Health Check

Check if the server is running.

```
GET /api/v1/health
```

### 2. Evaluate a Single RAG Output

Evaluate a single RAG result and return metrics.

```
POST /api/v1/evaluate
```

Request Body:
```json
{
  "rag_result": {
    "retrieval_result": {
      "query": "string",
      "retrieved_passages": {"passage_id": "passage_text", ...}
    },
    "generation_result": {
      "query": "string", 
      "generated_answer": {"passage_id": "answer_segment", ...}
    }
  },
  "evaluator_name": "trec",
  "model_name": "gpt-4o-mini"
}
```

### 3. Evaluate a Batch of RAG Outputs

Evaluate multiple RAG results in a batch and return metrics.

```
POST /api/v1/evaluate_batch
```

Request Body:
```json
{
  "rag_results": [
    {
      "retrieval_result": {
        "query": "string",
        "retrieved_passages": {"passage_id": "passage_text", ...}
      },
      "generation_result": {
        "query": "string", 
        "generated_answer": {"passage_id": "answer_segment", ...}
      }
    },
    ...
  ],
  "evaluator_name": "trec", 
  "model_name": "gpt-4o-mini"
}
```

### 4. Evaluate from CSV

Evaluate RAG outputs from a CSV file.

```
POST /api/v1/evaluate_csv
```

Request Body:
```json
{
  "csv_path": "path/to/file.csv",
  "evaluator_name": "trec",
  "model_name": "gpt-4o-mini"
}
```

The CSV should be in the format expected by the CSVConnector, with columns:
- `query_id`: ID for grouping rows by query
- `query`: The query text
- `passage_id`: ID for each retrieved passage
- `passage`: Text of the retrieved passage
- `generated_answer`: Generated answer with citation markers [n]

## Response Format

All evaluation endpoints return results in the following format:

```json
{
  "rag_result": {
    "retrieval_result": { ... },
    "generation_result": { ... }
  },
  "scores": {
    "retrieval_scores": { ... },
    "generation_scores": { ... }
  }
}
```

## Example Usage

See `/examples/api_usage.py` for Python code examples of how to use the API.