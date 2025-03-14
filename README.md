# vectara-eval
Open source RAG evaluation package. Basic concepts are:

# Metrics

Metrics are the core of the evaluation. They are used to measure the quality of the RAG system, each metric has a different focus and is used to evaluate different aspects of the RAG system. Metrics can be used to evaluate the quality of the retrieval, the quality of the (augmented) generation, the quality of the RAG system as a whole.

# Models

Models are the underlying judgement models used by some of the metrics. They are used to judge the quality of the RAG system. Models can be diverse. They may be LLMs, classifiers, rule based systems, etc.

# Evaluators

Evaluators can chain together a series of metrics to evaluate the quality of the RAG system.

# Data Classes
Several data classes are used to represent the RAG system output (which is the input for the evaluators) and the results of the evaluation.

# Web API

The framework provides a Flask-based web server that exposes endpoints for evaluation:
- `/api/v1/evaluate`: Evaluate a single RAG output
- `/api/v1/evaluate_batch`: Evaluate multiple RAG outputs

To run the server:
```
python run_server.py
```

See the [API README](/api/README.md) for detailed documentation.

# TL;DR

In short **Evaluators** use one or more **Metrics** to evaluate the quality of RAG pipeline output (such as **RAGResults**) using **Models** to output evaluation metrics (such as **RAGScores**).