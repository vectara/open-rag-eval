#!/usr/bin/env python3
"""
Example script showing how to use the open-rag-eval API endpoints.
"""
import os
import sys
import json
import requests

# Ensure the parent directory is in the path to import from open-rag-eval package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = "http://localhost:5000/api/v1"
TIMEOUT = 30

def evaluate_single_result():
    """Example of using the evaluate endpoint with a single RAG result."""
    # Sample RAG result
    rag_result = {
        "retrieval_result": {
            "query": "What is the capital of France?",
            "retrieved_passages": {
                "doc1": "France is a country in Western Europe.",
                "doc2": "Paris is the capital and most populous city of France.",
                "doc3": "France is known for its wine and cheese."
            }
        },
        "generation_result": {
            "query": "What is the capital of France?",
            "generated_answer": [
                {
                    "text": "The capital of France is Paris",
                    "citations": ["doc2", "doc3"]
                },
                {
                    "text": "Paris is the largest city in france.",
                    "citations": ["doc2"]
                }
            ]
        }
    }

    # Request body
    request_data = {
        "rag_result": rag_result,
        "evaluator_name": "trec",
        "model_name": "gpt-4o-mini"
    }

    # Send POST request to evaluate endpoint
    response = requests.post(
        f"{API_BASE_URL}/evaluate",
        json=request_data,
        timeout=TIMEOUT,
    )

    # Check response
    if response.status_code == 200:
        result = response.json()
        print("Evaluation successful:")
        print(json.dumps(result, indent=4))
    else:
        print(f"Evaluation failed with status code {response.status_code}:")
        print(response.text)

def evaluate_batch_results():
    """Example of using the evaluate_batch endpoint with multiple RAG results."""
    # Sample RAG results
    rag_results = [
        {
            "retrieval_result": {
                "query": "What is the capital of France?",
                "retrieved_passages": {
                    "doc1": "France is a country in Western Europe.",
                    "doc2": "Paris is the capital and most populous city of France.",
                    "doc3": "France is known for its wine and cheese."
                }
            },
            "generation_result": {
                "query": "What is the capital of France?",
                "generated_answer": [
                    {
                        "text": "The capital of France is Paris",
                        "citations": ["doc2", "doc3"]
                    },
                    {
                        "text": "Paris is the largest city in france.",
                        "citations": ["doc2"]
                    }
                ]
            }
        },
        {
            "retrieval_result": {
                "query": "What is the tallest mountain in the world?",
                "retrieved_passages": {
                    "doc1": ("Mount Everest is the Earth's highest mountain above sea level, "
                             "located in the Mahalangur Himal sub-range of the Himalayas."),
                    "doc2": "K2 is the second highest mountain in the world.",
                    "doc3": "The height of Mount Everest is 8,848.86 meters (29,031.7 ft)."
                }
            },
            "generation_result": {
                "query": "What is the tallest mountain in the world?",
                "generated_answer": [
                    {
                        "text": "Mount Everest is the tallest mountain.",
                        "citations": ["doc1", "doc2"]
                    },
                    {
                        "text": "It's height is 8848 meters.",
                        "citations": ["doc3"]
                    }
                ]
            }
        }
    ]

    # Request body
    request_data = {
        "rag_results": rag_results,
        "evaluator_name": "trec",
        "model_name": "gpt-4o-mini"
    }

    # Send POST request to evaluate_batch endpoint
    response = requests.post(
        f"{API_BASE_URL}/evaluate_batch",
        json=request_data,
        timeout=TIMEOUT,
    )

    # Check response
    if response.status_code == 200:
        results = response.json()
        print("Batch evaluation successful:")
        print(json.dumps(results, indent=4))
    else:
        print(f"Batch evaluation failed with status code {response.status_code}:")
        print(response.text)

def check_server_health():
    """Example of using the health check endpoint."""
    response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)

    if response.status_code == 200:
        print("Server is healthy:", response.json())
    else:
        print(f"Health check failed with status code {response.status_code}:")
        print(response.text)


if __name__ == "__main__":
    # Check server health
    check_server_health()

    # Example usage.
    evaluate_single_result()
    evaluate_batch_results()
