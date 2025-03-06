import re
import uuid
import csv
import requests


class VectaraConnector:
    def __init__(self, customer_id, api_key, corpus_key):
        self._customer_id = customer_id
        self._api_key = api_key
        self._corpus_key = corpus_key

    def fetch_data(self, num_results=5, input_csv="queries.csv",
        output_csv="results.csv"):
        if not all([self._api_key, self._corpus_key, self._customer_id]):
            raise ValueError(
                "Missing Vectara API configuration (api_key, corpus_key, or customer_id)"
            )

        # Read queries from CSV file.
        queries = []
        with open(input_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_text = row.get("query")
                if not query_text:
                    continue  # skip rows without a query
                # Use provided query_id or generate one if not present.
                query_id = row.get("query_id") or str(uuid.uuid4())
                queries.append({"query": query_text, "queryId": query_id})

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": f"{self._api_key}"
        }

        endpoint_url = f"https://api.vectara.io/v2/corpora/{self._corpus_key}/query"

        # Open the output CSV file and write header.
        with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ["query_id", "query", "passage_id", "passage",
                          "generated_answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Process each query individually.
            for query in queries:
                try:
                    data = self.query(endpoint_url, headers, query, num_results)
                except Exception as e:
                    print(f"Failed to process query {query['queryId']}: {e}")
                    continue

                # Get the overall summary (generated answer).
                generated_answer = data.get("summary", "")

                if not generated_answer:
                    # Skip queries with no generated answer.
                    continue

                # Get the search results.
                search_results = data.get("search_results", [])
                for idx, result in enumerate(search_results, start=1):
                    # Only include the generated summary in the first row.
                    row = {"query_id": query["queryId"],
                           "query": query["query"],
                           "passage_id": f"[{idx}]",
                           "passage": result.get("text", ""),
                           "generated_answer": generated_answer if idx == 1 else ""}
                    writer.writerow(row)

    def query(self, endpoint_url, headers, query, num_results):
        payload = {
            "query": query["query"],
            "search": {
                "limit": num_results,
                "context_configuration": {
                    "characters_before": 30,
                    "characters_after": 30,
                    "sentences_before": 3,
                    "sentences_after": 3,
                    "start_tag": "<em>",
                    "end_tag": "</em>"
                }
            },
            "generation": {
                "generation_preset_name": "vectara-summary-ext-v1.2.0",
                "max_used_search_results": num_results,
                "max_response_characters": 300,
                "response_language": "auto",
                "enable_factual_consistency_score": False
            },
            "stream_response": False,
            "save_history": False,
            "intelligent_query_rewriting": False
        }

        response = requests.post(endpoint_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(
                f"Request failed for query_id {query['queryId']} with status {response.status_code}: {response.text}"
            )
        return response.json()
