import re
import uuid
import csv
import requests

from tqdm import tqdm

from connectors.connector import Connector

class VectaraConnector(Connector):
    def __init__(self, customer_id, api_key, corpus_key):
        self._customer_id = customer_id
        self._api_key = api_key
        self._corpus_key = corpus_key

        self.default_config = {
            "search": {
                "num_results": 5,
                "context_configuration": {
                    "sentences_before": 3,
                    "sentences_after": 3,
                    "start_tag": "<em>",
                    "end_tag": "</em>"
                }
            },
            "generation": {
                "generation_preset_name": "vectara-summary-ext-v1.2.0",
                "max_used_search_results": 5,
                "max_response_characters": 1000,
                "response_language": "auto",
            }
        }

    def fetch_data(self, query_config = {}, input_csv="queries.csv", output_csv="results.csv"):
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
            for query in tqdm(queries, desc="Running Vectara queries"):
                try:
                    data = self.query(endpoint_url, headers, query, query_config)
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

    def query(self, endpoint_url, headers, query, query_config):
        # Get configs or use defaults
        search_config = query_config.get('search',
                                          self.default_config['search'])
        num_results = search_config.get('num_results',
                                         self.default_config['search']['num_results'])
        context_config = search_config.get('context_configuration',
                                            self.default_config['search']['context_configuration'])
        generation_config = query_config.get('generation',
                                              self.default_config['generation'])

        payload = {
            "query": query["query"],
            "search": {
                "limit": num_results,
                "context_configuration": {                    
                    "sentences_before": context_config.get('sentences_before',
                                                           self.default_config['search']['context_configuration']['sentences_before']),
                    "sentences_after": context_config.get('sentences_after',
                                                          self.default_config['search']['context_configuration']['sentences_after']),
                    "start_tag": context_config.get('start_tag',
                                                    self.default_config['search']['context_configuration']['start_tag']),
                    "end_tag": context_config.get('end_tag',
                                                  self.default_config['search']['context_configuration']['end_tag'])
                }
            },
            "generation": {
                "generation_preset_name": generation_config.get('generation_preset_name',
                                                               self.default_config['generation']['generation_preset_name']),
                "max_used_search_results": generation_config.get('max_used_search_results',
                                                                 self.default_config['generation']['max_used_search_results']),
                "max_response_characters": generation_config.get('max_response_characters',
                                                                self.default_config['generation']['max_response_characters']),
                "response_language": generation_config.get('response_language',
                                                          self.default_config['generation']['response_language'])
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
