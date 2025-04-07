import uuid
import csv
import requests

from tqdm import tqdm
import omegaconf

from open_rag_eval.connectors.connector import Connector

class VectaraConnector(Connector):
    def __init__(self, customer_id, api_key, corpus_key):
        self._customer_id = customer_id
        self._api_key = api_key
        self._corpus_key = corpus_key

        self.default_config = {
            "search": {
                "lexical_interpolation": 0.005,
                "limit": 100,
                "context_configuration": {
                    "sentences_before": 3,
                    "sentences_after": 3,
                    "start_tag": "<em>",
                    "end_tag": "</em>"
                },
                "reranker": {
                    "type": "chain",
                    "rerankers": [
                        {
                            "type": "customer_reranker",
                            "reranker_name": "Rerank_Multilingual_v1",
                            "limit": 50
                        },
                        {
                            "type": "mmr",
                            "diversity_bias": 0.01,
                            "limit": 10
                        }
                    ]
                }
            },
            "generation": {
                "generation_preset_name": "vectara-summary-table-md-query-ext-jan-2025-gpt-4o",
                "max_used_search_results": 5,
                "response_language": "auto",
                "citations": {"style": "numeric"},
            }
        }

    def fetch_data(self, query_config=None, input_csv="queries.csv", output_csv="results.csv"):
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
                    print(f"Skipping row without query: {row}")
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
        if query_config is None:
            search = self.default_config['search']
            generation = self.default_config['generation']
        else:
            search = query_config.get('search', self.default_config['search'])
            generation = query_config.get('generation', self.default_config['generation'])

        search_dict = (
            omegaconf.OmegaConf.to_container(search, resolve=True)
            if isinstance(search, omegaconf.dictconfig.DictConfig)
            else search
        )
        generation_dict = (
            omegaconf.OmegaConf.to_container(generation, resolve=True)
            if isinstance(generation, omegaconf.dictconfig.DictConfig)
            else generation
        )

        payload = {
            "query": query["query"],
            "search": search_dict,
            "generation": generation_dict,
            "stream_response": False,
            "save_history": True,
            "intelligent_query_rewriting": False
        }

        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(
                f"Request failed for query_id {query['queryId']} with status {response.status_code}: {response.text}"
            )
        return response.json()
