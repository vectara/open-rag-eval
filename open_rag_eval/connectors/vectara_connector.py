import csv
from itertools import islice
import logging
import uuid

import requests
from tqdm import tqdm
import omegaconf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from open_rag_eval.connectors.connector import Connector

# Configure logging for tenacity
logger = logging.getLogger(__name__)

DEFAULT_MAX_USED_SEARCH_RESULTS = 5

# Custom callback for tqdm progress bar with tenacity
def tqdm_progress_callback(retry_state):
    """Show progress with tqdm during retries."""
    # Extract query_id from the args passed to the function
    # The query_id should be the 4th argument to _make_request
    try:
        query_id = retry_state.args[3] if len(retry_state.args) > 3 else "unknown"
    except (IndexError, AttributeError):
        query_id = "unknown"

    if not hasattr(retry_state, 'tqdm_pbar'):
        retry_state.tqdm_pbar = tqdm(
            total=retry_state.retry_object.stop.max_attempt_number,
            desc=f"Retrying query {query_id}"
        )
    retry_state.tqdm_pbar.update(1)

    # Close the progress bar on the last attempt
    if retry_state.attempt_number == retry_state.retry_object.stop.max_attempt_number:
        retry_state.tqdm_pbar.close()

class VectaraConnector(Connector):
    def __init__(self, api_key, corpus_key):
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
                "max_used_search_results": DEFAULT_MAX_USED_SEARCH_RESULTS,
                "response_language": "auto",
                "citations": {"style": "numeric"},
            }
        }

    def _get_config_section(self, query_config, section_name):
        """
        Extract a configuration section from query_config or use defaults.

        Args:
            query_config: The query configuration
            section_name: The name of the section to extract (e.g., 'search', 'generation')

        Returns:
            The configuration section, with defaults applied if needed
        """
        if query_config is None:
            return self.default_config[section_name]
        return query_config.get(section_name, self.default_config[section_name])

    def _get_max_used_search_results(self, query_config):
        """
        Get the maximum number of search results to use for generation.
        """
        generation = self._get_config_section(query_config, 'generation')
        return generation.get("max_used_search_results", DEFAULT_MAX_USED_SEARCH_RESULTS)

    def fetch_data(self, query_config=None, input_csv="queries.csv", output_csv="results.csv"):
        if not all([self._api_key, self._corpus_key]):
            raise ValueError(
                "Missing Vectara API configuration (api_key, corpus_key)"
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
                except Exception as ex:
                    print(f"Failed to process query {query['queryId']}: {ex}")
                    continue

                # Get the overall summary (generated answer).
                generated_answer = data.get("summary", "")

                if not generated_answer:
                    # Skip queries with no generated answer.
                    continue

                # Get the search results.
                search_results = data.get("search_results", [])
                for idx, result in enumerate(islice(search_results, self._get_max_used_search_results(query_config)), start=1):
                    # Only include the generated summary in the first row.
                    row = {"query_id": query["queryId"],
                           "query": query["query"],
                           "passage_id": f"[{idx}]",
                           "passage": result.get("text", ""),
                           "generated_answer": generated_answer if idx == 1 else ""}
                    writer.writerow(row)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        before_sleep=tqdm_progress_callback,
        after=tqdm_progress_callback
    )
    def _send_request(self, endpoint_url, headers, payload, query_id):
        """
        Send request to Vectara API with retry logic using tenacity.

        This method is decorated with tenacity's retry decorator to automatically
        handle retries with exponential backoff for failed requests.

        Args:
            endpoint_url: The Vectara API endpoint URL
            headers: Request headers
            payload: Request payload (JSON)
            query_id: Query ID for tracking and reporting

        Returns:
            JSON response from the Vectara API

        Raises:
            Exception: If the request fails after all retries
        """
        try:
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            raise Exception(
                f"Request failed for query_id {query_id} with status {response.status_code}: {response.text}"
            )
        except requests.exceptions.RequestException as ex:
            raise Exception(
                f'Request failed for query_id {query_id}: {str(ex)}') from ex

    def query(self, endpoint_url, headers, query, query_config):
        """
        Query the Vectara API with retry logic.

        Args:
            endpoint_url: The Vectara API endpoint URL
            headers: Request headers
            query: Query parameters
            query_config: Configuration for search and generation

        Returns:
            JSON response from the Vectara API
        """
        # Get configs or use defaults
        search = self._get_config_section(query_config, 'search')
        generation = self._get_config_section(query_config, 'generation')

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

        # Use the default retry configuration
        return self._send_request(endpoint_url, headers, payload, query["queryId"])
