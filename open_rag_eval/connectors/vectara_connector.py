from itertools import islice
import logging
import os

import requests
from tqdm import tqdm
import omegaconf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.utils.constants import NO_ANSWER, API_ERROR

# Configure logging for tenacity
logger = logging.getLogger(__name__)


# Custom callback for tqdm progress bar with tenacity
def tqdm_progress_callback(retry_state):
    """Show progress with tqdm during retries."""
    # Extract query_id from the args passed to the function
    # The query_id should be the 4th argument to _make_request
    try:
        query_id = retry_state.args[3] if len(
            retry_state.args) > 3 else "unknown"
    except (IndexError, AttributeError):
        query_id = "unknown"

    if not hasattr(retry_state, "tqdm_pbar"):
        retry_state.tqdm_pbar = tqdm(
            total=retry_state.retry_object.stop.max_attempt_number,
            desc=f"Retrying query {query_id}")
    retry_state.tqdm_pbar.update(1)

    # Close the progress bar on the last attempt
    if retry_state.attempt_number == retry_state.retry_object.stop.max_attempt_number:
        retry_state.tqdm_pbar.close()


class VectaraConnector(Connector):

    def __init__(self,
                 config: dict,
                 api_key: str,
                 corpus_key: str,
                 max_workers: int = -1,
                 repeat_query: int = 1,
                 query_config: dict = None) -> None:
        self.config = config
        self._api_key = api_key
        self._corpus_key = corpus_key
        self.query_config = query_config

        # Configuration for paths
        queries_csv = config.get("input_queries", "")
        results_folder = config.get("results_folder",
                                    ".")  # Default to current directory
        generated_answers_filename = config.get(
            "generated_answers", "vectara_generated_answers.csv")
        outputs_csv = os.path.join(results_folder, generated_answers_filename)

        # Initialize base class with calculated values
        super().__init__(queries_csv=queries_csv,
                         output_path=outputs_csv,
                         max_workers=max_workers,
                         repeat_query=repeat_query)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=1, max=60),
           retry=retry_if_exception_type(
               (requests.exceptions.RequestException, Exception)),
           before_sleep=tqdm_progress_callback,
           after=tqdm_progress_callback)
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
            response = requests.post(endpoint_url,
                                     headers=headers,
                                     json=payload,
                                     timeout=30)
            if response.status_code == 200:
                return response.json()
            error = f"Request failed for query_id {query_id} with status {response.status_code}: {response.text}"
            logger.error(error)
            raise Exception(error)
        except requests.exceptions.RequestException as ex:
            raise Exception(
                f"Request failed for query_id {query_id}: {str(ex)}") from ex

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
        search = query_config.search
        generation = query_config.generation

        search_dict = (omegaconf.OmegaConf.to_container(search, resolve=True)
                       if isinstance(search, omegaconf.dictconfig.DictConfig)
                       else search)
        generation_dict = (omegaconf.OmegaConf.to_container(
            generation, resolve=True) if isinstance(
                generation, omegaconf.dictconfig.DictConfig) else generation)

        # Check if prompt_template is specified and load from file if it exists
        prompt_template_file = generation.get("prompt_template")
        if prompt_template_file and os.path.exists(prompt_template_file):
            try:
                with open(prompt_template_file, "r", encoding="utf-8") as f:
                    prompt_content = f.read().strip()
                    generation_dict["prompt_template"] = prompt_content
            except Exception as e:
                generation_dict.pop("prompt_template", None)
                logger.warning("Failed to read prompt template file %s: %s",
                               prompt_template_file, e)
        else:
            generation_dict.pop("prompt_template", None)

        payload = {
            "query":
                query["query"],
            "search":
                search_dict,
            "generation":
                generation_dict,
            "stream_response": (query_config or {}).get("stream_response",
                                                        False),
            "save_history": (query_config or {}).get("save_history", False),
            "intelligent_query_rewriting":
                (query_config or {}).get("intelligent_query_rewriting", False),
        }

        # Use the default retry configuration
        return self._send_request(endpoint_url, headers, payload,
                                  query["queryId"])

    def process_query(self, query, run_idx=1):
        if not all([self._api_key, self._corpus_key]):
            raise ValueError(
                "Missing Vectara API configuration (api_key, corpus_key)")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": f"{self._api_key}"
        }

        endpoint_url = f"https://api.vectara.io/v2/corpora/{self._corpus_key}/query"

        return self.execute_vectara_query(query, endpoint_url, headers, self.query_config, run_idx)

    def execute_vectara_query(self, query, endpoint_url, headers, query_config, run_idx):
        """
        Process a single query by sending it to the Vectara API and handling the response.
        Args:
            query: The query dictionary containing "query" and "queryId"
            endpoint_url: The Vectara API endpoint URL
            headers: Request headers
            query_config: Configuration for search and generation
            run_idx: The index of the query run (1-based)
        """
        try:
            data = self.query(endpoint_url, headers, query, query_config)

            # Get the overall summary (generated answer).
            generated_answer = data.get("summary", "")

            if not generated_answer:
                logger.warning("No generated answer for query %s. Skipping...",
                               query["query"])
                return [{
                    "query_id": query["queryId"],
                    "query": query["query"],
                    "query_run": run_idx,
                    "passage_id": "NA",
                    "passage": "NA",
                    "generated_answer": NO_ANSWER
                }]

            # Get the search results.
            results = []
            search_results = data.get("search_results", [])
            num_results_used = query_config.generation.get(
                "max_used_search_results")
            for idx, result in enumerate(islice(search_results,
                                                num_results_used),
                                         start=1):
                # Only include the generated summary in the first row.
                row = {
                    "query_id": query["queryId"],
                    "query": query["query"],
                    "query_run": run_idx,
                    "passage_id": f"[{idx}]",
                    "passage": result.get("text", ""),
                    "generated_answer": generated_answer if idx == 1 else ""
                }
                results.append(row)
            return results
        except Exception as ex:
            logger.error(
                "Failed to process query_id %s ('%s'): %s",
                query["queryId"],
                query["query"],
                str(ex),
                exc_info=True,
            )
            return [{
                "query_id": query["queryId"],
                "query": query["query"],
                "query_run": run_idx,
                "passage_id": "ERROR",
                "passage": f"Runtime error: {ex}",
                "generated_answer": API_ERROR,
            }]
