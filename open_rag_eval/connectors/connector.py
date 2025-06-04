import os
from abc import abstractmethod, ABC
import csv
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Connector(ABC):

    def __init__(self,
                 queries_csv,
                 output_path,
                 max_workers=-1,
                 repeat_query=1):
        if not queries_csv:
            logger.error("Config dictionary must contain 'input_queries' path.")
            raise ValueError(
                "Config dictionary must contain 'input_queries' path.")
        self.queries_csv = queries_csv
        self.output_path = output_path
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if max_workers == -1:
            self.max_workers = min(32, os.cpu_count() * 4)
        else:
            self.max_workers = max_workers
        self.parallel = max_workers > 1
        self.repeat_query = repeat_query

    def fetch_data(self) -> None:
        """Common implementation of fetch_data that handles parallelization and result writing."""
        queries = self.read_queries()
        fieldnames = self.get_fieldnames()
        output_path = self.output_path

        if not self.parallel:
            with open(output_path, "w", newline="",
                      encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for query in tqdm(
                        queries,
                        total=len(queries) * self.repeat_query,
                        desc=f"Processing {self.get_connector_name()} queries"):
                    for run_idx in range(self.repeat_query):
                        results = self.process_query(query, run_idx + 1)
                        if results:
                            for row in results:
                                writer.writerow(row)
        else:
            # Create repeated queries based on self.repeat_query
            repeated_queries = [(query, run_idx + 1)
                                for query in queries
                                for run_idx in range(self.repeat_query)]

            results_buffer = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query_info = {
                    executor.submit(self.process_query, query, run_idx):
                        (i, run_idx)
                    for i, (query, run_idx) in enumerate(repeated_queries)
                }
                for future in tqdm(
                        as_completed(future_to_query_info),
                        total=len(queries) * self.repeat_query,
                        desc=f"Processing {self.get_connector_name()} queries"):
                    idx, run_idx = future_to_query_info[future]
                    results = future.result()
                    if results:
                        for row in results:
                            results_buffer.append((idx, row))

            with open(output_path, "w", newline="",
                      encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                results_buffer.sort(key=lambda x: x[0])
                for _, row in results_buffer:
                    writer.writerow(row)

        logger.info("%s query processing is complete. Results saved to %s",
                    self.get_connector_name(), output_path)

    def read_queries(self) -> list:
        """
        Read queries from a CSV file. The CSV file should have a header with
        "query" and optionally "query_id" columns. If "query_id" is not present,
        a unique ID will be generated for each query.
        """
        queries_file_name = self.queries_csv
        queries = []
        with open(queries_file_name, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_text = row.get("query")
                if not (query_text and query_text.strip()):
                    print(f"Skipping row without query: {row}")
                    continue  # skip rows without a query
                # Use provided query_id or generate one if not present.
                query_id = row.get("query_id") or str(uuid.uuid4())
                queries.append({"query": query_text, "queryId": query_id})

        return queries

    def get_fieldnames(self):
        """Return the fieldnames for the CSV output."""
        return [
            "query_id", "query", "query_run", "passage_id", "passage",
            "generated_answer"
        ]

    def get_connector_name(self):
        """Return the name of the connector for logging and progress bars."""
        return self.__class__.__name__

    @abstractmethod
    def process_query(self, query, run_idx=1):
        """Process a single query and return a list of result dictionaries."""
        pass
