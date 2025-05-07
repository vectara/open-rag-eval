from abc import abstractmethod
import csv
import uuid

class Connector:

    @abstractmethod
    def fetch_data(
        self,
    ) -> None:
        pass

    def read_queries(self, queries_file_name: str) -> list:
        """
        Read queries from a CSV file. The CSV file should have a header with
        "query" and optionally "query_id" columns. If "query_id" is not present,
        a unique ID will be generated for each query.
        """
        queries = []
        with open(queries_file_name, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_text = row.get("query")
                if not query_text:
                    print(f"Skipping row without query: {row}")
                    continue  # skip rows without a query
                # Use provided query_id or generate one if not present.
                query_id = row.get("query_id") or str(uuid.uuid4())
                queries.append({"query": query_text, "queryId": query_id})

        return queries
