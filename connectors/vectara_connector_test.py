import os
import csv
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from connectors.vectara_connector import VectaraConnector

# Dummy response JSON to simulate the Vectara API response.
DUMMY_RESPONSE = {
    "summary": "Test summary[1]",
    "search_results": [
        {"text": "Passage one"},
        {"text": "Passage two"}
    ]
}


class TestVectaraConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with one test query.
        self.test_csv_path = Path("../data/test_vectara_connector.csv")
        self.test_csv_path.parent.mkdir(exist_ok=True)
        with self.test_csv_path.open("w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "query"])
            writer.writeheader()
            writer.writerow({"query_id": "query_1",
                             "query": "What is the meaning of life?"})

        # Retrieve test credentials (or set dummy values for unit testing)
        api_key = os.getenv("VECTARA_API_KEY", "dummy_api_key")
        corpus_key = os.getenv("VECTARA_CORPUS_KEY", "dummy_corpus_key")
        customer_id = os.getenv("VECTARA_CUSTOMER_ID", "dummy_customer_id")
        self.connector = VectaraConnector(customer_id, api_key, corpus_key)

        # Output CSV file for testing.
        self.output_csv = "results.csv"

    def tearDown(self):
        # Cleanup the temporary test CSV and output CSV.
        if self.test_csv_path.exists():
            self.test_csv_path.unlink()
        if Path(self.output_csv).exists():
            Path(self.output_csv).unlink()

    @patch("requests.post")
    def test_fetch_data(self, mock_post):
        # Configure the mock to return a dummy response.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = DUMMY_RESPONSE
        mock_post.return_value = mock_response

        # Call the fetch_data method.
        self.connector.fetch_data(str(self.test_csv_path), self.output_csv)

        # Now read the output CSV and validate its contents.
        with open(self.output_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # We expect two rows since our dummy response returns 1 search result with citations.
        self.assertEqual(len(rows), 1)

        # Check the first row: it should have the generated summary and passage_id "[1]"
        row1 = rows[0]
        self.assertEqual(row1["query_id"], "query_1")
        self.assertEqual(row1["query"], "What is the meaning of life?")
        self.assertEqual(row1["passage_id"], "[1]")
        self.assertEqual(row1["passage"], "Passage one")
        self.assertEqual(row1["generated_answer"], "Test summary[1]")


if __name__ == '__main__':
    unittest.main()
