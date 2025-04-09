from dotenv import load_dotenv
import os
import csv
import unittest
from pathlib import Path
from open_rag_eval.connectors.vectara_connector import VectaraConnector

class TestVectaraConnectorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        # Check for required environment variables; skip integration tests if missing.
        required_vars = ["VECTARA_API_KEY", "VECTARA_CORPUS_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise unittest.SkipTest(
                f"Skipping integration tests, missing env vars: {', '.join(missing)}")

        cls.api_key = os.getenv("VECTARA_API_KEY")
        cls.corpus_key = os.getenv("VECTARA_CORPUS_KEY")

        # Create a temporary CSV file with one test query.
        cls.test_csv_path = Path("tests/data/test_vectara_connector_integration.csv")
        cls.generated_answers = "results_integration.csv"
        cls.connector = VectaraConnector(cls.api_key, cls.corpus_key)

    @classmethod
    def tearDownClass(cls):
        output_path = Path(cls.generated_answers)
        if output_path.exists():
            output_path.unlink()

    def test_fetch_data_integration(self):
        # This integration test hits the real Vectara API.
        self.connector.fetch_data(input_csv=str(self.test_csv_path), output_csv=self.generated_answers)

        # Verify that the output CSV file exists.
        output_path = Path(self.generated_answers)
        self.assertTrue(output_path.exists(), "Output CSV was not created.")

        # Read the output CSV and check that it contains at least one row.
        with output_path.open("r", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        self.assertGreater(len(rows), 0, "Output CSV is empty.")

        # Optional: Print out the CSV content for debugging.
        for row in rows:
            print(row)


if __name__ == '__main__':
    unittest.main()
