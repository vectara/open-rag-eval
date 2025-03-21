import unittest
from pathlib import Path
from connectors.csv_connector import CSVConnector
from data_classes.rag_results import GeneratedAnswerPart

class TestCSVConnector(unittest.TestCase):
    def setUp(self):
        test_csv_path = Path("data/test_csv_connector.csv")
        self.connector = CSVConnector(test_csv_path)
        
    def test_read_results(self):
        results = self.connector.fetch_data()
        
        # Should return 3 RAGResults (one per query_id)
        self.assertEqual(len(results), 3)
        
        # Test first result (query_1)
        result1 = results[0]
        self.assertEqual(result1.retrieval_result.query, "What is a blackhole?")
        self.assertEqual(len(result1.retrieval_result.retrieved_passages), 3)
        self.assertEqual(
            result1.retrieval_result.retrieved_passages["[1]"],
            "Blackholes are big."
        )
        self.assertEqual(
            result1.retrieval_result.retrieved_passages["[2]"],
            "Blackholes are dense."
        )
        self.assertEqual(
            result1.retrieval_result.retrieved_passages["[3]"],
            "The earth is round."
        )
        self.assertEqual(
            result1.generation_result.generated_answer,
            [
                GeneratedAnswerPart(text="Black holes are dense", citations=["[1]"]),
                GeneratedAnswerPart(text="They are also big and round", citations=["[2]", "[3]"]),
                GeneratedAnswerPart(text="My name is x", citations=['[4]'])
            ]
        )

        # Test second result (query_2) 
        result2 = results[1]
        self.assertEqual(result2.retrieval_result.query, "How big is the sun?")
        self.assertEqual(len(result2.retrieval_result.retrieved_passages), 1)
        self.assertEqual(
            result2.retrieval_result.retrieved_passages["[1]"],
            "The sun is several million kilometers in diameter."
        )        
        self.assertEqual(
            result2.generation_result.generated_answer,
            [GeneratedAnswerPart(text="The sun is several million kilometers in diameter", citations=["[1]"])]
        )

        # Test third result (query_3)
        result3 = results[2]
        self.assertEqual(result3.retrieval_result.query, "How many planets have moons?")
        self.assertEqual(len(result3.retrieval_result.retrieved_passages), 2)
        self.assertEqual(
            result3.retrieval_result.retrieved_passages["[2]"],
            "Seven planets have a moon."
        )        
        self.assertEqual(
            result3.retrieval_result.retrieved_passages["[3]"],
            "Earth has a moon."
        )                        
        self.assertEqual(
            result3.generation_result.generated_answer,
            [GeneratedAnswerPart(text="Seven", citations=["[2]"])]
        )

if __name__ == '__main__':
    unittest.main()