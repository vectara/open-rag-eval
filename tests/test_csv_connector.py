import unittest
from pathlib import Path
from open_rag_eval.connectors.csv_connector import CSVConnector
from open_rag_eval.data_classes.rag_results import GeneratedAnswerPart

class TestCSVConnector(unittest.TestCase):
    def setUp(self):
        test_csv_path = Path("tests/data/test_csv_connector.csv")
        self.connector = CSVConnector(test_csv_path)

    def test_read_results(self):
        results = self.connector.fetch_data()

        # Should return 5 RAGResults (one per query_id)
        self.assertEqual(len(results), 5)

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

        # Test fourth result (query_4)
        result4 = results[3]
        self.assertEqual(result4.retrieval_result.query, "What does the Dodd-Frank Act regulate?")
        self.assertEqual(len(result4.retrieval_result.retrieved_passages), 2)
        self.assertEqual(
            result4.retrieval_result.retrieved_passages["[1]"],
            "The Dodd-Frank Act regulates the use of credit ratings by financial institutions, requiring them to perform due diligence before purchasing financial instruments."
        )
        self.assertEqual(
            result4.retrieval_result.retrieved_passages["[2]"],
            "The Dodd-Frank Act was passed in 2010 in response to the financial crisis."
        )
        self.assertEqual(
            result4.generation_result.generated_answer,
            [GeneratedAnswerPart(text='Based on the provided sources, the DFA also regulates the use of credit ratings by financial institutions, requiring them to do their own due diligence before buying financial instruments', citations=['[1]']), GeneratedAnswerPart(text='.\n\nNo other information about the DFA is provided in the given sources.\n\nSources:', citations=['[1]', '[2]'])]
        )

        # Test fifth result (query_5)
        result5 = results[4]
        self.assertEqual(result5.retrieval_result.query, "What is the purpose of job training?")
        self.assertEqual(len(result5.retrieval_result.retrieved_passages), 5)
        self.assertEqual(
            result5.retrieval_result.retrieved_passages["[1]"],
            "Job training helps individuals acquire marketable skills necessary for specific jobs or professions."
        )
        self.assertEqual(
            result5.retrieval_result.retrieved_passages["[2]"],
            "Training can occur on the job, through formal education, or specialized programs."
        )
        self.assertEqual(
            result5.retrieval_result.retrieved_passages["[3]"],
            "The primary purpose of job training is to equip workers with essential skills and knowledge."
        )
        self.assertEqual(
            result5.retrieval_result.retrieved_passages["[4]"],
            "Effective job training increases an individual's chances of getting hired."
        )
        self.assertEqual(
            result5.retrieval_result.retrieved_passages["[5]"],
            "Job training can also improve promotion opportunities for employees."
        )
        self.assertEqual(
            result5.generation_result.generated_answer,
            [GeneratedAnswerPart(text='Based on the provided sources, the purpose of job training is to acquire marketable skills that are necessary for a particular job or profession. This training can be done on the job, through formal education, or through specialized training programs. The purpose of job training is to equip individuals with the skills and knowledge they need to perform their jobs effectively and to increase their chances of getting hired or promoted.', citations=['[1]', '[2]', '[3]', '[4]', '[5]'])]
        )


if __name__ == '__main__':
    unittest.main()
