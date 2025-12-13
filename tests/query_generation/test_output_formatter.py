"""Tests for output formatter."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from open_rag_eval.query_generation.output_formatter import OutputFormatter
from open_rag_eval.query_generation.llm_generator import QueryWithAnswer


class TestOutputFormatter(unittest.TestCase):
    """Test cases for OutputFormatter."""

    def test_save_to_csv_basic(self):
        """Test basic CSV output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?", "How does ML work?"]
            OutputFormatter.save_to_csv(queries, tmp_path)

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertIn('query_id', rows[0])
            self.assertIn('query', rows[0])
            self.assertEqual(rows[0]['query'], "What is AI?")
            self.assertEqual(rows[1]['query'], "How does ML work?")
        finally:
            Path(tmp_path).unlink()

    def test_save_to_csv_with_metadata(self):
        """Test CSV output with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?"]
            metadata = [{'word_count': 3, 'source': 'doc1'}]

            OutputFormatter.save_to_csv(
                queries,
                tmp_path,
                include_metadata=True,
                metadata=metadata
            )

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['query'], "What is AI?")
            self.assertEqual(rows[0]['word_count'], '3')
            self.assertEqual(rows[0]['source'], 'doc1')
        finally:
            Path(tmp_path).unlink()

    def test_save_to_csv_empty_queries_raises_error(self):
        """Test that empty queries list raises error."""
        with self.assertRaises(ValueError):
            OutputFormatter.save_to_csv([], "test.csv")

    def test_save_to_csv_metadata_length_mismatch_raises_error(self):
        """Test that metadata length mismatch raises error."""
        queries = ["Query 1", "Query 2"]
        metadata = [{'key': 'value'}]  # Only one metadata entry

        with self.assertRaises(ValueError):
            OutputFormatter.save_to_csv(
                queries,
                "test.csv",
                include_metadata=True,
                metadata=metadata
            )

    def test_save_to_jsonl_basic(self):
        """Test basic JSONL output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?", "How does ML work?"]
            OutputFormatter.save_to_jsonl(queries, tmp_path)

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)

            record1 = json.loads(lines[0])
            self.assertIn('query_id', record1)
            self.assertIn('query', record1)
            self.assertEqual(record1['query'], "What is AI?")

            record2 = json.loads(lines[1])
            self.assertEqual(record2['query'], "How does ML work?")
        finally:
            Path(tmp_path).unlink()

    def test_save_to_jsonl_with_metadata(self):
        """Test JSONL output with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?"]
            metadata = [{'word_count': 3, 'source': 'doc1'}]

            OutputFormatter.save_to_jsonl(
                queries,
                tmp_path,
                include_metadata=True,
                metadata=metadata
            )

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                record = json.loads(f.readline())

            self.assertEqual(record['query'], "What is AI?")
            self.assertEqual(record['word_count'], 3)
            self.assertEqual(record['source'], 'doc1')
        finally:
            Path(tmp_path).unlink()

    def test_save_queries_csv_format(self):
        """Test save_queries with CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?"]
            OutputFormatter.save_queries(queries, tmp_path, output_format='csv')

            # Verify file was created and is CSV
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
        finally:
            Path(tmp_path).unlink()

    def test_save_queries_jsonl_format(self):
        """Test save_queries with JSONL format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = ["What is AI?"]
            OutputFormatter.save_queries(queries, tmp_path, output_format='jsonl')

            # Verify file was created and is JSONL
            with open(tmp_path, 'r', encoding='utf-8') as f:
                record = json.loads(f.readline())

            self.assertIn('query', record)
        finally:
            Path(tmp_path).unlink()

    def test_save_queries_invalid_format_raises_error(self):
        """Test that invalid format raises error."""
        queries = ["What is AI?"]

        with self.assertRaises(ValueError):
            OutputFormatter.save_queries(queries, "test.txt", output_format='txt')

    def test_save_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "queries.csv"
            queries = ["What is AI?"]

            OutputFormatter.save_to_csv(queries, str(output_path))

            self.assertTrue(output_path.exists())


class TestOutputFormatterQAPairs(unittest.TestCase):
    """Test cases for OutputFormatter QA pair methods."""

    def test_save_qa_pairs_to_csv_basic(self):
        """Test basic QA pair CSV output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence."),
                QueryWithAnswer("How does ML work?", "ML learns from data.")
            ]
            OutputFormatter.save_qa_pairs_to_csv(qa_pairs, tmp_path)

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertIn('query_id', rows[0])
            self.assertIn('query', rows[0])
            self.assertIn('expected_answer', rows[0])
            self.assertEqual(rows[0]['query'], "What is AI?")
            self.assertEqual(rows[0]['expected_answer'], "AI is artificial intelligence.")
            self.assertEqual(rows[1]['query'], "How does ML work?")
            self.assertEqual(rows[1]['expected_answer'], "ML learns from data.")
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_to_csv_with_metadata(self):
        """Test QA pair CSV output with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
            ]
            metadata = [{'source': 'doc1', 'difficulty': 'easy'}]

            OutputFormatter.save_qa_pairs_to_csv(
                qa_pairs,
                tmp_path,
                include_metadata=True,
                metadata=metadata
            )

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['query'], "What is AI?")
            self.assertEqual(rows[0]['expected_answer'], "AI is artificial intelligence.")
            self.assertEqual(rows[0]['source'], 'doc1')
            self.assertEqual(rows[0]['difficulty'], 'easy')
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_to_csv_empty_raises_error(self):
        """Test that empty QA pairs list raises error."""
        with self.assertRaises(ValueError):
            OutputFormatter.save_qa_pairs_to_csv([], "test.csv")

    def test_save_qa_pairs_to_csv_metadata_mismatch_raises_error(self):
        """Test that metadata length mismatch raises error."""
        qa_pairs = [
            QueryWithAnswer("Q1?", "A1"),
            QueryWithAnswer("Q2?", "A2")
        ]
        metadata = [{'key': 'value'}]  # Only one metadata entry

        with self.assertRaises(ValueError):
            OutputFormatter.save_qa_pairs_to_csv(
                qa_pairs,
                "test.csv",
                include_metadata=True,
                metadata=metadata
            )

    def test_save_qa_pairs_to_jsonl_basic(self):
        """Test basic QA pair JSONL output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence."),
                QueryWithAnswer("How does ML work?", "ML learns from data.")
            ]
            OutputFormatter.save_qa_pairs_to_jsonl(qa_pairs, tmp_path)

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)

            record1 = json.loads(lines[0])
            self.assertIn('query_id', record1)
            self.assertIn('query', record1)
            self.assertIn('expected_answer', record1)
            self.assertEqual(record1['query'], "What is AI?")
            self.assertEqual(record1['expected_answer'], "AI is artificial intelligence.")

            record2 = json.loads(lines[1])
            self.assertEqual(record2['query'], "How does ML work?")
            self.assertEqual(record2['expected_answer'], "ML learns from data.")
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_to_jsonl_with_metadata(self):
        """Test QA pair JSONL output with metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
            ]
            metadata = [{'source': 'doc1', 'difficulty': 'easy'}]

            OutputFormatter.save_qa_pairs_to_jsonl(
                qa_pairs,
                tmp_path,
                include_metadata=True,
                metadata=metadata
            )

            # Read and verify
            with open(tmp_path, 'r', encoding='utf-8') as f:
                record = json.loads(f.readline())

            self.assertEqual(record['query'], "What is AI?")
            self.assertEqual(record['expected_answer'], "AI is artificial intelligence.")
            self.assertEqual(record['source'], 'doc1')
            self.assertEqual(record['difficulty'], 'easy')
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_csv_format(self):
        """Test save_qa_pairs with CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
            ]
            OutputFormatter.save_qa_pairs(qa_pairs, tmp_path, output_format='csv')

            # Verify file was created and is CSV with expected_answer
            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertIn('expected_answer', rows[0])
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_jsonl_format(self):
        """Test save_qa_pairs with JSONL format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
            ]
            OutputFormatter.save_qa_pairs(qa_pairs, tmp_path, output_format='jsonl')

            # Verify file was created and is JSONL with expected_answer
            with open(tmp_path, 'r', encoding='utf-8') as f:
                record = json.loads(f.readline())

            self.assertIn('query', record)
            self.assertIn('expected_answer', record)
        finally:
            Path(tmp_path).unlink()

    def test_save_qa_pairs_invalid_format_raises_error(self):
        """Test that invalid format raises error."""
        qa_pairs = [
            QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
        ]

        with self.assertRaises(ValueError):
            OutputFormatter.save_qa_pairs(qa_pairs, "test.txt", output_format='txt')

    def test_save_qa_pairs_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "qa_pairs.csv"
            qa_pairs = [
                QueryWithAnswer("What is AI?", "AI is artificial intelligence.")
            ]

            OutputFormatter.save_qa_pairs_to_csv(qa_pairs, str(output_path))

            self.assertTrue(output_path.exists())


if __name__ == '__main__':
    unittest.main()
