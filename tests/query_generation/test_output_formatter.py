"""Tests for output formatter."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from open_rag_eval.query_generation.output_formatter import OutputFormatter


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


if __name__ == '__main__':
    unittest.main()
