"""Tests for document source adapters."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from open_rag_eval.query_generation.document_sources import (
    VectaraCorpusSource,
    LocalFileSource,
    CSVSource,
)


class TestVectaraCorpusSource(unittest.TestCase):
    """Test cases for VectaraCorpusSource."""

    def test_init_requires_api_key(self):
        """Test that API key is required."""
        with self.assertRaises(ValueError):
            VectaraCorpusSource(api_key="", corpus_key="test")

    def test_init_requires_corpus_key(self):
        """Test that corpus key is required."""
        with self.assertRaises(ValueError):
            VectaraCorpusSource(api_key="test", corpus_key="")

    @patch('open_rag_eval.query_generation.document_sources.requests.Session')
    def test_list_docs_success(self, mock_session_class):
        """Test successful document listing."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'documents': [
                {'id': 'doc1'},
                {'id': 'doc2'}
            ],
            'metadata': {}
        }
        mock_session.get.return_value = mock_response

        source = VectaraCorpusSource(api_key="test_key", corpus_key="test_corpus")
        doc_ids = source.list_docs()

        self.assertEqual(len(doc_ids), 2)
        self.assertIn('doc1', doc_ids)
        self.assertIn('doc2', doc_ids)

    @patch('open_rag_eval.query_generation.document_sources.requests.Session')
    def test_list_docs_handles_pagination(self, mock_session_class):
        """Test that pagination is handled correctly."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock paginated responses
        responses = [
            Mock(status_code=200, json=lambda: {
                'documents': [{'id': 'doc1'}],
                'metadata': {'page_key': 'next_page'}
            }),
            Mock(status_code=200, json=lambda: {
                'documents': [{'id': 'doc2'}],
                'metadata': {}
            })
        ]
        mock_session.get.side_effect = responses

        source = VectaraCorpusSource(api_key="test_key", corpus_key="test_corpus")
        doc_ids = source.list_docs()

        self.assertEqual(len(doc_ids), 2)
        self.assertEqual(mock_session.get.call_count, 2)

    @patch('open_rag_eval.query_generation.document_sources.requests.Session')
    def test_list_docs_handles_errors(self, mock_session_class):
        """Test error handling in document listing."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_session.get.return_value = mock_response

        source = VectaraCorpusSource(api_key="test_key", corpus_key="test_corpus")

        with self.assertRaises(IOError):
            source.list_docs()

    @patch('open_rag_eval.query_generation.document_sources.requests.Session')
    def test_get_doc_text_success(self, mock_session_class):
        """Test successful document text retrieval."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'parts': [
                {'text': 'Part 1'},
                {'text': 'Part 2'}
            ]
        }
        mock_session.get.return_value = mock_response

        source = VectaraCorpusSource(api_key="test_key", corpus_key="test_corpus")
        text = source.get_doc_text('doc1')

        self.assertEqual(text, 'Part 1\nPart 2')


class TestLocalFileSource(unittest.TestCase):
    """Test cases for LocalFileSource."""

    def test_init_requires_path(self):
        """Test that path is required."""
        with self.assertRaises(ValueError):
            LocalFileSource(path="")

    def test_init_validates_path_exists(self):
        """Test that path must exist."""
        with self.assertRaises(ValueError):
            LocalFileSource(path="/nonexistent/path")

    def test_init_validates_path_is_directory(self):
        """Test that path must be a directory."""
        with tempfile.NamedTemporaryFile() as tmp:
            with self.assertRaises(ValueError):
                LocalFileSource(path=tmp.name)

    def test_load_documents_from_files(self):
        """Test loading documents from local files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file1 = Path(tmpdir) / "test1.txt"
            test_file2 = Path(tmpdir) / "test2.txt"
            test_file1.write_text("This is document 1")
            test_file2.write_text("This is document 2")

            source = LocalFileSource(path=tmpdir)
            documents = source.fetch_random_documents()

            self.assertEqual(len(documents), 2)
            self.assertIn("This is document 1", documents)
            self.assertIn("This is document 2", documents)

    def test_load_documents_filters_by_extension(self):
        """Test that only specified extensions are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            (Path(tmpdir) / "test.txt").write_text("Text file")
            (Path(tmpdir) / "test.md").write_text("Markdown file")
            (Path(tmpdir) / "test.pdf").write_text("PDF file")

            source = LocalFileSource(path=tmpdir, file_extensions=['.txt'])
            documents = source.fetch_random_documents()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0], "Text file")

    def test_load_documents_filters_by_min_size(self):
        """Test filtering by minimum document size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files of different sizes
            (Path(tmpdir) / "small.txt").write_text("Short")
            (Path(tmpdir) / "large.txt").write_text("A" * 1000)

            source = LocalFileSource(path=tmpdir)
            documents = source.fetch_random_documents(min_doc_size=100)

            self.assertEqual(len(documents), 1)
            self.assertEqual(len(documents[0]), 1000)

    def test_load_documents_limits_num_docs(self):
        """Test limiting number of documents loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(10):
                (Path(tmpdir) / f"test{i}.txt").write_text(f"Document {i}")

            source = LocalFileSource(path=tmpdir)
            documents = source.fetch_random_documents(max_num_docs=5)

            self.assertEqual(len(documents), 5)


class TestCSVSource(unittest.TestCase):
    """Test cases for CSVSource."""

    def test_init_requires_csv_path(self):
        """Test that CSV path is required."""
        with self.assertRaises(ValueError):
            CSVSource(csv_path="")

    def test_init_validates_csv_exists(self):
        """Test that CSV file must exist."""
        with self.assertRaises(ValueError):
            CSVSource(csv_path="/nonexistent/file.csv")

    def test_load_documents_from_csv(self):
        """Test loading documents from CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=['text'])
            writer.writeheader()
            writer.writerow({'text': 'Document 1'})
            writer.writerow({'text': 'Document 2'})
            tmp_path = tmp.name

        try:
            source = CSVSource(csv_path=tmp_path)
            documents = source.fetch_random_documents()

            self.assertEqual(len(documents), 2)
            self.assertIn('Document 1', documents)
            self.assertIn('Document 2', documents)
        finally:
            Path(tmp_path).unlink()

    def test_load_documents_custom_column(self):
        """Test loading from custom column name."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=['content'])
            writer.writeheader()
            writer.writerow({'content': 'Document 1'})
            tmp_path = tmp.name

        try:
            source = CSVSource(csv_path=tmp_path, text_column='content')
            documents = source.fetch_random_documents()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0], 'Document 1')
        finally:
            Path(tmp_path).unlink()

    def test_load_documents_validates_column_exists(self):
        """Test that specified column must exist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=['other_column'])
            writer.writeheader()
            writer.writerow({'other_column': 'Data'})
            tmp_path = tmp.name

        try:
            source = CSVSource(csv_path=tmp_path, text_column='text')
            with self.assertRaises(ValueError):
                source.fetch_random_documents()
        finally:
            Path(tmp_path).unlink()

    def test_load_documents_filters_empty_rows(self):
        """Test that empty rows are filtered out."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=['text'])
            writer.writeheader()
            writer.writerow({'text': 'Document 1'})
            writer.writerow({'text': ''})
            writer.writerow({'text': 'Document 2'})
            tmp_path = tmp.name

        try:
            source = CSVSource(csv_path=tmp_path)
            documents = source.fetch_random_documents()

            self.assertEqual(len(documents), 2)
        finally:
            Path(tmp_path).unlink()

    def test_load_documents_reproducible_with_seed(self):
        """Test that using the same seed produces reproducible sampling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=['text'])
            writer.writeheader()
            for i in range(10):
                writer.writerow({'text': f'Document {i} with content'})
            tmp_path = tmp.name

        try:
            source = CSVSource(csv_path=tmp_path)

            # Same seed should produce same results
            docs1 = source.fetch_random_documents(max_num_docs=3, seed=42)
            docs2 = source.fetch_random_documents(max_num_docs=3, seed=42)
            self.assertEqual(docs1, docs2)

            # Different seed should produce different results
            docs3 = source.fetch_random_documents(max_num_docs=3, seed=123)
            self.assertNotEqual(docs1, docs3)
        finally:
            Path(tmp_path).unlink()


if __name__ == '__main__':
    unittest.main()
