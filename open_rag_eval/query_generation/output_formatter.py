"""Output formatter for generated queries."""

import csv
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_generator import QueryWithAnswer

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Format and save generated queries to files.

    Supports CSV and JSONL output formats with optional metadata.
    """

    @staticmethod
    def save_to_csv(
        queries: List[str],
        output_path: str,
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save queries to CSV file.

        Args:
            queries: List of query strings
            output_path: Path to output CSV file
            include_metadata: Whether to include metadata columns
            metadata: Optional list of metadata dicts (one per query)

        Raises:
            ValueError: If queries list is empty or metadata length mismatch
            IOError: If file cannot be written
        """
        if not queries:
            raise ValueError("No queries to save")

        if metadata and len(metadata) != len(queries):
            raise ValueError(
                f"Metadata length ({len(metadata)}) does not match "
                f"queries length ({len(queries)})"
            )

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if include_metadata and metadata:
                    # Get all unique metadata keys
                    all_keys = set()
                    for meta in metadata:
                        all_keys.update(meta.keys())
                    fieldnames = ['query_id', 'query'] + sorted(all_keys)

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for query, meta in zip(queries, metadata):
                        row = {
                            'query_id': str(uuid.uuid4()),
                            'query': query,
                            **meta
                        }
                        writer.writerow(row)
                else:
                    # Simple format: just query_id and query
                    fieldnames = ['query_id', 'query']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for query in queries:
                        writer.writerow({
                            'query_id': str(uuid.uuid4()),
                            'query': query
                        })

            logger.info("Saved %d queries to %s", len(queries), output_path)

        except IOError as e:
            raise IOError(f"Failed to write CSV file {output_path}: {str(e)}") from e

    @staticmethod
    def save_to_jsonl(
        queries: List[str],
        output_path: str,
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save queries to JSONL file (one JSON object per line).

        Args:
            queries: List of query strings
            output_path: Path to output JSONL file
            include_metadata: Whether to include metadata fields
            metadata: Optional list of metadata dicts (one per query)

        Raises:
            ValueError: If queries list is empty or metadata length mismatch
            IOError: If file cannot be written
        """
        if not queries:
            raise ValueError("No queries to save")

        if metadata and len(metadata) != len(queries):
            raise ValueError(
                f"Metadata length ({len(metadata)}) does not match "
                f"queries length ({len(queries)})"
            )

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as jsonlfile:
                for idx, query in enumerate(queries):
                    record = {
                        'query_id': str(uuid.uuid4()),
                        'query': query
                    }

                    if include_metadata and metadata:
                        record.update(metadata[idx])

                    jsonlfile.write(json.dumps(record) + '\n')

            logger.info("Saved %d queries to %s", len(queries), output_path)

        except IOError as e:
            raise IOError(f"Failed to write JSONL file {output_path}: {str(e)}") from e

    @staticmethod
    def save_queries(
        queries: List[str],
        output_path: str,
        output_format: str = 'csv',
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save queries to file in specified format.

        Args:
            queries: List of query strings
            output_path: Path to output file
            output_format: Output format ('csv' or 'jsonl')
            include_metadata: Whether to include metadata
            metadata: Optional list of metadata dicts

        Raises:
            ValueError: If output_format is invalid
        """
        if output_format == 'csv':
            OutputFormatter.save_to_csv(
                queries,
                output_path,
                include_metadata,
                metadata
            )
        elif output_format in ['jsonl', 'json']:
            OutputFormatter.save_to_jsonl(
                queries,
                output_path,
                include_metadata,
                metadata
            )
        else:
            raise ValueError(
                f"Invalid output format: {output_format}. "
                "Must be 'csv' or 'jsonl'"
            )

    @staticmethod
    def save_qa_pairs_to_csv(
        qa_pairs: List["QueryWithAnswer"],
        output_path: str,
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save query-answer pairs to CSV file with expected_answer column.

        Args:
            qa_pairs: List of QueryWithAnswer objects
            output_path: Path to output CSV file
            include_metadata: Whether to include metadata columns
            metadata: Optional list of metadata dicts (one per QA pair)

        Raises:
            ValueError: If qa_pairs list is empty or metadata length mismatch
            IOError: If file cannot be written
        """
        if not qa_pairs:
            raise ValueError("No QA pairs to save")

        if metadata and len(metadata) != len(qa_pairs):
            raise ValueError(
                f"Metadata length ({len(metadata)}) does not match "
                f"QA pairs length ({len(qa_pairs)})"
            )

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if include_metadata and metadata:
                    # Get all unique metadata keys
                    all_keys = set()
                    for meta in metadata:
                        all_keys.update(meta.keys())
                    fieldnames = ['query_id', 'query', 'expected_answer'] + sorted(all_keys)

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for qa_pair, meta in zip(qa_pairs, metadata):
                        row = {
                            'query_id': str(uuid.uuid4()),
                            'query': qa_pair.query,
                            'expected_answer': qa_pair.expected_answer,
                            **meta
                        }
                        writer.writerow(row)
                else:
                    # Standard format: query_id, query, expected_answer
                    fieldnames = ['query_id', 'query', 'expected_answer']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for qa_pair in qa_pairs:
                        writer.writerow({
                            'query_id': str(uuid.uuid4()),
                            'query': qa_pair.query,
                            'expected_answer': qa_pair.expected_answer
                        })

            logger.info("Saved %d QA pairs to %s", len(qa_pairs), output_path)

        except IOError as e:
            raise IOError(f"Failed to write CSV file {output_path}: {str(e)}") from e

    @staticmethod
    def save_qa_pairs_to_jsonl(
        qa_pairs: List["QueryWithAnswer"],
        output_path: str,
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save query-answer pairs to JSONL file with expected_answer field.

        Args:
            qa_pairs: List of QueryWithAnswer objects
            output_path: Path to output JSONL file
            include_metadata: Whether to include metadata fields
            metadata: Optional list of metadata dicts (one per QA pair)

        Raises:
            ValueError: If qa_pairs list is empty or metadata length mismatch
            IOError: If file cannot be written
        """
        if not qa_pairs:
            raise ValueError("No QA pairs to save")

        if metadata and len(metadata) != len(qa_pairs):
            raise ValueError(
                f"Metadata length ({len(metadata)}) does not match "
                f"QA pairs length ({len(qa_pairs)})"
            )

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as jsonlfile:
                for idx, qa_pair in enumerate(qa_pairs):
                    record = {
                        'query_id': str(uuid.uuid4()),
                        'query': qa_pair.query,
                        'expected_answer': qa_pair.expected_answer
                    }

                    if include_metadata and metadata:
                        record.update(metadata[idx])

                    jsonlfile.write(json.dumps(record) + '\n')

            logger.info("Saved %d QA pairs to %s", len(qa_pairs), output_path)

        except IOError as e:
            raise IOError(f"Failed to write JSONL file {output_path}: {str(e)}") from e

    @staticmethod
    def save_qa_pairs(
        qa_pairs: List["QueryWithAnswer"],
        output_path: str,
        output_format: str = 'csv',
        include_metadata: bool = False,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Save query-answer pairs to file in specified format.

        Args:
            qa_pairs: List of QueryWithAnswer objects
            output_path: Path to output file
            output_format: Output format ('csv' or 'jsonl')
            include_metadata: Whether to include metadata
            metadata: Optional list of metadata dicts

        Raises:
            ValueError: If output_format is invalid
        """
        if output_format == 'csv':
            OutputFormatter.save_qa_pairs_to_csv(
                qa_pairs,
                output_path,
                include_metadata,
                metadata
            )
        elif output_format in ['jsonl', 'json']:
            OutputFormatter.save_qa_pairs_to_jsonl(
                qa_pairs,
                output_path,
                include_metadata,
                metadata
            )
        else:
            raise ValueError(
                f"Invalid output format: {output_format}. "
                "Must be 'csv' or 'jsonl'"
            )
