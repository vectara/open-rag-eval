from typing import List
from pathlib import Path
import re
import uuid
import pandas as pd

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.data_classes.rag_results import RAGResult, RetrievalResult, GeneratedAnswerPart, AugmentedGenerationResult


class CSVConnector(Connector):
    def __init__(self, csv_path: str):
        """Initialize the CSV connector with path to CSV file."""
        self.csv_path = Path(csv_path)

    def fetch_data(self) -> List[RAGResult]:
        """Read the CSV file and convert to RAGResult objects."""
        # Read CSV file
        df = pd.read_csv(self.csv_path)

        # Add query_id if it doesn't exist
        if 'query_id' not in df.columns:
            df['query_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

        results = []
        # Group by query_id to process each query's results
        for _, group in df.groupby('query_id'):
            # Get the query (same for all rows in group)
            query = group['query'].iloc[0]

            # Create retrieved passages dictionary
            retrieved_passages = {
                str(row['passage_id']): str(row['passage'])
                for _, row in group.iterrows()
            }

            # Create retrieval result
            retrieval_result = RetrievalResult(
                query=query,
                retrieved_passages=retrieved_passages
            )

            # Get the generated answer and parse passage attributions
            # Take first non-empty generated answer
            generated_answer_raw = group['generated_answer'].dropna().iloc[0]

            # Parse generated answer to map passages to text segments
            generated_answer = self._parse_generated_answer(generated_answer_raw)

            # Create generation result
            generation_result = AugmentedGenerationResult(
                query=query,
                generated_answer=generated_answer
            )

            # Create final RAG result
            rag_result = RAGResult(
                retrieval_result=retrieval_result,
                generation_result=generation_result
            )

            results.append(rag_result)

        return results

    def _parse_generated_answer(self, text: str) -> List[GeneratedAnswerPart]:
        """Extracts text associated with numbered reference markers from a given string."""
        # First, expand multi-number citations like [1, 2, 3] to [1][2][3]
        def expand_multi_number_citation(match):
            numbers = re.findall(r'\d+', match.group())
            return ''.join([f"[{num}]" for num in numbers])

        # Find and replace [1, 2, 3] format
        multi_number_pattern = r'\[\d+(?:,\s*\d+)+\]'
        text = re.sub(multi_number_pattern, expand_multi_number_citation, text)

        # Next, handle [1], [2] format by removing commas between citations
        comma_separated_pattern = r'\]\s*,\s*\['
        text = re.sub(comma_separated_pattern, '][', text)

        # Now use the original pattern to find citation blocks
        citation_blocks = list(re.finditer(r'(?:\[\d+\])+', text))

        if not citation_blocks:
            return [GeneratedAnswerPart(text=text, citations=[])]

        # List to store results
        results = []

        # Process each segment between citation blocks
        for i, block in enumerate(citation_blocks):
            # Determine start and end of text segment
            if i == 0:
                text_start = 0
            else:
                text_start = citation_blocks[i-1].end()

            text_end = block.start()
            text_part = text[text_start:text_end].strip()

            # Extract individual citations from the block
            citations = re.findall(r'\[\d+\]', block.group())

            # Add the segment with its citations if it's not empty
            if text_part:
                results.append(GeneratedAnswerPart(text=text_part, citations=citations))

        # Process the last segment (after the last citation)
        last_segment = text[citation_blocks[-1].end():].strip()
        if len(last_segment) > 1:
            results.append(GeneratedAnswerPart(text=last_segment, citations=[]))

        return results
