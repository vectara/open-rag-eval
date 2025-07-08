from typing import List
import re
import uuid
import logging
import pandas as pd

from open_rag_eval.utils.constants import NO_ANSWER, API_ERROR
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    GeneratedAnswerPart,
    AugmentedGenerationResult,
    MultiRAGResult
)

logger = logging.getLogger(__name__)


class RAGResultsLoader:

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> List[MultiRAGResult]:
        """Read the CSV file and organize RAGResult objects by query, including all runs."""
        # Read CSV file
        df = pd.read_csv(self.csv_path)

        # Add query_id if it doesn't exist
        if "query_id" not in df.columns:
            df["query_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Add query_run if it doesn't exist
        if "query_run" not in df.columns:
            df["query_run"] = 1

        # Create a dictionary to store RAGResults objects by query_id
        query_results_dict = {}

        # Process each query_id and query_run combination
        for (query_id, run_id), group in df.groupby(["query_id", "query_run"]):
            # Get the query (same for all rows in group)
            query = group["query"].iloc[0]

            # Create or get RAGResults object for this query
            if query_id not in query_results_dict:
                query_results_dict[query_id] = MultiRAGResult(query, query_id)

            # Create retrieved passages dictionary
            retrieved_passages = {
                str(row["passage_id"]): str(row["passage"])
                for _, row in group.iterrows()
            }

            # Create retrieval result
            retrieval_result = RetrievalResult(
                query=query, retrieved_passages=retrieved_passages)

            # Get the generated answer and parse passage attributions
            # Take first non-empty generated answer
            generated_answers = group["generated_answer"].dropna()

            if (generated_answers.empty or generated_answers.iloc[0] == NO_ANSWER or
                    generated_answers.iloc[0] == API_ERROR):
                logger.warning(
                    "Skipping query %s (run %s) with no generated answer/API error.",
                    query, run_id
                )
                continue

            # Parse generated answer to map passages to text segments
            generated_answer_raw = generated_answers.iloc[0]
            generated_answer = self._parse_generated_answer(generated_answer_raw)

            # Create generation result
            generation_result = AugmentedGenerationResult(
                query=query, generated_answer=generated_answer)

            # Create final RAG result
            rag_result = RAGResult(
                retrieval_result=retrieval_result,
                generation_result=generation_result
            )

            # Add to the appropriate RAGResults object
            query_results_dict[query_id].add_result(rag_result)

        # Return list of RAGResults objects that have at least one valid result
        return [qr for qr in query_results_dict.values() if qr.rag_results]

    def _parse_generated_answer(self, text: str) -> List[GeneratedAnswerPart]:
        """Extracts text associated with numbered reference markers from a given string."""

        # First, expand multi-number citations like [1, 2, 3] to [1][2][3]
        def expand_multi_number_citation(match):
            numbers = re.findall(r"\d+", match.group())
            return "".join([f"[{num}]" for num in numbers])

        # Find and replace [1, 2, 3] format
        multi_number_pattern = r"\[\d+(?:,\s*\d+)+\]"
        text = re.sub(multi_number_pattern, expand_multi_number_citation, text)

        # Next, handle [1], [2] format by removing commas between citations
        comma_separated_pattern = r"\]\s*,\s*\["
        text = re.sub(comma_separated_pattern, "][", text)

        # Now use the original pattern to find citation blocks
        citation_blocks = list(re.finditer(r"(?:\[\d+\])+", text))

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
                text_start = citation_blocks[i - 1].end()

            text_end = block.start()
            text_part = text[text_start:text_end].strip()

            # Extract individual citations from the block
            citations = re.findall(r"\[\d+\]", block.group())

            # Add the segment with its citations if it's not empty
            if text_part:
                results.append(
                    GeneratedAnswerPart(text=text_part, citations=citations))

        # Process the last segment (after the last citation)
        last_segment = text[citation_blocks[-1].end():].strip()
        if len(last_segment) > 1:
            results.append(GeneratedAnswerPart(text=last_segment, citations=[]))

        return results
