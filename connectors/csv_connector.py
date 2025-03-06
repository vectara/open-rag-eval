
import pandas as pd
from pathlib import Path
import re
import uuid
from data_classes.rag_results import RAGResult, RetrievalResult, AugmentedGenerationResult

from typing import Dict, List

class CSVConnector:
    def __init__(self, csv_path: str):
        """Initialize the CSV connector with path to CSV file."""
        self.csv_path = Path(csv_path)
        
    def read(self) -> List[RAGResult]:
        """Read the CSV file and convert to RAGResult objects."""
        # Read CSV file
        df = pd.read_csv(self.csv_path)
        
        # Add query_id if it doesn't exist
        if 'query_id' not in df.columns:
            df['query_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
            
        results = []
        # Group by query_id to process each query's results
        for query_id, group in df.groupby('query_id'):
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
    
    def _parse_generated_answer(self, text: str) -> Dict[str, str]:
        """Extracts text associated with numbered reference markers from a given string."""        
        # Dictionary to store results
        references = {}
        
        # Split text into segments
        segments = re.split(r'\[(\d+)\]', text)
        
        # Iterate through segments in pairs (text, marker)
        for i in range(0, len(segments)-1, 2):
            text_segment = segments[i].strip()
            marker = f'[{segments[i+1]}]'
            
            # Only add non-empty text segments
            if text_segment:
                references[marker] = text_segment
        
        return references