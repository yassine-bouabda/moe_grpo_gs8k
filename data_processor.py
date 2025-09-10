"""
Data processing for GSM8K mathematical reasoning dataset.
"""

import re
import logging
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class GSM8KDataProcessor:
    """Handles loading and preprocessing of GSM8K dataset."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.answer_map: Dict[str, int] = {}
        
    def load_and_prepare_dataset(self, split: str = "train", max_samples: int = 1000) -> Dataset:
        """
        Load and prepare the GSM8K dataset for training.
        
        Args:
            split: Dataset split to load ("train" or "test")
            max_samples: Maximum number of samples to load
            
        Returns:
            Preprocessed dataset
        """
        logger.info(f"Loading GSM8K {split} dataset...")
        
        # Load dataset
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Limit samples for demo
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        
        # Process dataset
        processed_data = []
        for item in dataset:
            processed_item = self._process_single_item(item)
            if processed_item:
                processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} samples")
        
        return Dataset.from_list(processed_data)
    
    def _process_single_item(self, item: Dict) -> Dict:
        """
        Process a single GSM8K item.
        
        Args:
            item: Raw dataset item
            
        Returns:
            Processed item with prompt and answer_value
        """
        question = item["question"]
        answer = item["answer"]
        
        # Extract the numerical answer
        answer_value = self._extract_answer(answer)
        if answer_value is None:
            logger.warning(f"Could not extract answer from: {answer[:100]}...")
            return None
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Store answer mapping for reward calculation
        self.answer_map[prompt] = answer_value
        
        return {
            "prompt": prompt,
            "question": question,
            "answer": answer,
            "answer_value": answer_value
        }
    
    def _extract_answer(self, answer_text: str) -> int:
        """
        Extract numerical answer from answer text.
        
        Args:
            answer_text: The answer text containing the solution
            
        Returns:
            Extracted numerical answer or None if not found
        """
        # Look for the last number in the answer
        numbers = re.findall(r'-?\d+', answer_text)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass
        return None
