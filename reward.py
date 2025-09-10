"""
Reward function implementation for GSM8K mathematical reasoning.
"""

import re
from typing import Optional, List, Dict
import torch

# Simple reward configuration for mathematical reasoning (Interview-friendly)
REWARD_CONFIG = {
    # Positive rewards
    "CORRECT": 1.0,                 # Exact correct answer
    "CLOSE": 0.3,                   # Within 10% of correct answer
    
    # Negative rewards  
    "WRONG": -0.3,                  # Wrong numerical answer
    "NO_NUMBER": -0.5,              # No number found in response
    "PARSE_ERROR": -0.5,            # Failed to parse completion
    
    # Simple threshold
    "CLOSE_THRESHOLD": 0.1,         # 10% error tolerance
}


class GSM8KRewardFunction:
    """Calculates rewards for generated completions."""
    
    def __init__(self, answer_map: Dict[str, int]):
        self.answer_map = answer_map
        
    def calculate_rewards(
        self, 
        prompts: List[str], 
        completions: List[str],
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate rewards for completions based on mathematical correctness.
        
        Args:
            prompts: List of prompt strings (batch_size)
            completions: List of completion strings (batch_size * num_generations)
            
        Returns:
            Tensor of rewards (batch_size * num_generations)
        """
        rewards = []
        num_prompts = len(prompts)
        num_completions = len(completions)
        
        if num_prompts == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Calculate generations per prompt
        gens_per_prompt = num_completions // num_prompts if num_completions % num_prompts == 0 else 1
        
        for i, prompt in enumerate(prompts):
            ground_truth = self.answer_map.get(prompt)
            
            for j in range(gens_per_prompt):
                idx = i * gens_per_prompt + j
                if idx >= num_completions:
                    break
                    
                completion = completions[idx]
                reward = self._calculate_single_reward(completion, ground_truth)
                rewards.append(reward)
        
        # Handle any remaining completions
        while len(rewards) < num_completions:
            prompt_idx = len(rewards) % num_prompts
            ground_truth = self.answer_map.get(prompts[prompt_idx])
            reward = self._calculate_single_reward(completions[len(rewards)], ground_truth)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _calculate_single_reward(self, completion: str, ground_truth: Optional[int]) -> float:
        """Calculate reward for a single completion - simple and clean."""
        if ground_truth is None:
            return REWARD_CONFIG["PARSE_ERROR"]
        
        # Extract numbers from completion
        numbers = re.findall(r'-?\d+(?:\.\d+)?', str(completion))
        
        if not numbers:
            return REWARD_CONFIG["NO_NUMBER"]
        
        try:
            # Take the last number as the answer
            predicted = int(float(numbers[-1]))
            
            # Simple reward logic
            if predicted == ground_truth:
                return REWARD_CONFIG["CORRECT"]
            elif ground_truth != 0:
                # Check if close (within 10%)
                error_ratio = abs(predicted - ground_truth) / abs(ground_truth)
                if error_ratio < REWARD_CONFIG["CLOSE_THRESHOLD"]:
                    return REWARD_CONFIG["CLOSE"]
            
            return REWARD_CONFIG["WRONG"]
            
        except (ValueError, TypeError):
            return REWARD_CONFIG["PARSE_ERROR"]
    
    def get_reward_summary(self) -> Dict[str, float]:
        """Get a summary of all reward values for logging/debugging."""
        return {k: v for k, v in REWARD_CONFIG.items() if not k.endswith("_THRESHOLD")}
