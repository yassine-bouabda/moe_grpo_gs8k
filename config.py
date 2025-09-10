"""
Configuration management for GRPO training.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Simple configuration for GRPO training pipeline."""
    
    # Core settings - interview focus
    model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct"
    learning_rate: float = 2e-5
    batch_size: int = 6
    max_steps: int = 100
    
    # GRPO specific  
    num_generations: int = 3  # Key parameter for GRPO
    
    # Generation settings
    max_completion_length: int = 128
    temperature: float = 0.7
    
    # Logging
    output_dir: str = "./grpo_gsm8k_output"
    use_wandb: bool = True
    
    # Defaults (less important for interview)
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 1
    max_prompt_length: int = 256
    top_p: float = 0.9
    logging_steps: int = 5
    save_steps: int = 50
    wandb_project: str = "grpo-gsm8k"
    wandb_run_name: str = None
    wandb_entity: str = None
    seed: int = 42
    use_bf16: bool = False
    
    # LoRA defaults
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
