"""
Training pipeline for GRPO (Group Relative Policy Optimization).
"""

import gc
import torch
import logging
from trl import GRPOConfig, GRPOTrainer
from transformers import set_seed

from config import TrainingConfig
from model_manager import ModelManager
from data_processor import GSM8KDataProcessor
from reward import GSM8KRewardFunction

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GRPOTrainingPipeline:
    """Main training pipeline for GRPO."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        set_seed(config.seed)
        
        # Initialize wandb if requested and available
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif config.use_wandb and not WANDB_AVAILABLE:
            logger.warning("wandb requested but not installed. Install with: pip install wandb")
            self.config.use_wandb = False
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.model, self.tokenizer = self.model_manager.setup_model_and_tokenizer()
        
        self.data_processor = GSM8KDataProcessor(self.tokenizer)
        self.train_dataset = self.data_processor.load_and_prepare_dataset()
        
        self.reward_function = GSM8KRewardFunction(self.data_processor.answer_map)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb_config = {
            # Model config
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            
            # Training hyperparameters
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_generations": self.config.num_generations,
            "max_steps": self.config.max_steps,
            "num_train_epochs": self.config.num_train_epochs,
            
            # Generation parameters
            "max_prompt_length": self.config.max_prompt_length,
            "max_completion_length": self.config.max_completion_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            
            # System
            "seed": self.config.seed,
            "use_bf16": self.config.use_bf16,
        }
        
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            entity=self.config.wandb_entity,
            config=wandb_config,
            tags=["grpo", "gsm8k", "math-reasoning"]
        )
        
        logger.info(f"Initialized wandb tracking: {wandb.run.url}")
        
    def create_grpo_config(self):
        """Create GRPO training configuration."""
        return GRPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            
            # GRPO specific
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            
            # Simple optimization settings
            max_grad_norm=0.5,  # Prevent gradient explosions
            
            # Logging
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to=["tensorboard", "wandb"] if self.config.use_wandb else ["tensorboard"],
            
            # Memory efficiency
            gradient_checkpointing=True,
            bf16=self.config.use_bf16,
        )
    
    def train(self):
        """Run the training loop."""
        logger.info("Starting GRPO training...")
        
        # Create trainer
        grpo_config = self.create_grpo_config()
        
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function.calculate_rewards],
            args=grpo_config,
            train_dataset=self.train_dataset,
        )
        
        # Train
        train_output = trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}/final_model")
        trainer.save_model(f"{self.config.output_dir}/final_model")
        
        return train_output
    
    def evaluate_sample(self, num_samples: int = 5):
        """Generate and evaluate sample outputs."""
        logger.info("Generating sample outputs...")
        
        self.model.eval()
        samples = self.train_dataset.select(range(min(num_samples, len(self.train_dataset))))
        
        results = []
        with torch.no_grad():
            for sample in samples:
                # Tokenize prompt
                inputs = self.tokenizer(
                    sample['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.model.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(sample['prompt']):]
                
                # Calculate reward
                reward = self.reward_function._calculate_single_reward(
                    completion, 
                    sample.get('answer_value')
                )
                
                results.append({
                    'prompt': sample['prompt'][:100] + "...",
                    'completion': completion[:200] + "...",
                    'ground_truth': sample.get('answer_value'),
                    'reward': reward
                })
                
                logger.info(f"Sample {len(results)}: Reward = {reward:.2f}")
        
        # Log sample results to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            avg_reward = sum(r['reward'] for r in results) / len(results)
            log_data = {
                "eval/avg_reward": avg_reward,
                "eval/num_samples": len(results)
            }
            
            # Add expert utilization metrics
            expert_metrics = self.model_manager.expert_monitor.get_expert_metrics()
            if expert_metrics:
                for key, value in expert_metrics.items():
                    if key != 'expert_usage_mean':  # Skip the list
                        log_data[f"experts/{key}"] = value
            
            wandb.log(log_data)
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        # Finish wandb run if active
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")
        
        # Clean up expert monitoring hooks
        self.model_manager.expert_monitor.cleanup()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
