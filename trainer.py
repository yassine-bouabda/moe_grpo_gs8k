"""
Training pipeline for GRPO (Group Relative Policy Optimization).
"""

import gc
import torch
import logging
from trl import GRPOConfig, GRPOTrainer
from transformers import set_seed, TrainerCallback

from config import TrainingConfig
from model_manager import ModelManager
from data_processor import GSM8KDataProcessor
from reward import GSM8KRewardFunction


import wandb
WANDB_AVAILABLE = True


logger = logging.getLogger(__name__)


class ExpertMonitoringCallback(TrainerCallback):
    """Callback to monitor expert utilization during training."""
    
    def __init__(self, expert_monitor):
        self.expert_monitor = expert_monitor
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log expert metrics during training."""
        if logs is not None:
            expert_metrics = self.expert_monitor.get_expert_metrics()
            if expert_metrics:
                # Add expert metrics to logs
                for key, value in expert_metrics.items():
                    logs[f"experts/{key}"] = value
                
                # Log directly to wandb to guarantee persistence
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log(
                {f"experts/{k}": float(v) for k, v in expert_metrics.items()},
                commit=True,  # force flush each call
            )
                
                # Console logging for key metrics
                entropy = expert_metrics.get('routing_entropy', 0)
                std = expert_metrics.get('expert_usage_std', 0)
                ratio = expert_metrics.get('expert_usage_ratio', 0)
                calls = expert_metrics.get('total_routing_calls', 0)
                
                logger.info(f"Expert Utilization - Entropy: {entropy:.3f}, Std: {std:.3f}, Ratio: {ratio:.2f}, Calls: {calls}")


class GradientDebugCallback(TrainerCallback):
    """Callback that prints true gradient norm to verify gradients flow."""
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            return
        total_norm = 0.0
        num_params = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.float().norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1
        if num_params > 0:
            total_norm = total_norm ** 0.5
            logger.info(f"[GradDebug] True grad-norm L2: {total_norm:.4e} over {num_params} tensors")
        else:
            logger.warning("[GradDebug] No gradients found on any trainable parameter!")


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

        # ------------------------------------------------------------------
        # Debug helper: print parameter statistics BEFORE any training begins
        # ------------------------------------------------------------------
        self._log_trainable_parameters()
        
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
            max_grad_norm=1.0,  # Allow larger gradients for LoRA layers
            
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
            peft_config=self.model_manager.peft_config,
        )
        
        # Add expert monitoring callback
        expert_callback = ExpertMonitoringCallback(self.model_manager.expert_monitor)
        trainer.add_callback(expert_callback)

        # Add gradient debug callback (prints true grad norm every step)
        trainer.add_callback(GradientDebugCallback())
        
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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _log_trainable_parameters(self):
        """Log how many parameters are trainable vs. frozen for debugging."""
        total_params = 0
        trainable_params = 0
        trainable_names = []
        for n, p in self.model.named_parameters():
            numel = p.numel()
            total_params += numel
            if p.requires_grad:
                trainable_params += numel
                if len(trainable_names) < 20:
                    trainable_names.append(n)

        perc_trainable = 100.0 * trainable_params / total_params if total_params else 0.0
        logger.info("================ PARAMETER DEBUG ================")
        logger.info(f"Total parameters: {total_params / 1e6:.2f} M")
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f} M ({perc_trainable:.2f}%)")
        logger.info("Sample trainable parameter names (first 20):")
        for name in trainable_names:
            logger.info(f"  • {name}")
        if trainable_params == 0:
            logger.warning("⚠️  No trainable parameters found! The model appears to be fully frozen.")
        logger.info("===============================================")
