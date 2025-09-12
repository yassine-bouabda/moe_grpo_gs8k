"""
Model management for loading and configuring models with LoRA and quantization.
"""

import torch
import logging
from typing import Dict, List
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import TrainingConfig

logger = logging.getLogger(__name__)


class ExpertUtilizationMonitor:
    """Monitors expert utilization in MoE models."""
    
    def __init__(self):
        self.expert_stats = {
            'router_logits': [],
            'expert_usage': [],
            'routing_entropy': [],
            'load_balance': []
        }
        self.hooks = []
    
    def register_hooks(self, model):
        """Register hooks to monitor expert utilization - fixed for OLMoE."""
        def router_hook(module, input, output):
            router_logits = None
            
            # Method 1: Check if output has router_logits attribute
            if hasattr(output, 'router_logits') and output.router_logits is not None:
                router_logits = output.router_logits
            
            # Method 2: OLMoE specific - router logits are second element in tuple
            elif isinstance(output, tuple) and len(output) >= 2:
                potential_router = output[1]
                # Verify this looks like router logits (should have num_experts dimension)
                if hasattr(potential_router, 'shape') and len(potential_router.shape) >= 2:
                    # Router logits typically have shape [..., num_experts]
                    if potential_router.shape[-1] == 64:  # OLMoE has 64 experts
                        router_logits = potential_router
            
            if router_logits is not None:
                try:
                    # Calculate expert usage and entropy
                    router_probs = F.softmax(router_logits, dim=-1)
                    
                    # Average over all dimensions except the expert dimension
                    if len(router_probs.shape) == 2:  # [seq_len, num_experts]
                        expert_usage = router_probs.mean(dim=0)  # Average over sequence
                    elif len(router_probs.shape) == 3:  # [batch, seq_len, num_experts]
                        expert_usage = router_probs.mean(dim=(0, 1))  # Average over batch and sequence
                    else:
                        expert_usage = router_probs.mean(dim=tuple(range(len(router_probs.shape)-1)))
                    
                    # Calculate routing entropy
                    entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean()
                    
                    # Store metrics
                    self.expert_stats['expert_usage'].append(expert_usage.detach().cpu())
                    self.expert_stats['routing_entropy'].append(entropy.detach().cpu())
                    
                except Exception as e:
                    logger.debug(f"Error processing router logits: {e}")
        
        # Find and hook MoE layers - be more selective to avoid hooking sub-modules
        hooked_count = 0
        for name, module in model.named_modules():
            # Hook only the main MLP layers, not their sub-components
            if (name.endswith('.mlp') and 
                'layers.' in name and 
                not any(x in name for x in ['experts.', 'gate.', 'proj'])):
                hook = module.register_forward_hook(router_hook)
                self.hooks.append(hook)
                hooked_count += 1
        
        logger.info(f"Registered {hooked_count} expert monitoring hooks on MoE layers")
    
    def get_expert_metrics(self) -> Dict:
        """Get comprehensive expert utilization metrics."""
        if not self.expert_stats['expert_usage']:
            return {}
        
        # Get recent statistics (last 5 steps or all if fewer)
        recent_usage = self.expert_stats['expert_usage'][-5:]
        recent_entropy = self.expert_stats['routing_entropy'][-5:]
        
        if recent_usage:
            # Stack usage tensors and compute statistics
            avg_usage = torch.stack(recent_usage).mean(dim=0)
            
            # Calculate comprehensive metrics
            max_usage = torch.max(avg_usage).item()
            min_usage = torch.min(avg_usage).item()
            usage_ratio = max_usage / (min_usage + 1e-8)  # Imbalance ratio
            
            return {
                'num_experts': len(avg_usage),
                'routing_entropy': torch.tensor(recent_entropy).mean().item(),
                'expert_usage_std': torch.std(avg_usage).item(),
                'expert_max_usage': max_usage,
                'expert_min_usage': min_usage,
                'expert_usage_ratio': usage_ratio,
                'total_routing_calls': len(self.expert_stats['expert_usage'])
            }
        return {}
    
    def cleanup(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ModelManager:
    """Manages model loading and configuration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.expert_monitor = ExpertUtilizationMonitor()
        
    def setup_model_and_tokenizer(self):
        """Load and configure model with LoRA and quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"  # Important for generation
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Standard attention modules
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Ensure generation config is properly set
        if hasattr(model, 'generation_config'):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Register expert monitoring hooks
        self.expert_monitor.register_hooks(model)
        logger.info("Expert utilization monitoring enabled")
        
        return model, tokenizer
