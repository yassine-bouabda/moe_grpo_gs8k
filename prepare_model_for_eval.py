"""
Prepare the trained model for evaluation by merging LoRA weights and saving properly.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_and_save_model(
    base_model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    lora_path: str = "./grpo_gsm8k_output/final_model",
    output_path: str = "./grpo_gsm8k_merged_model"
):
    """
    Merge LoRA weights with base model and save as a standalone model.
    
    Args:
        base_model_name: Original base model name
        lora_path: Path to saved LoRA adapter
        output_path: Where to save the merged model
    """
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    
    # Load the LoRA model
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
        logger.info("✅ LoRA adapter loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA adapter: {e}")
        logger.info("Trying to load as regular model...")
        model = AutoModelForCausalLM.from_pretrained(
            lora_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    logger.info("Merging LoRA weights with base model...")
    
    # Merge and unload LoRA weights
    if hasattr(model, 'merge_and_unload'):
        merged_model = model.merge_and_unload()
        logger.info("✅ LoRA weights merged successfully")
    else:
        merged_model = model
        logger.info("No LoRA weights to merge, using model as-is")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Saving merged model to: {output_path}")
    
    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        save_function=torch.save,
        safe_serialization=True
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    logger.info("✅ Model saved successfully!")
    logger.info(f"You can now evaluate with: python evaluate_gsm8k.py {output_path}")
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA weights and prepare model for evaluation")
    parser.add_argument("--lora_path", default="./grpo_gsm8k_output/final_model", help="Path to LoRA adapter")
    parser.add_argument("--base_model", default="allenai/OLMoE-1B-7B-0125-Instruct", help="Base model name")
    parser.add_argument("--output_path", default="./grpo_gsm8k_merged_model", help="Output path for merged model")
    
    args = parser.parse_args()
    
    try:
        merged_path = merge_and_save_model(
            base_model_name=args.base_model,
            lora_path=args.lora_path,
            output_path=args.output_path
        )
        
        print("\n" + "="*60)
        print("SUCCESS! Model prepared for evaluation.")
        print("="*60)
        print(f"Merged model saved to: {merged_path}")
        print(f"Now run: python evaluate_gsm8k.py {merged_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare model: {e}")
        raise

if __name__ == "__main__":
    main()
