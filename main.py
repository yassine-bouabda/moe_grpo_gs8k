"""
Simple GRPO training pipeline for GSM8K mathematical reasoning.
Interview-ready implementation.
"""

import os
import logging
from config import TrainingConfig
from trainer import GRPOTrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simple main function for GRPO training."""
    # Setup
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize training pipeline
    pipeline = GRPOTrainingPipeline(config)
    
    try:
        # Train
        train_output = pipeline.train()
        
        # Log training results
        logger.info(f"Training completed!")
        logger.info(f"Final loss: {train_output.training_loss:.4f}")
        logger.info(f"Total steps: {train_output.global_step}")
        
        # Evaluate samples
        results = pipeline.evaluate_sample(num_samples=3)
        
        logger.info("\n" + "="*50)
        logger.info("Sample Evaluations:")
        for i, result in enumerate(results, 1):
            logger.info(f"\nSample {i}:")
            logger.info(f"Ground Truth: {result['ground_truth']}")
            logger.info(f"Reward: {result['reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up
        pipeline.cleanup()
        logger.info("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
