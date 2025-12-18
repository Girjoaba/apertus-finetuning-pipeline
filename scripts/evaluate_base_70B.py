#!/usr/bin/env python3
"""
Evaluate the base Apertus-70B-Instruct model on MedQA test split.

Usage:
    python scripts/evaluate_base_70B.py [--batch_size 4] [--max_samples N] [--output results/base_70B_test.json]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.medqa_eval_utils import (
    setup_hf_auth,
    load_medqa_test_split,
    run_mcq_evaluation,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base Apertus-70B on MedQA test split"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/base_70B_test.json",
        help="Output JSON file path (default: results/base_70B_test.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MedQA Test Evaluation - Base Apertus-70B")
    logger.info("=" * 60)

    # Setup HuggingFace authentication
    setup_hf_auth()

    # Import here to avoid slow startup if just checking args
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    # Load model with automatic device mapping for multi-GPU
    logger.info(f"Loading base model from {BASE_MODEL}...")
    logger.info("Using device_map='auto' for multi-GPU sharding")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    # Load test data
    test_data = load_medqa_test_split(tokenizer, max_samples=args.max_samples)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = run_mcq_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    save_results(
        results=results,
        output_path=args.output,
        model_name=BASE_MODEL,
        additional_info={
            "model_type": "base",
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Model: {BASE_MODEL}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
