#!/usr/bin/env python3
"""
Evaluate the base Apertus-8B-Instruct model on MedQA test split.

Usage:
    python scripts/evaluate_base_8B.py [--batch_size 8] [--max_samples N] [--output results/base_8B_test.json]
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
    setup_logging,
    load_medqa_test_split,
    run_mcq_evaluation,
    save_results,
)

logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base Apertus-8B on MedQA test split"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)",
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
        default="results/base_8B_test.json",
        help="Output JSON file path (default: results/base_8B_test.json)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/eval_base_8B_detailed.log",
        help="Detailed log file path (default: logs/eval_base_8B_detailed.log)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging with file output
    setup_logging(log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("MedQA Test Evaluation - Base Apertus-8B")
    logger.info("=" * 60)
    logger.info(f"Log file: {args.log_file}")

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

    # Load model - 8B fits on a single GPU
    logger.info(f"Loading base model from {BASE_MODEL}...")
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
