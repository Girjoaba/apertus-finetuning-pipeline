2#!/usr/bin/env python3
"""
Evaluate Apertus-8B with local LoRA adapter on MedQA test split.

This script evaluates the fine-tuned Apertus-8B model using the locally
trained LoRA weights from output/apertus_lora.

Usage:
    python scripts/evaluate_finetuned_8B.py
    python scripts/evaluate_finetuned_8B.py --max_samples 50
    python scripts/evaluate_finetuned_8B.py --adapter ./output/apertus_lora_8B_improved
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

# Configuration
BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_ADAPTER = "./output/apertus_lora"
MODEL_NAME = "Apertus-8B + LoRA (local)"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Apertus-8B with local LoRA adapter on MedQA"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help=f"Path to LoRA adapter (default: {DEFAULT_ADAPTER})",
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
        help="Maximum test samples (default: all 1273)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/apertus_8b_lora_test.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_file = "logs/eval_apertus_8b_lora_detailed.log"
    setup_logging(log_file=log_file)

    logger.info("=" * 70)
    logger.info("MedQA Test Evaluation - Apertus-8B + LoRA (local)")
    logger.info("=" * 70)
    logger.info(f"Base Model:  {BASE_MODEL}")
    logger.info(f"Adapter:     {args.adapter}")
    logger.info(f"Batch Size:  {args.batch_size}")
    logger.info(f"Max Tokens:  {args.max_new_tokens}")
    logger.info(f"Max Samples: {args.max_samples or 'All'}")
    logger.info(f"Output:      {args.output}")
    logger.info("=" * 70)

    # Verify adapter exists
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {args.adapter}")
        sys.exit(1)
    
    logger.info(f"Found adapter at: {adapter_path.absolute()}")

    # Setup HuggingFace authentication
    setup_hf_auth()

    # Import here to avoid slow startup
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load tokenizer
    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    # Load base model
    logger.info(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_dtype=torch.bfloat16,
    )
    logger.info("LoRA adapter loaded successfully")

    if hasattr(model, 'hf_device_map'):
        logger.info(f"Model device map: {model.hf_device_map}")

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
        model_name=MODEL_NAME,
        additional_info={
            "model_type": "finetuned_lora",
            "base_model": BASE_MODEL,
            "adapter": str(adapter_path.absolute()),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 70)

    print(f"\n{'='*50}")
    print(f"RESULTS: {MODEL_NAME}")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
