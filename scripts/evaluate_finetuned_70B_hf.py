#!/usr/bin/env python3
"""
Evaluate Apertus-70B with LSAIE-TEAM LoRA adapter on MedQA test split.

This script evaluates the Apertus-70B model with the LoRA adapter from
LSAIE-TEAM/Apertus-70B-Instruct-LoRA.

Usage:
    python scripts/evaluate_finetuned_70B_hf.py
    python scripts/evaluate_finetuned_70B_hf.py --max_samples 50
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
BASE_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
ADAPTER_ID = "LSAIE-TEAM/Apertus-70B-Instruct-LoRA"
MODEL_NAME = "Apertus-70B + LoRA (HuggingFace)"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Apertus-70B with HuggingFace LoRA adapter on MedQA"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model ID (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_ID,
        help=f"LoRA adapter ID (default: {ADAPTER_ID})",
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
        default="results/apertus_70b_lora_hf_test.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_file = "logs/eval_apertus_70b_lora_hf_detailed.log"
    setup_logging(log_file=log_file)

    logger.info("=" * 70)
    logger.info("MedQA Test Evaluation - Apertus-70B + LoRA (HuggingFace)")
    logger.info("=" * 70)
    logger.info(f"Base Model:  {args.base_model}")
    logger.info(f"Adapter:     {args.adapter}")
    logger.info(f"Batch Size:  {args.batch_size}")
    logger.info(f"Max Tokens:  {args.max_new_tokens}")
    logger.info(f"Max Samples: {args.max_samples or 'All'}")
    logger.info(f"Output:      {args.output}")
    logger.info("=" * 70)

    # Setup HuggingFace authentication
    setup_hf_auth()

    # Import here to avoid slow startup
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # Load base model
    logger.info(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter from HuggingFace
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
            "base_model": args.base_model,
            "adapter": args.adapter,
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
