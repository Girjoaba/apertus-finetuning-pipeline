#!/usr/bin/env python3
"""
Evaluate the finetuned Apertus-70B with LoRA adapter on MedQA test split.
Loads adapter from HuggingFace: LSAIE-TEAM/apertus-medqa-lora-70B-tuned-parameters

Usage:
    python scripts/evaluate_finetuned_70B.py [--batch_size 4] [--max_samples N] [--output results/finetuned_70B_test.json]
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
ADAPTER_REPO = "LSAIE-TEAM/apertus-medqa-lora-70B-tuned-parameters"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned Apertus-70B (LoRA) on MedQA test split"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model path (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_REPO,
        help=f"LoRA adapter HuggingFace repo (default: {ADAPTER_REPO})",
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
        default="results/finetuned_70B_test.json",
        help="Output JSON file path (default: results/finetuned_70B_test.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MedQA Test Evaluation - Finetuned Apertus-70B (LoRA)")
    logger.info("=" * 60)

    # Setup HuggingFace authentication (required for private adapter repo)
    auth_success = setup_hf_auth()
    if not auth_success:
        logger.warning(
            "HuggingFace authentication may be required for private adapter repo. "
            "If loading fails, run `huggingface-cli login` first."
        )

    # Import here to avoid slow startup if just checking args
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # Load base model with automatic device mapping for multi-GPU
    logger.info(f"Loading base model from {args.base_model}...")
    logger.info("Using device_map='auto' for multi-GPU sharding")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Base model loaded. Device map: {base_model.hf_device_map if hasattr(base_model, 'hf_device_map') else 'N/A'}")

    # Load LoRA adapter from HuggingFace
    logger.info(f"Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        torch_dtype=torch.bfloat16,
    )

    logger.info("LoRA adapter loaded successfully")

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
        model_name=f"{args.base_model} + {args.adapter}",
        additional_info={
            "model_type": "finetuned_lora",
            "base_model": args.base_model,
            "adapter": args.adapter,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"LoRA Adapter: {args.adapter}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
