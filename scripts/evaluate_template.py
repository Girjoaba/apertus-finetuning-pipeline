#!/usr/bin/env python3
"""
=============================================================================
MedQA Evaluation Template - Evaluate Your Own Model
=============================================================================

This is a template script to evaluate any HuggingFace model on the MedQA test split.
Copy this file and modify the configuration section below for your model.

Usage:
    python scripts/evaluate_template.py [options]

Examples:
    # Evaluate a base model
    python scripts/evaluate_template.py --base_model meta-llama/Llama-3-8B-Instruct

    # Evaluate a base model with LoRA adapter
    python scripts/evaluate_template.py --base_model swiss-ai/Apertus-8B-Instruct-2509 \
                                        --adapter your-username/your-lora-adapter

    # Quick test with limited samples
    python scripts/evaluate_template.py --base_model your-model --max_samples 50

    # Custom output path
    python scripts/evaluate_template.py --base_model your-model --output results/my_model_test.json

Requirements:
    - HuggingFace access token (for private models): huggingface-cli login
    - GPU with sufficient VRAM (8B models: ~16GB, 70B models: 4x80GB)
    - Python packages: transformers, peft, torch, datasets
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


# =============================================================================
# CONFIGURATION - Modify these defaults for your model
# =============================================================================

# Default base model (HuggingFace model ID or local path)
DEFAULT_BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"

# Default LoRA adapter (set to None for base model only, or HuggingFace ID/local path)
DEFAULT_ADAPTER = None

# Model display name for results (optional, defaults to model ID)
DEFAULT_MODEL_NAME = None

# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace model on MedQA test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --base_model meta-llama/Llama-3-8B-Instruct
  %(prog)s --base_model swiss-ai/Apertus-8B-Instruct-2509 --adapter my-org/my-lora
  %(prog)s --base_model ./local/model/path --max_samples 100
        """,
    )
    
    # Model configuration
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model HuggingFace ID or local path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="LoRA adapter HuggingFace ID or local path (default: None = base model only)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Display name for results (default: auto-generated from model/adapter)",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8, reduce if OOM)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum test samples to evaluate (default: all 1273)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per question (default: 128)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Detailed log file path (default: auto-generated in logs/)",
    )
    
    # Advanced options
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code in model (default: True)",
    )
    
    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def generate_output_paths(args):
    """Generate default output and log paths based on model name."""
    # Create a safe filename from model name
    if args.adapter:
        model_id = args.adapter.split("/")[-1]
    else:
        model_id = args.base_model.split("/")[-1]
    
    safe_name = model_id.replace("/", "_").replace(".", "_").replace("-", "_").lower()
    
    if args.output is None:
        args.output = f"results/{safe_name}_test.json"
    
    if args.log_file is None:
        args.log_file = f"logs/eval_{safe_name}_detailed.log"
    
    return args


def get_model_display_name(args):
    """Generate a display name for the model."""
    if args.model_name:
        return args.model_name
    
    if args.adapter:
        return f"{args.base_model} + {args.adapter}"
    else:
        return args.base_model


def main():
    args = parse_args()
    args = generate_output_paths(args)
    
    # Setup logging with file output
    setup_logging(log_file=args.log_file)
    
    model_display_name = get_model_display_name(args)
    
    logger.info("=" * 70)
    logger.info("MedQA Test Evaluation")
    logger.info("=" * 70)
    logger.info(f"Base Model:    {args.base_model}")
    logger.info(f"LoRA Adapter:  {args.adapter or 'None (base model only)'}")
    logger.info(f"Display Name:  {model_display_name}")
    logger.info(f"Batch Size:    {args.batch_size}")
    logger.info(f"Max Tokens:    {args.max_new_tokens}")
    logger.info(f"Max Samples:   {args.max_samples or 'All'}")
    logger.info(f"Output:        {args.output}")
    logger.info(f"Log File:      {args.log_file}")
    logger.info("=" * 70)

    # Setup HuggingFace authentication (for private models)
    setup_hf_auth()

    # Import here to avoid slow startup
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    # Load base model
    logger.info(f"Loading model from {args.base_model}...")
    logger.info(f"Using dtype={args.dtype}, device_map={args.device_map}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    # Load LoRA adapter if specified
    if args.adapter:
        from peft import PeftModel
        
        logger.info(f"Loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            torch_dtype=get_dtype(args.dtype),
        )
        logger.info("LoRA adapter loaded successfully")
        model_type = "finetuned_lora"
    else:
        model_type = "base"

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
        model_name=model_display_name,
        additional_info={
            "model_type": model_type,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
        },
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Model: {model_display_name}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Detailed log: {args.log_file}")
    logger.info("=" * 70)
    
    # Print quick summary to stdout
    print(f"\n{'='*50}")
    print(f"RESULTS: {model_display_name}")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
