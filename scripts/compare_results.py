#!/usr/bin/env python3
"""
Compare evaluation results from base and finetuned models.

Usage:
    python scripts/compare_results.py [--base results/base_70B_test.json] [--finetuned results/finetuned_70B_test.json]
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compare_predictions(base_preds: list, finetuned_preds: list) -> dict:
    """
    Compare predictions between base and finetuned models.
    Returns analysis of where models differ.
    """
    base_correct = {p["index"]: p["correct"] for p in base_preds}
    finetuned_correct = {p["index"]: p["correct"] for p in finetuned_preds}

    both_correct = 0
    both_wrong = 0
    base_only_correct = 0
    finetuned_only_correct = 0

    for idx in base_correct:
        bc = base_correct.get(idx, False)
        fc = finetuned_correct.get(idx, False)

        if bc and fc:
            both_correct += 1
        elif not bc and not fc:
            both_wrong += 1
        elif bc and not fc:
            base_only_correct += 1
        else:
            finetuned_only_correct += 1

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_only_correct": base_only_correct,
        "finetuned_only_correct": finetuned_only_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare base vs finetuned model results")
    parser.add_argument(
        "--base",
        type=str,
        default="results/base_70B_test.json",
        help="Path to base model results JSON",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default="results/finetuned_70B_test.json",
        help="Path to finetuned model results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_70B_test.json",
        help="Output path for comparison results",
    )
    args = parser.parse_args()

    # Check if files exist
    if not Path(args.base).exists():
        logger.error(f"Base results file not found: {args.base}")
        return

    if not Path(args.finetuned).exists():
        logger.error(f"Finetuned results file not found: {args.finetuned}")
        return

    # Load results
    base_results = load_results(args.base)
    finetuned_results = load_results(args.finetuned)

    # Extract metrics
    base_acc = base_results["accuracy"]
    finetuned_acc = finetuned_results["accuracy"]
    delta = finetuned_acc - base_acc
    relative_improvement = (delta / base_acc * 100) if base_acc > 0 else 0

    # Compare predictions
    pred_comparison = compare_predictions(
        base_results.get("predictions", []),
        finetuned_results.get("predictions", []),
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("MEDQA TEST SPLIT EVALUATION COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<50} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 70)
    print(f"{'Base (Apertus-70B-Instruct-2509)':<50} {base_acc:>10.4f} {base_results['correct']:>10}/{base_results['total']}")
    print(f"{'Finetuned (LoRA)':<50} {finetuned_acc:>10.4f} {finetuned_results['correct']:>10}/{finetuned_results['total']}")
    print("-" * 70)
    print(f"{'Delta (Finetuned - Base)':<50} {delta:>+10.4f}")
    print(f"{'Relative Improvement':<50} {relative_improvement:>+10.2f}%")
    print("=" * 70)

    print("\nPREDICTION ANALYSIS")
    print("-" * 70)
    print(f"Both models correct:        {pred_comparison['both_correct']:>5}")
    print(f"Both models wrong:          {pred_comparison['both_wrong']:>5}")
    print(f"Only base correct:          {pred_comparison['base_only_correct']:>5}")
    print(f"Only finetuned correct:     {pred_comparison['finetuned_only_correct']:>5}")
    print("=" * 70 + "\n")

    # Save comparison results
    comparison_output = {
        "base": {
            "model": base_results.get("model", "base"),
            "accuracy": base_acc,
            "correct": base_results["correct"],
            "total": base_results["total"],
        },
        "finetuned": {
            "model": finetuned_results.get("model", "finetuned"),
            "accuracy": finetuned_acc,
            "correct": finetuned_results["correct"],
            "total": finetuned_results["total"],
        },
        "comparison": {
            "delta": delta,
            "relative_improvement_percent": relative_improvement,
            "prediction_analysis": pred_comparison,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comparison_output, f, indent=2)

    logger.info(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    main()
