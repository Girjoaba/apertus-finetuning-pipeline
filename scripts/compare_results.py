#!/usr/bin/env python3
"""
Compare evaluation results from multiple models.

Usage:
    # Compare two models:
    python scripts/compare_results.py --base results/base_70B_test.json --finetuned results/finetuned_70B_test.json

    # Compare all available results:
    python scripts/compare_results.py --all
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


def compare_predictions(preds1: list, preds2: list) -> dict:
    """
    Compare predictions between two models.
    Returns analysis of where models differ.
    """
    correct1 = {p["index"]: p["correct"] for p in preds1}
    correct2 = {p["index"]: p["correct"] for p in preds2}

    both_correct = 0
    both_wrong = 0
    first_only_correct = 0
    second_only_correct = 0

    for idx in correct1:
        c1 = correct1.get(idx, False)
        c2 = correct2.get(idx, False)

        if c1 and c2:
            both_correct += 1
        elif not c1 and not c2:
            both_wrong += 1
        elif c1 and not c2:
            first_only_correct += 1
        else:
            second_only_correct += 1

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "first_only_correct": first_only_correct,
        "second_only_correct": second_only_correct,
    }


def find_all_results(results_dir: str = "results") -> list[Path]:
    """Find all *_test.json result files."""
    results_path = Path(results_dir)
    return sorted(results_path.glob("*_test.json"))


def compare_all_models(results_dir: str = "results") -> None:
    """Load and compare all available result files."""
    result_files = find_all_results(results_dir)

    if not result_files:
        logger.error(f"No *_test.json files found in {results_dir}/")
        return

    # Load all results
    all_results = []
    for path in result_files:
        try:
            data = load_results(str(path))
            data["_file"] = path.name
            all_results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    if not all_results:
        logger.error("No valid result files found")
        return

    # Sort by accuracy (descending)
    all_results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

    # Print comparison table
    print("\n" + "=" * 80)
    print("MEDQA TEST SPLIT - ALL MODELS COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<45} {'Accuracy':>12} {'Correct':>12} {'Total':>8}")
    print("-" * 80)

    for result in all_results:
        model_name = result.get("model", result["_file"])
        # Truncate long model names
        if len(model_name) > 43:
            model_name = model_name[:40] + "..."
        acc = result.get("accuracy", 0)
        correct = result.get("correct", 0)
        total = result.get("total", 0)
        print(f"{model_name:<45} {acc:>12.4f} {correct:>12}/{total:<8}")

    print("=" * 80)

    # Save combined results
    output_path = Path(results_dir) / "all_models_comparison.json"
    combined = {
        "models": [
            {
                "model": r.get("model", r["_file"]),
                "file": r["_file"],
                "accuracy": r.get("accuracy", 0),
                "correct": r.get("correct", 0),
                "total": r.get("total", 0),
                "model_type": r.get("model_type", "unknown"),
            }
            for r in all_results
        ]
    }
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all *_test.json files in results/ directory",
    )
    args = parser.parse_args()

    # If --all flag, compare all available results
    if args.all:
        compare_all_models()
        return

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
