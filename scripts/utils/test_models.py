#!/usr/bin/env python3
"""
Testing script for evaluating all trained models and generating comparison plots.
Models: 8B LoRA, 8B Full, 70B LoRA
Uses wandb to fetch training metrics and lm_eval for evaluation.
"""

import argparse
import os
import subprocess
import json
from pathlib import Path

# Optional dependencies - check availability
MATPLOTLIB_AVAILABLE = False
PANDAS_AVAILABLE = False
WANDB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None


# Model configurations
MODELS = {
    "8b_lora": {
        "name": "8B LoRA",
        "base_model": "swiss-ai/Apertus-8B-Instruct-2509",
        "output_dir": "output/apertus_lora",
        "peft": True,
    },
    "8b_full": {
        "name": "8B Full",
        "base_model": "swiss-ai/Apertus-8B-Instruct-2509",
        "output_dir": "output/apertus_full",
        "peft": False,
    },
    "70b_lora": {
        "name": "70B LoRA", 
        "base_model": "swiss-ai/Apertus-70B-Instruct-2509",
        "output_dir": "output/apertus_70b_lora",
        "peft": True,
    },
}


def get_project_dir():
    """Get project root directory."""
    script_dir = Path(__file__).resolve().parent
    # Go up to project root
    return script_dir.parent.parent


def fetch_wandb_metrics(project_name: str, run_names: list = None) -> dict:
    """Fetch training metrics from wandb for specified runs."""
    if not WANDB_AVAILABLE:
        return {}
    
    if not PANDAS_AVAILABLE:
        print("Warning: pandas not available, skipping wandb metrics")
        return {}
    
    api = wandb.Api()
    metrics = {}
    
    try:
        runs = api.runs(project_name)
        for run in runs:
            if run_names and run.name not in run_names:
                continue
            
            history = run.history()
            metrics[run.name] = {
                "train_loss": history.get("train/loss", history.get("loss", pd.Series())).dropna().tolist(),
                "eval_loss": history.get("eval/loss", history.get("eval_loss", pd.Series())).dropna().tolist(),
                "eval_accuracy": history.get("eval_medqa_mcq_accuracy", history.get("eval/accuracy", pd.Series())).dropna().tolist(),
                "learning_rate": history.get("train/learning_rate", history.get("learning_rate", pd.Series())).dropna().tolist(),
                "steps": history.get("_step", pd.Series()).dropna().tolist(),
            }
    except Exception as e:
        print(f"Warning: Could not fetch wandb metrics: {e}")
    
    return metrics


def run_lm_eval(model_key: str, model_config: dict, project_dir: Path, device: str = "cuda:0") -> dict:
    """Run lm_eval harness for a model and return results."""
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    log_file = results_dir / f"log_{model_key}.txt"
    
    # Build model args
    model_args = f"pretrained={model_config['base_model']},trust_remote_code=True,dtype=bfloat16"
    
    if model_config["peft"]:
        peft_path = project_dir / model_config["output_dir"]
        if not peft_path.exists():
            print(f"Warning: {peft_path} does not exist, skipping {model_config['name']}")
            return {"accuracy": None, "error": "Model not found"}
        model_args += f",peft={peft_path}"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "medqa_4options",
        "--device", device,
        "--batch_size", "auto",
    ]
    
    print(f"Evaluating {model_config['name']}...")
    
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        
        # Parse accuracy from log
        with open(log_file, "r") as f:
            log_content = f.read()
        
        accuracy = None
        for line in log_content.split("\n"):
            if "|medqa_4options|" in line:
                parts = line.split("|")
                for part in parts:
                    try:
                        val = float(part.strip())
                        if 0 <= val <= 1:
                            accuracy = val
                            break
                    except ValueError:
                        continue
        
        return {"accuracy": accuracy, "log_file": str(log_file)}
    
    except subprocess.TimeoutExpired:
        return {"accuracy": None, "error": "Timeout"}
    except Exception as e:
        return {"accuracy": None, "error": str(e)}


def plot_training_loss(metrics: dict, output_path: Path):
    """Plot training loss comparison across models."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    for run_name, data in metrics.items():
        if data.get("train_loss"):
            steps = list(range(len(data["train_loss"])))
            plt.plot(steps, data["train_loss"], label=run_name, alpha=0.8)
    
    plt.xlabel("Steps")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_eval_loss(metrics: dict, output_path: Path):
    """Plot evaluation loss comparison across models."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    for run_name, data in metrics.items():
        if data.get("eval_loss"):
            steps = list(range(len(data["eval_loss"])))
            plt.plot(steps, data["eval_loss"], label=run_name, marker="o", alpha=0.8)
    
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Evaluation Loss")
    plt.title("Evaluation Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_eval_accuracy(metrics: dict, output_path: Path):
    """Plot evaluation accuracy (MedQA MCQ) comparison across models."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    for run_name, data in metrics.items():
        if data.get("eval_accuracy"):
            steps = list(range(len(data["eval_accuracy"])))
            plt.plot(steps, data["eval_accuracy"], label=run_name, marker="o", alpha=0.8)
    
    plt.xlabel("Evaluation Steps")
    plt.ylabel("MedQA MCQ Accuracy")
    plt.title("MedQA Accuracy During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_learning_rate(metrics: dict, output_path: Path):
    """Plot learning rate schedule comparison."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    for run_name, data in metrics.items():
        if data.get("learning_rate"):
            steps = list(range(len(data["learning_rate"])))
            plt.plot(steps, data["learning_rate"], label=run_name, alpha=0.8)
    
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_accuracy_bar(eval_results: dict, output_path: Path):
    """Plot bar chart of final evaluation accuracies."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    models = []
    accuracies = []
    
    for model_key, result in eval_results.items():
        if result.get("accuracy") is not None:
            models.append(MODELS[model_key]["name"])
            accuracies.append(result["accuracy"])
    
    if not models:
        print("No accuracies to plot")
        return
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=["#2ecc71", "#3498db", "#9b59b6"])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
    
    plt.xlabel("Model")
    plt.ylabel("MedQA Accuracy")
    plt.title("Final MedQA Evaluation Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_results_csv(eval_results: dict, output_path: Path):
    """Save evaluation results to CSV."""
    rows = []
    for model_key, result in eval_results.items():
        rows.append({
            "Model": MODELS[model_key]["name"],
            "Model_Key": model_key,
            "Accuracy": result.get("accuracy"),
            "Task": "MedQA",
            "Error": result.get("error", ""),
        })
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print("\nResults Summary:")
        print(df.to_string(index=False))
    else:
        # Fallback: write CSV manually
        with open(output_path, "w") as f:
            f.write("Model,Model_Key,Accuracy,Task,Error\n")
            for row in rows:
                f.write(f"{row['Model']},{row['Model_Key']},{row['Accuracy']},{row['Task']},{row['Error']}\n")
        print(f"Saved: {output_path}")
        print("\nResults Summary:")
        for row in rows:
            print(f"  {row['Model']}: {row['Accuracy']}")


def main():
    parser = argparse.ArgumentParser(description="Test and compare all trained models")
    parser.add_argument("--wandb-project", type=str, default="apertus-finetune",
                       help="Wandb project name")
    parser.add_argument("--run-names", type=str, nargs="+", default=None,
                       help="Specific wandb run names to fetch (default: all)")
    parser.add_argument("--models", type=str, nargs="+", 
                       choices=list(MODELS.keys()) + ["all"], default=["all"],
                       help="Models to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for evaluation")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip lm_eval and only generate plots from wandb")
    parser.add_argument("--plots-only", action="store_true",
                       help="Only generate plots from existing results")
    args = parser.parse_args()
    
    project_dir = get_project_dir()
    results_dir = project_dir / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Model Testing and Comparison Script")
    print("=" * 50)
    print(f"Project Directory: {project_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Plots Directory: {plots_dir}")
    print()
    
    # Determine which models to evaluate
    if "all" in args.models:
        models_to_eval = list(MODELS.keys())
    else:
        models_to_eval = args.models
    
    # Fetch wandb metrics for training plots
    print("Fetching training metrics from wandb...")
    wandb_metrics = fetch_wandb_metrics(args.wandb_project, args.run_names)
    
    if wandb_metrics:
        print(f"Found {len(wandb_metrics)} runs in wandb")
        
        # Generate training plots
        plot_training_loss(wandb_metrics, plots_dir / "training_loss.png")
        plot_eval_loss(wandb_metrics, plots_dir / "eval_loss.png")
        plot_eval_accuracy(wandb_metrics, plots_dir / "eval_accuracy.png")
        plot_learning_rate(wandb_metrics, plots_dir / "learning_rate.png")
    else:
        print("No wandb metrics found. Training plots will be skipped.")
    
    if args.plots_only:
        # Load existing results
        results_file = results_dir / "all_results.json"
        if results_file.exists():
            with open(results_file) as f:
                eval_results = json.load(f)
            plot_final_accuracy_bar(eval_results, plots_dir / "final_accuracy.png")
        else:
            print("No existing results found for plots-only mode")
        return
    
    # Run evaluations
    eval_results = {}
    
    if not args.skip_eval:
        print("\n" + "=" * 50)
        print("Running Model Evaluations")
        print("=" * 50)
        
        for model_key in models_to_eval:
            model_config = MODELS[model_key]
            print(f"\n--- {model_config['name']} ---")
            result = run_lm_eval(model_key, model_config, project_dir, args.device)
            eval_results[model_key] = result
            
            if result.get("accuracy"):
                print(f"Accuracy: {result['accuracy']:.4f}")
            elif result.get("error"):
                print(f"Error: {result['error']}")
        
        # Save results
        with open(results_dir / "all_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        save_results_csv(eval_results, results_dir / "all_results.csv")
        plot_final_accuracy_bar(eval_results, plots_dir / "final_accuracy.png")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
    print(f"Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
