"""
Shared utilities for MedQA test evaluation.
Provides functions for loading data, running MCQ evaluation, and saving results.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Setup logging configuration with optional file output.
    
    Args:
        log_file: Optional path to log file. If None, only console logging.
        level: Logging level (default: INFO)
    """
    handlers = [
        logging.StreamHandler(),  # Console output
    ]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def log_question_answer(
    index: int,
    question: str,
    options: str,
    generated_text: str,
    predicted: Optional[str],
    gold: str,
    correct: bool,
) -> None:
    """
    Log a single question-answer pair with full details.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "✓ CORRECT" if correct else "✗ WRONG"
    
    logger.info("=" * 80)
    logger.info(f"[{timestamp}] Question {index + 1}")
    logger.info("-" * 80)
    logger.info(f"QUESTION: {question[:500]}{'...' if len(question) > 500 else ''}")
    logger.info(f"OPTIONS:\n{options}")
    logger.info("-" * 40)
    logger.info(f"MODEL RESPONSE: {generated_text}")
    logger.info(f"EXTRACTED ANSWER: {predicted}")
    logger.info(f"GOLD ANSWER: {gold}")
    logger.info(f"RESULT: {status}")
    logger.info("=" * 80)


def setup_hf_auth() -> bool:
    """
    Setup HuggingFace authentication by reading token from cache.
    Returns True if authentication was successful, False otherwise.
    """
    from huggingface_hub import login

    # Check for token in environment variable first
    hf_token = os.environ.get("HF_TOKEN")
    
    if hf_token:
        login(token=hf_token)
        logger.info("Authenticated with HuggingFace using HF_TOKEN environment variable")
        return True

    # Try to read from cache file (created by `huggingface-cli login`)
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]

    for token_path in token_paths:
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                login(token=token)
                logger.info(f"Authenticated with HuggingFace using token from {token_path}")
                return True

    logger.warning(
        "No HuggingFace token found. Run `huggingface-cli login` or set HF_TOKEN env var. "
        "Private repos will not be accessible."
    )
    return False


def format_medqa_prompt(example: dict) -> str:
    """
    Format a MedQA example into the prompt format used for evaluation.
    Returns the user content string (without chat template applied).
    """
    question = example.get("sent1", "").strip()
    if example.get("sent2"):
        question += " " + example["sent2"]

    option_keys = ["ending0", "ending1", "ending2", "ending3"]
    labels = ["A", "B", "C", "D"]
    options_text_lines = []

    for i, key in enumerate(option_keys):
        opt_text = example.get(key, "N/A")
        options_text_lines.append(f"{labels[i]}: {opt_text}")

    options_block = "\n".join(options_text_lines)

    user_content = (
        f"Answer the following multiple choice question about medical knowledge.\n\n"
        f"{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Answer:"
    )

    return user_content


def build_chat_prompt(user_content: str, tokenizer) -> str:
    """
    Apply chat template to user content for generation.
    """
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_choice(text: str) -> Optional[str]:
    """
    Extract A, B, C, or D from generated text.
    Uses multiple patterns to robustly extract the answer.
    """
    # Look for explicit "Answer: X" pattern
    m = re.search(r"Answer:\s*([ABCD])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Look for "X:" or "X)" at the start of a line or sentence
    start_pattern = re.search(r"\b([ABCD])[\:\)\.]", text)
    if start_pattern:
        return start_pattern.group(1).upper()

    # Look for standalone letters with word boundaries
    standalone = re.search(r"\b([ABCD])\b", text)
    if standalone:
        return standalone.group(1).upper()

    return None


def load_medqa_test_split(tokenizer, max_samples: Optional[int] = None) -> list[dict]:
    """
    Load and format the MedQA test split.
    
    Args:
        tokenizer: HuggingFace tokenizer for applying chat template
        max_samples: Optional limit on number of samples to load
        
    Returns:
        List of dicts with 'prompt', 'gold_label', 'gold_idx', 'question', and 'options' keys
    """
    logger.info("Loading MedQA test split...")
    dataset = load_dataset(
        "openlifescienceai/MedQA-USMLE-4-options-hf",
        split="test",
    )

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples")

    labels = ["A", "B", "C", "D"]
    formatted_data = []

    for example in tqdm(dataset, desc="Formatting prompts"):
        user_content = format_medqa_prompt(example)
        prompt = build_chat_prompt(user_content, tokenizer)
        gold_idx = example["label"]
        gold_label = labels[gold_idx]

        # Build question text for logging
        question_text = example.get("sent1", "").strip()
        if example.get("sent2"):
            question_text += " " + example["sent2"]

        # Build options text for logging
        option_keys = ["ending0", "ending1", "ending2", "ending3"]
        options_lines = []
        for i, key in enumerate(option_keys):
            options_lines.append(f"{labels[i]}: {example.get(key, 'N/A')}")
        options_text = "\n".join(options_lines)

        formatted_data.append({
            "prompt": prompt,
            "gold_label": gold_label,
            "gold_idx": gold_idx,
            "question": question_text,
            "options": options_text,
        })

    logger.info(f"Loaded {len(formatted_data)} test samples")
    return formatted_data


def run_mcq_evaluation(
    model,
    tokenizer,
    test_data: list[dict],
    batch_size: int = 4,
    max_new_tokens: int = 16,
) -> dict:
    """
    Run MCQ evaluation on the test data.
    
    Args:
        model: HuggingFace model for generation
        tokenizer: HuggingFace tokenizer
        test_data: List of formatted test examples from load_medqa_test_split()
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dict with 'accuracy', 'correct', 'total', and 'predictions' keys
    """
    model.eval()
    device = next(model.parameters()).device

    # Set padding side to left for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions = []
    correct = 0
    total = 0

    prompts = [d["prompt"] for d in test_data]
    gold_labels = [d["gold_label"] for d in test_data]

    logger.info(f"Running evaluation on {len(prompts)} samples with batch_size={batch_size}")

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_prompts = prompts[i : i + batch_size]
        batch_gold = gold_labels[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, (pred_text, gold_letter) in enumerate(zip(decoded_preds, batch_gold)):
            pred_letter = extract_choice(pred_text)
            is_correct = pred_letter == gold_letter

            if is_correct:
                correct += 1
            total += 1

            idx = i + j
            predictions.append({
                "index": idx,
                "predicted": pred_letter,
                "gold": gold_letter,
                "correct": is_correct,
                "generated_text": pred_text.strip(),
            })

            # Log question and answer
            log_question_answer(
                index=idx,
                question=test_data[idx].get("question", ""),
                options=test_data[idx].get("options", ""),
                generated_text=pred_text.strip(),
                predicted=pred_letter,
                gold=gold_letter,
                correct=is_correct,
            )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def save_results(
    results: dict,
    output_path: str,
    model_name: str,
    additional_info: Optional[dict] = None,
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dict from run_mcq_evaluation()
        output_path: Path to save JSON file
        model_name: Name of the evaluated model
        additional_info: Optional additional metadata to include
    """
    output_data = {
        "model": model_name,
        "dataset": "openlifescienceai/MedQA-USMLE-4-options-hf",
        "split": "test",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "predictions": results["predictions"],
    }

    if additional_info:
        output_data.update(additional_info)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
