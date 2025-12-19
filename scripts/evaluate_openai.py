#!/usr/bin/env python3
"""
Evaluate OpenAI models (GPT-4, GPT-3.5-turbo, etc.) on MedQA test split.

Usage:
    python scripts/evaluate_openai.py --model gpt-4o [--max_samples N] [--output results/openai_gpt4o_test.json]

Requires:
    - OPENAI_API_KEY environment variable set
    - openai package installed: pip install openai
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Setup logging configuration with optional file output.
    """
    handlers = [
        logging.StreamHandler(),
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
        force=True,
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

# Available models
OPENAI_MODELS = [
    # GPT-5 family
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-4.1 family
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o family
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    # Reasoning models
    "o3",
    "o4-mini",
    "o1-preview",
    "o1-mini",
]


def format_medqa_prompt(example: dict) -> str:
    """
    Format a MedQA example into the prompt format.
    Same format as used for Apertus evaluation.
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


def extract_choice(text: str) -> Optional[str]:
    """
    Extract A, B, C, or D from generated text.
    Uses multiple patterns to robustly extract the answer.
    Returns None if no valid choice found.
    """
    if not text:
        return None
    
    # Normalize text
    text_clean = text.strip()
    
    # Pattern 1: Explicit "Answer: X" or "Answer is X" pattern (highest priority)
    m = re.search(r"(?:answer|correct answer|correct option)(?:\s+is)?[:\s]+([ABCD])\b", text_clean, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Pattern 2: "X:" or "X)" or "X." at start of response or after newline
    start_pattern = re.search(r"(?:^|\n)\s*([ABCD])[\:\)\.\s]", text_clean)
    if start_pattern:
        return start_pattern.group(1).upper()
    
    # Pattern 3: "(X)" format like "(A)" or "( A )"
    paren_pattern = re.search(r"\(\s*([ABCD])\s*\)", text_clean, re.IGNORECASE)
    if paren_pattern:
        return paren_pattern.group(1).upper()
    
    # Pattern 4: "Option X" or "choice X"
    option_pattern = re.search(r"(?:option|choice)\s+([ABCD])\b", text_clean, re.IGNORECASE)
    if option_pattern:
        return option_pattern.group(1).upper()
    
    # Pattern 5: Letter followed by colon/paren anywhere in text
    anywhere_pattern = re.search(r"\b([ABCD])[\:\)]", text_clean)
    if anywhere_pattern:
        return anywhere_pattern.group(1).upper()
    
    # Pattern 6: Standalone letter with word boundaries (last resort, first match)
    standalone = re.search(r"\b([ABCD])\b", text_clean)
    if standalone:
        return standalone.group(1).upper()
    
    # Pattern 7: Check if the FIRST word/character is A, B, C, or D
    first_char = text_clean.lstrip()[:1].upper()
    if first_char in "ABCD":
        return first_char
    
    return None


# Models that use max_completion_tokens instead of max_tokens
NEW_API_MODELS = [
    "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "o3", "o4-mini", "o1-preview", "o1-mini",
]

# Reasoning models need more tokens (includes thinking + output)
REASONING_MODELS = [
    "gpt-5.1", "gpt-5-mini", "gpt-5-nano",
    "o3", "o4-mini", "o1-preview", "o1-mini",
]


def call_openai_api(
    client,
    model: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Call OpenAI API with retry logic for rate limits.
    Handles both old (max_tokens) and new (max_completion_tokens) API formats.
    """
    # Determine which parameter to use based on model
    use_new_api = model in NEW_API_MODELS
    
    for attempt in range(max_retries):
        try:
            # Build request parameters
            request_params = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical expert. Answer the multiple choice question by stating the letter of the correct answer (A, B, C, or D).",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            
            # Use appropriate token limit parameter and temperature
            if use_new_api:
                # Reasoning models need more tokens (thinking + output)
                if model in REASONING_MODELS:
                    request_params["max_completion_tokens"] = 2048
                else:
                    request_params["max_completion_tokens"] = 100
                # New models don't support temperature=0, use default
            else:
                request_params["max_tokens"] = 50
                request_params["temperature"] = 0  # Greedy decoding for reproducibility
            
            response = client.chat.completions.create(**request_params)
            
            # Handle different response structures
            choice = response.choices[0]
            
            # Log full response for debugging new models
            if use_new_api:
                logger.info(f"Full response for {model}: finish_reason={choice.finish_reason}, message={choice.message}")
            
            # Try standard message content first
            if choice.message and choice.message.content:
                return choice.message.content.strip()
            
            # For reasoning models, check if there's a different field
            # Log the full response structure for debugging
            logger.warning(f"Empty content in response. Choice: {choice}")
            
            # Return empty string if no content found
            return ""

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = retry_delay * (2**attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"API error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    return ""


def load_medqa_test_split(max_samples: Optional[int] = None) -> list[dict]:
    """
    Load and format the MedQA test split.
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

    for example in dataset:
        prompt = format_medqa_prompt(example)
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


def run_evaluation(
    client,
    model: str,
    test_data: list[dict],
    requests_per_minute: int = 60,
) -> dict:
    """
    Run evaluation on test data using OpenAI API.
    """
    predictions = []
    correct = 0
    total = 0

    # Rate limiting
    delay_between_requests = 60.0 / requests_per_minute

    logger.info(f"Running evaluation with {model} on {len(test_data)} samples")
    logger.info(f"Rate limit: {requests_per_minute} requests/min ({delay_between_requests:.2f}s delay)")

    for i, data in enumerate(tqdm(test_data, desc=f"Evaluating {model}")):
        prompt = data["prompt"]
        gold_label = data["gold_label"]

        # Call API
        response_text = call_openai_api(client, model, prompt)

        # Extract answer
        pred_letter = extract_choice(response_text)
        is_correct = pred_letter == gold_label

        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "index": i,
            "predicted": pred_letter,
            "gold": gold_label,
            "correct": is_correct,
            "generated_text": response_text,
        })

        # Log question and answer
        log_question_answer(
            index=i,
            question=data.get("question", ""),
            options=data.get("options", ""),
            generated_text=response_text,
            predicted=pred_letter,
            gold=gold_label,
            correct=is_correct,
        )

        # Rate limiting delay
        if i < len(test_data) - 1:
            time.sleep(delay_between_requests)

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
) -> None:
    """
    Save evaluation results to JSON file.
    """
    output_data = {
        "model": model_name,
        "dataset": "openlifescienceai/MedQA-USMLE-4-options-hf",
        "split": "test",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "model_type": "openai",
        "predictions": results["predictions"],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI models on MedQA test split"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=OPENAI_MODELS,
        help=f"OpenAI model to evaluate (default: gpt-4o-mini). Options: {OPENAI_MODELS}",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--requests_per_minute",
        type=int,
        default=60,
        help="Rate limit for API requests (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/openai_{model}_test.json)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Detailed log file path (default: logs/eval_openai_{model}_detailed.log)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it: export OPENAI_API_KEY='your-key-here'"
        )
        return

    # Import OpenAI
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return

    # Determine log file path
    log_file = args.log_file
    if log_file is None:
        model_safe_name = args.model.replace("-", "_").replace(".", "_")
        log_file = f"logs/eval_openai_{model_safe_name}_detailed.log"

    # Setup logging with file output
    setup_logging(log_file=log_file)

    logger.info("=" * 60)
    logger.info(f"MedQA Test Evaluation - OpenAI {args.model}")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Load test data
    test_data = load_medqa_test_split(max_samples=args.max_samples)

    # Run evaluation
    results = run_evaluation(
        client=client,
        model=args.model,
        test_data=test_data,
        requests_per_minute=args.requests_per_minute,
    )

    # Determine output path
    output_path = args.output
    if output_path is None:
        model_safe_name = args.model.replace("-", "_").replace(".", "_")
        output_path = f"results/openai_{model_safe_name}_test.json"

    # Save results
    save_results(
        results=results,
        output_path=output_path,
        model_name=f"openai/{args.model}",
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Model: {args.model}")
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
