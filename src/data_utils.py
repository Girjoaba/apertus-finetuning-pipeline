import logging

logger = logging.getLogger(__name__)


def format_medqa(example):
    """
    Formats the MedQA dataset into a chat structure.
    """
    # Get Question
    question = example.get("sent1", "").strip()
    if example.get("sent2"):
        question += " " + example["sent2"]

    # Get options
    option_keys = ["ending0", "ending1", "ending2", "ending3"]
    labels = ["A", "B", "C", "D"]
    options_list = []
    options_text_lines = []

    for i, key in enumerate(option_keys):
        opt_text = example.get(key, "N/A")
        options_list.append(opt_text)
        options_text_lines.append(f"{labels[i]}: {opt_text}")

    options_block = "\n".join(options_text_lines)

    # Get Correct Answer
    correct_idx = example["label"]
    correct_label = labels[correct_idx]
    correct_text = options_list[correct_idx]

    # Build User Prompt
    user_content = (
        f"Answer the following multiple choice question about medical knowledge.\n\n"
        f"{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Answer:"
    )

    assistant_content = f"{correct_label}: {correct_text}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# Registry for supported datasets
DATASET_FORMATTERS = {
    "medqa": format_medqa,
}


def get_formatting_func(dataset_name):
    """Returns the formatting function based on dataset name match."""
    for key, func in DATASET_FORMATTERS.items():
        if key.lower() in dataset_name.lower():
            return func
    return None
