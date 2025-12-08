import logging
import re

import torch
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class MedQASFTTrainer(SFTTrainer):
    """
    SFTTrainer that, on every evaluate(), also computes MedQA multiple-choice accuracy.
    """

    def __init__(
        self,
        *args,
        medqa_eval_dataset=None,
        medqa_tokenizer=None,
        medqa_max_samples: int = 200,
        medqa_max_new_tokens: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.medqa_eval_dataset = medqa_eval_dataset
        self.medqa_tokenizer = medqa_tokenizer
        self.medqa_max_samples = medqa_max_samples
        self.medqa_max_new_tokens = medqa_max_new_tokens
        self.medqa_labels = ["A", "B", "C", "D"]

    def _medqa_build_prompt(self, example):
        # If you kept messages from format_medqa, use chat template:
        if "messages" in example and example["messages"]:
            return self.medqa_tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback: reconstruct like format_medqa
        question = example.get("sent1", "").strip()
        if example.get("sent2"):
            question += " " + example["sent2"]

        option_keys = ["ending0", "ending1", "ending2", "ending3"]
        options_lines = []
        for i, key in enumerate(option_keys):
            opt_text = example.get(key, "N/A")
            options_lines.append(f"{self.medqa_labels[i]}: {opt_text}")
        options_block = "\n".join(options_lines)

        return (
            f"Answer the following multiple choice question about medical knowledge.\n\n"
            f"{question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"Answer:"
        )

    def _medqa_extract_choice(self, text: str):
        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)
        for ch in self.medqa_labels:
            if ch in text:
                return ch
        return None

    def _run_medqa_eval(self):
        if self.medqa_eval_dataset is None or self.medqa_tokenizer is None:
            return None

        model = self.model
        tokenizer = self.medqa_tokenizer
        device = model.device

        model_was_training = model.training
        model.eval()

        correct = 0
        total = 0
        num_samples = min(self.medqa_max_samples, len(self.medqa_eval_dataset))

        for idx in range(num_samples):
            ex = self.medqa_eval_dataset[idx]
            prompt = self._medqa_build_prompt(ex)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.medqa_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            pred_letter = self._medqa_extract_choice(gen_text)
            gold_idx = ex["label"]
            gold_letter = self.medqa_labels[gold_idx]

            if pred_letter == gold_letter:
                correct += 1
            total += 1

        if model_was_training:
            model.train()

        return correct / total if total > 0 else 0.0

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        1. Run the normal SFTTrainer evaluate() to get eval_loss, etc.
        2. Run MedQA MCQ eval, add `medqa_mcq_accuracy` to metrics.
        3. Log updated metrics (so W&B also gets it).
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # Only main process should run this extra eval
        if self.args.local_rank in (-1, 0):
            acc = self._run_medqa_eval()
            if acc is not None:
                metrics["eval_medqa_mcq_accuracy"] = acc
                # log merged metrics again so W&B sees it
                self.log({"eval_medqa_mcq_accuracy": acc})

                print(
                    f"[MedQASFTTrainer] Step {self.state.global_step}: "
                    f"eval_medqa_mcq_accuracy = {acc:.4f}"
                )

        return metrics
