import logging
import re
from contextlib import nullcontext

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
            messages = [
                msg for msg in example["messages"] if msg["role"] != "assistant"
            ]

            return self.medqa_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return ""

    def _medqa_extract_choice(self, text: str):
        """
        Extract A, B, C, D.
        """
        # Look for explicit "Answer: X" pattern
        m: re.Match[str] | None = re.search(
            pattern=r"Answer:\s*([ABCD])", string=text, flags=re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # Look for "X:" or "X)" at the start of a line or sentence
        start_pattern: re.Match[str] | None = re.search(
            pattern=r"\b([ABCD])[\:\)\.]", string=text
        )
        if start_pattern:
            return start_pattern.group(1).upper()

        # Look for standalone letters with word boundaries
        standalone: re.Match[str] | None = re.search(
            pattern=r"\b([ABCD])\b", string=text
        )
        if standalone:
            return standalone.group(1).upper()

        return None

    def _run_medqa_eval(self):
        """
        Becuase this is a custom evaluation script, we must implement distribution ourselves.
        """
        if self.medqa_eval_dataset is None or self.medqa_tokenizer is None:
            return None

        model = self.model
        tokenizer = self.medqa_tokenizer
        device = model.device

        num_global_samples: int = min(
            self.medqa_max_samples, len(self.medqa_eval_dataset)
        )
        my_indices: list[int] = list(
            range(self.args.process_index, num_global_samples, self.args.world_size)
        )

        batch_size = 8
        prompts = [
            self._medqa_build_prompt(self.medqa_eval_dataset[i]) for i in my_indices
        ]
        labels = [self.medqa_eval_dataset[i]["label"] for i in my_indices]

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_size = "left"

        if isinstance(model, FSDP):
            fsdp_context = FSDP.summon_full_params(
                model, writeback=False, rank0_only=False
            )
        else:
            fsdp_context = nullcontext()

        local_correct = 0
        local_total = 0

        with fsdp_context:
            model_was_training = model.training
            model.eval()

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)

                with torch.no_grad():
                    # Force BF16 to stop FlashAttn from complaining about FP32 master weights
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.medqa_max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                for pred_text, gold_idx in zip(decoded_preds, batch_labels):
                    pred_letter = self._medqa_extract_choice(pred_text)
                    gold_letter = self.medqa_labels[gold_idx]

                    if pred_letter == gold_letter:
                        local_correct += 1
                    local_total += 1

            if model_was_training:
                model.train()

        tokenizer.padding_side = original_padding_side

        stats = torch.tensor(
            [local_correct, local_total], dtype=torch.float32, device=device
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        global_correct = stats[0].item()
        global_total = stats[1].item()

        return global_correct / global_total if global_total > 0 else 0.0

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        1. Run the normal SFTTrainer evaluate() to get eval_loss, etc.
        2. Run MedQA MCQ eval, add `medqa_mcq_accuracy` to metrics.
        3. Log updated metrics (so W&B also gets it).
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        acc = self._run_medqa_eval()

        # Only main process should run this extra eval
        if self.args.process_index == 0:
            if acc is not None:
                metrics["eval_medqa_mcq_accuracy"] = acc
                self.log({"eval_medqa_mcq_accuracy": acc})

                print(
                    f"[MedQASFTTrainer] Step {self.state.global_step}: "
                    f"eval_medqa_mcq_accuracy = {acc:.4f}"
                )

        return metrics
