# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import os
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
import wandb
import torch
import re

def format_medqa(example):
    # Get Question
    # Sometimes the question is split between sent1 and sent2, usually it's just sent1
    question = example.get("sent1", "").strip()
    if example.get("sent2"):
        question += " " + example["sent2"]

    # Get options
    option_keys = ["ending0", "ending1", "ending2", "ending3"]
    labels = ["A", "B", "C", "D"]
    options_text_lines = []
    options_list = []
    
    for i, key in enumerate(option_keys):
        opt_text = example.get(key, "N/A")
        options_list.append(opt_text)
        options_text_lines.append(f"{labels[i]}: {opt_text}")
    options_block = "\n".join(options_text_lines)

    # Get Correct Answer
    correct_idx = example["label"] 
    correct_label = labels[correct_idx]         # e.g., "A"
    correct_text = options_list[correct_idx]    # e.g., "Ampicillin"

    # Build User Prompt
    user_content = (
        f"Answer the following multiple choice question about medical knowledge.\n\n"
        f"{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Answer:"
    )

    # Build Assistant Answer
    assistant_content = f"{correct_label}: {correct_text}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

class MedQASFTTrainer(SFTTrainer):
    """
    SFTTrainer that, on every evaluate(), also computes MedQA multiple-choice
    accuracy via generation and adds it to the metrics dict as
    `medqa_mcq_accuracy`.
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

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
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


def main(script_args, training_args, model_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    #Set base directory to store model
    store_base_dir = "./" #os.getenv("STORE")

    # Weigths and Biases tracking
    report_to = training_args.report_to
    if isinstance(report_to, str):
        report_to = [report_to]

    if report_to and "wandb" in report_to:
        run_name = (
            getattr(training_args, "run_name", None)
            or os.environ.get("WANDB_RUN_NAME")
            or os.path.basename(training_args.output_dir)
        )

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "sft-medqa"),
            name=run_name,
            # This lets you see all hyperparameters in the W&B run config
            config={
                "script_args": vars(script_args),
                "training_args": training_args.to_dict(),
                "model_args": vars(model_args),
            },
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation=model_args.attn_implementation, 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Preprocess MedQA dataset
    if "MedQA" in script_args.dataset_name:
            print("Formatting MedQA dataset to chat format...")
            print(f"Dataset Columns detected: {dataset['train'].column_names}")
            dataset = dataset.map(format_medqa, num_proc=training_args.dataset_num_proc)

    print(f"Loaded dataset: {dataset}")

    # -------------
    # Train model
    # -------------
    trainer = MedQASFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        medqa_eval_dataset=dataset[script_args.dataset_test_split],
        medqa_tokenizer=tokenizer,
        medqa_max_samples=100,
        medqa_max_new_tokens=16,
    )

    trainer.can_return_loss = True

    print("Starting training...")

    trainer.train()
    trainer.save_model(os.path.join(store_base_dir, training_args.output_dir))
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    
    # Close WandB run
    if report_to and "wandb" in report_to and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    raw_args = parser.parse_args_and_config(return_remaining_strings=True)
    
    # Robust unpacking
    if len(raw_args) == 2:
        (script_args, training_args, model_args), remaining_args = raw_args
    else:
        script_args, training_args, model_args, remaining_args = raw_args

    print("Script arguments:", script_args)
    print("Training arguments:", training_args)
    print("Model arguments:", model_args)
    main(script_args, training_args, model_args)