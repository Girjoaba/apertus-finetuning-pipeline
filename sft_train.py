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
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

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

def main(script_args, training_args, model_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    #Set base directory to store model
    store_base_dir = "./" #os.getenv("STORE")

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
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    print("Starting training...")

    trainer.train()
    trainer.save_model(os.path.join(store_base_dir, training_args.output_dir))
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


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