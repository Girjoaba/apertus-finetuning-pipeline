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

import logging
import os
from logging import Logger

import wandb
from datasets import load_dataset
from src.callbacks import WandbPredictionCallback
from src.data_utils import get_formatting_func
from src.trainers import MedQASFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(name=__name__)


def sanity_check_dataset(dataset, tokenizer) -> None:
    """
    Inspect message before training...
    """
    logger.info("Inspecting messages...")
    sample = dataset[0]
    if "messages" in sample:
        logger.info(f"Sample messages:\n{sample['messages']}")
        formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        logger.info(f"Formatted text (first 200 chars):\n{formatted[:200]}...")
    else:
        logger.warning("Dataset does not contain 'messages' key after formatting!")


def main(script_args, training_args, model_args):
    # ========================
    # Init. Logging
    # ========================
    logger.info(f"Model Parameters: {model_args}")
    logger.info(f"Training Parameters: {training_args}")

    if "wandb" in (training_args.report_to or []):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", default="sft-project"),
            name=training_args.run_name or os.path.basename(p=training_args.output_dir),
            config={
                "script_args": vars(script_args),
                "training_args": training_args.to_dict(),
                "model_args": vars(model_args),
            },
        )

    # ========================
    # Load Model
    # ========================
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation=model_args.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ========================
    # Dataset
    # ========================
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Dynamic Formatter Selection
    formatting_func = get_formatting_func(script_args.dataset_name)

    if formatting_func:
        logger.info(f"Applying formatting for {script_args.dataset_name}...")
        dataset = dataset.map(formatting_func, num_proc=training_args.dataset_num_proc)
    else:
        logger.warning(
            f"No specific formatter found for {script_args.dataset_name}. Assuming dataset is already formatted."
        )

    # Split dataset
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]

    # Validate before training
    sanity_check_dataset(train_dataset, tokenizer)

    # ========================
    # Training Mode (PEFT vs Full)
    # ========================
    peft_config = get_peft_config(model_args)

    if peft_config is None:
        logger.info("No PEFT config found. Running FULL parameters training.")
    else:
        logger.info("PEFT config detected. Running LoRA/PEFT training.")

    # ========================
    # Trainer
    # ========================
    trainer_cls = (
        MedQASFTTrainer if "medqa" in script_args.dataset_name.lower() else SFTTrainer
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        **(
            {
                "medqa_eval_dataset": eval_dataset,
                "medqa_tokenizer": tokenizer,
                "medqa_max_samples": 100,
            }
            if trainer_cls == MedQASFTTrainer
            else {}
        ),
    )

    # visualize predictions
    if "wandb" in (training_args.report_to or []):
        pred_callback = WandbPredictionCallback(
            trainer=trainer, tokenizer=tokenizer, dataset=eval_dataset, num_samples=10
        )
        trainer.add_callback(pred_callback)

    trainer.can_return_loss = True

    # ========================
    # Power it up
    # ========================
    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    if "wandb" in (training_args.report_to or []) and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    raw_args = parser.parse_args_and_config(return_remaining_strings=True)
    if len(raw_args) == 2:
        (script_args, training_args, model_args), _ = raw_args
    else:
        script_args, training_args, model_args, _ = raw_args

    main(script_args, training_args, model_args)
