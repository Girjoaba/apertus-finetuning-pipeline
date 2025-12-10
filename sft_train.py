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
import sys
from logging import Logger

import datasets
import torch
import transformers
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


def disable_fsdp_padding_idx(model: torch.nn.Module) -> None:
    """
    Iterates through the model to find Embedding layers and unset padding_idx.
    This prevents the FSDP sharding from crashing.
    """
    if not isinstance(model, torch.nn.Module):
        return

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            if hasattr(module, "padding_idx") and module.padding_idx is not None:
                logger.warning(f"FSDP Fix: Disabling padding_idx in layer '{name}'")
                module.padding_idx = None


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


def main(
    script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig
):
    # ========================
    # Init. Logging
    # ========================

    # Silence logging
    if training_args.process_index != 0:
        logger.setLevel(logging.WARNING)
        transformers.utils.logging.set_verbosity_warning()
        datasets.utils.logging.set_verbosity_warning()
    else:
        logger.setLevel(logging.INFO)
        transformers.utils.logging.set_verbosity_info()
        datasets.utils.logging.set_verbosity_info()

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)

    logger.info(f"Model Parameters: {model_args}")
    logger.info(f"Training Parameters: {training_args}")

    if "wandb" in (training_args.report_to or []) and training_args.process_index == 0:
        wandb_config = {
            "script_args": vars(script_args),
            "training_args": training_args.to_dict(),
            "model_args": vars(model_args),
        }

        if getattr(training_args, "fsdp", None):
            wandb_config["fsdp_config"] = getattr(training_args, "fsdp_config", "N/A")

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", default="apertus-finetune"),
            name=training_args.run_name or os.path.basename(p=training_args.output_dir),
            config=wandb_config,
        )

    if training_args.fsdp:
        logger.info(f"FSDP Detected: {training_args.fsdp}")
        device_map = None
    else:
        device_map = None

    # ========================
    # Load Model
    # ========================
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        attn_implementation=model_args.attn_implementation,
        device_map=device_map,
        trust_remote_code=model_args.trust_remote_code,
    )

    if getattr(training_args, "fsdp", None):
        disable_fsdp_padding_idx(model=model)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ========================
    # Training Mode (PEFT vs Full)
    # ========================
    if hasattr(model_args, "use_peft") and not model_args.use_peft:
        peft_config = None
    else:
        peft_config = get_peft_config(model_args)

    if peft_config is None:
        logger.info("No PEFT config found. Running FULL parameters training.")
    else:
        logger.info("PEFT config detected. Running LoRA/PEFT training.")

    # ========================
    # Dataset
    # ========================
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
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
    # Trainer
    # ========================
    trainer_cls = (
        MedQASFTTrainer if "medqa" in script_args.dataset_name.lower() else SFTTrainer
    )

    trainer_kwargs = {}
    if trainer_cls == MedQASFTTrainer:
        trainer_kwargs = {
            "medqa_eval_dataset": eval_dataset,
            "medqa_tokenizer": tokenizer,
            "medqa_max_samples": 100,
        }

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        **trainer_kwargs,
    )

    # visualize predictions
    if "wandb" in (training_args.report_to or []) and training_args.process_index == 0:
        if not getattr(training_args, "fsdp", None):
            logger.info("FSDP not detected - Adding WandbPredictionCallback")
            pred_callback = WandbPredictionCallback(
                trainer=trainer,
                tokenizer=tokenizer,
                dataset=eval_dataset,
                num_samples=25,
            )
            trainer.add_callback(pred_callback)
        else:
            logger.info(
                "FSDP detected - Skipping WandbPredictionCallback to avoid sharding complexity"
            )

    trainer.can_return_loss = True

    # ========================
    # Power it up
    # ========================
    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub()

    if (
        "wandb" in (training_args.report_to or [])
        and wandb.run is not None
        and training_args.process_index == 0
    ):
        wandb.finish()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    raw_args = parser.parse_args_and_config(return_remaining_strings=True)
    if len(raw_args) == 2:
        (script_args, training_args, model_args), _ = raw_args
    else:
        script_args, training_args, model_args, _ = raw_args

    main(script_args, training_args, model_args)
