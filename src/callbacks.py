import wandb
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class WandbPredictionCallback(TrainerCallback):
    """
    Custom callback to log a sample of model predictions to W&B at the end of evaluation.
    """

    def __init__(self, trainer, tokenizer, dataset, num_samples=10):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self._log_artifacts = True  # Flag to ensure we only log dataset once if needed

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only run on the main process and if W&B is active
        if args.local_rank not in (-1, 0) or wandb.run is None:
            return

        if self._log_artifacts:
            # Create a W&B Table for the raw dataset
            raw_table = wandb.Table(data=self.dataset.to_pandas().head(100))
            wandb.log({"validation_dataset_sample": raw_table})
            self._log_artifacts = False

        model = self.trainer.model
        device = model.device

        # Grab a small subset to visualize
        subset = self.dataset.select(range(self.num_samples))

        columns = ["Question", "Label", "Prediction", "Exact Match"]
        data = []

        # Put model in eval
        was_training = model.training
        model.eval()

        # Populate Table
        for sample in subset:
            prompt = self.trainer._medqa_build_prompt(sample)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Extract basic info
            label_idx = sample.get("label", -1)
            labels = ["A", "B", "C", "D"]
            gold_label = labels[label_idx] if 0 <= label_idx < 4 else "N/A"

            # Simple check
            match = (gold_label in response) or (
                response.strip().startswith(gold_label)
            )

            data.append([prompt, gold_label, response, match])

        if was_training:
            model.train()

        wandb.log(
            {
                "eval_predictions": wandb.Table(columns=columns, data=data),
                "global_step": state.global_step,
            }
        )
