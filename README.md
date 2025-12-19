# apertus-finetuning-pipeline
Large Scale AI Engineering 2025 - Course Project


<!-- ## Finetuning Recipes

Follow [this guide](https://github.com/swiss-ai/Apertus-finetuning-recipes) on how to finetune Apertus. -->

## Set-Up on the Alps Supercomputer (~15 min)

We must create a container for this. Please follow the instructions step-by-step.

<!-- We finetune our model on our Alps supercomputer nocdes. We provide 2 ways to do it:

### Create an environment

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

### Create a container -->

1. Acquire a compute node ($GROUP=large-sc-2 for us)
```bash
srun --account=$GROUP -p debug --time=01:00:00 --pty bash
```

2. Create your own image. This takes about 10 minutes.
```bash
podman build -t apertus_finetune:v1 .

# After it is built save it somewhere
cd /iopsstor/scratch/cscs/$USER
enroot import -o apertus_finetune.sqsh podman://apertus_finetune:v1
```

Now you can exit the interactive session.

3. Create a toml file and name it. We will refer to this file name by `$ENVIRONMENT`:
```toml
# Update $USER with your username
image = "/iopsstor/scratch/cscs/$USER/apertus_finetune.sqsh"

mounts = [
    "/capstor",
    "/iopsstor",
    "/users",
]

workdir = "/workspace"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

4. Copy it in the `~/.edf/` folder.

5. Set in the `$REPO_ROOT/config.sh` the `$ENVIRONMENT` variable.
```bash
export SCRATCH_DIR="/iopsstor/scratch/cscs/$USER"   
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton_cache"

# make this point to your environment defined in .edf
export ENVIRONMENT=  # <--- this one
```

6. Verify 
```bash
srun --account=$GROUP --environment=$ENVIRONMENT -p debug --pty bash
pip list | grep -E "kernels|peft|trl|transformers|deepspeed|accelerate|lm_eval"
exit
```


## Tracking Training with wandb

Log into wandb to get your personal API key and add your API key to the `$REPO_ROOT/config.sh` file.

```bash
export WANDB_API_KEY="YOUR_API_KEY" # <-- put your wandb api key here
export WANDB_DIR="/iopsstor/scratch/cscs/$USER/wandb"
export WANDB_PROJECT="apertus-finetune"
```

You will then be able to see real-time performance assessment during training here: https://wandb.ai/lsae/apertus-finetune
The metric *eval/medqa_mcq_accuracy* gives the multiple-choice accuracy.

## Training

### Small model:
To train the 8B model with LoRa:
```bash
sbatch scripts/alps/single_gpu_lora_8B.sbatch
```

To perform full parameter finetuning with LoRa on the 8B:
```bash
sbatch scripts/alps/multi_gpu_full_param_8B.sbatch
```

To train the 70B model with LoRa:
```bash
sbatch scripts/alps/multi_gpu_lora_70B.sbatch
```


### Large model:


See the results on W&B.

## Run hyperparameter optimization with Wandb sweeps

Get a sweep ID (only when starting a new optimization round):
```bash
srun --account=large-sc-2 -p debug --environment="/iopsstor/scratch/cscs/wglanzer/group_project/container/container.toml"--pty bash
cd /iopsstor/scratch/cscs/$USER/group_project/apertus-finetuning-pipeline
wandb sweep configs/sweep.yaml
```

Add sweep ID to *scripts/alps/single_gpu_lora_8B_sweep.sbatch* and modify "#SBATCH --array=1-50%10" to sepcfiy how many hyperparameter combinations to test.

Run a sweep.
```bash
bash scripts/alps/single_gpu_lora_8B_sweep.sbatch
```

On Wandb filter for Sweeps by selecting the broom icon.

<!-- ## Datasets

A first attempt is to finetune for the medical domain. But, see [this repository](https://github.com/mlabonne/llm-datasets?tab=readme-ov-file) for many LLM training datasets.



## References -->
