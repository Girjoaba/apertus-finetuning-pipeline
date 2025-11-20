# apertus-finetuning-pipeline
Large Scale AI Engineering 2025 - Course Project


## Finetuning Recipes

Follow [this guide](https://github.com/swiss-ai/Apertus-finetuning-recipes) on how to finetune Apertus.

## Running on the Alps Supercomputer

We finetune our model on our Alps supercomputer nocdes. These instructions replicate how we did it:

### Create an environment

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

### Run a first 1 GPU Lora finetuning script
```bash
sbatch -A large-sc-2 single_gpu_alps.sbatch
```

## Datasets

A first attempt is to finetune for the medical domain. But, see [this repository](https://github.com/mlabonne/llm-datasets?tab=readme-ov-file) for many LLM training datasets.


## References
