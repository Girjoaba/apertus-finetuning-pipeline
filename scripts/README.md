# MedQA Model Evaluation

Evaluate HuggingFace models on the MedQA test split (1,273 USMLE-style questions).

## Quick Start

```bash
# Evaluate a base model (from HuggingFace)
python scripts/evaluate_template.py --base_model your-org/your-model

# Evaluate with LoRA adapter (from HuggingFace)
python scripts/evaluate_template.py \
    --base_model swiss-ai/Apertus-8B-Instruct-2509 \
    --adapter your-org/your-lora-adapter

# Quick test (10 samples)
python scripts/evaluate_template.py --base_model your-model --max_samples 10
```

## Local Weights

Both `--base_model` and `--adapter` accept local paths:

```bash
# Local base model
python scripts/evaluate_template.py --base_model /path/to/your/model

# Local LoRA adapter with HuggingFace base
python scripts/evaluate_template.py \
    --base_model swiss-ai/Apertus-8B-Instruct-2509 \
    --adapter ./output/my_lora_checkpoint

# Both local
python scripts/evaluate_template.py \
    --base_model /scratch/models/llama-8b \
    --adapter /scratch/adapters/my_lora
```

## SLURM (Alps Cluster)

```bash
# Copy and edit the template
cp scripts/alps/eval_template.sbatch scripts/alps/eval_mymodel.sbatch
# Edit BASE_MODEL and ADAPTER variables in the file

# Submit job
sbatch scripts/alps/eval_mymodel.sbatch
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base_model` | HuggingFace model ID | Required |
| `--adapter` | LoRA adapter (optional) | None |
| `--batch_size` | Reduce if OOM | 8 |
| `--max_samples` | Limit samples for testing | All |

## Compare Results

```bash
python scripts/compare_results.py --all --plot
```

## Output

- Results: `results/<model>_test.json`
- Logs: `logs/eval_<model>_detailed.log`

For detailed docs, see [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).
