# Evaluation Script for IL_SMC

This script is used to evaluate the **Intermediate-Layer Sequential Monte Carlo (IL_SMC)** framework on the **TruthfulQA** and **GSM8K** datasets. It supports both **greedy decoding** and **Sequential Monte Carlo (SMC)** sampling methods for model generation.

## Usage

You can run the evaluation script with different configurations for either **TruthfulQA** or **GSM8K** datasets. The script allows you to control various aspects of the evaluation process, including the sampling method, the number of particles for SMC, and whether to visualize the results.

```bash
python scripts/run_eval.py --task <task_name> --model_name <model_name> [additional arguments]
```

## Arguments

The following table provides a detailed description of the arguments used to configure the evaluation script.

### Task Selection

| Argument       | Description                                                                                           | Default  |
|----------------|-------------------------------------------------------------------------------------------------------|----------|
| `--task`       | Specifies which dataset to run the evaluation on. Choose between `truthfulqa` or `gsm8k`.              | *Required* |

### Model Arguments

| Argument       | Description                                                                                           | Default   |
|----------------|-------------------------------------------------------------------------------------------------------|-----------|
| `--model_name` | The model to be used for evaluation (e.g., `gpt2`, `bert-base-uncased`).                              | `gpt2`    |
| `--layer_idx`  | The index of the transformer layer to derive intermediate signals from (default is `-1` for the last layer). | `-1`      |
| `--temperature`| Temperature for sampling during generation.                            | `0.0001`  |
| `--offline`    | If set, the script will run in offline mode, using locally available files.                           | `False`   |
| `--cache_dir`  | Directory for caching model files.                                                                    | `lens_cache` |
| `--csv_path`   | Path to a local CSV file for **TruthfulQA** evaluation in offline mode.                               | *None*    |

### Inference Method

| Argument      | Description                                                                                           | Default |
|---------------|-------------------------------------------------------------------------------------------------------|---------|
| `--greedy`    | If set, enables greedy decoding (no SMC). Uses the model to predict tokens deterministically.        | `False` |

### SMC Arguments

| Argument          | Description                                                                                           | Default |
|-------------------|-------------------------------------------------------------------------------------------------------|---------|
| `--particles`     | Number of particles for Sequential Monte Carlo sampling.                                         | `5`     |
| `--max_tokens`    | Maximum number of tokens to generate.                                                                  | `50`    |
| `--no_critic`     | If set, disables the **Tuned Lens** potential, using standard SMC sampling without additional guidance. | `False` |
| `--weight`        | Weight for the **Tuned Lens** potential, used when the critic is not disabled.                         | `1.0`   |
| `--ess_threshold` | Effective Sample Size (ESS) threshold for resampling during the SMC process.                           | `0.5`   |

### Evaluation Arguments

| Argument          | Description                                                                                           | Default |
|-------------------|-------------------------------------------------------------------------------------------------------|---------|
| `--max_instances` | The number of instances to evaluate. If set to `0`, all instances in the dataset will be evaluated.    | `0`     |
| `--output_dir`    | Directory where the evaluation results will be saved.                                                  | `results` |
| `--verbose`       | Enables debug-level logging, providing more detailed output during execution.                         | `False` |
| `--viz`           | If set, the script enables **SMC visualization** and launches a visualization server.                  | `False` |
| `--viz_port`      | Port for the visualization server (if `--viz` is enabled).                                              | `8080`  |
| `--metrics`       | A list of evaluation metrics to override the default (e.g., `bleu`, `rouge`, `accuracy`).              | `[]`    |

### Example Commands

#### 1. Run **TruthfulQA** evaluation with greedy decoding:

```bash
python scripts/run_eval.py --task truthfulqa --model_name gpt2 --greedy --output_dir results
```

#### 2. Run GSM8K evaluation with IL-SMC and custom particle number:

```bash
python scripts/run_eval.py --task gsm8k --model_name gpt2 --particles 10 --weight 1.5  --ess_threshold 0.7 --max_tokens 100 --output_dir results
```

#### 3. Run TruthfulQA evaluation with standard SMC:

```bash
python scripts/run_eval.py --task truthfulqa --model_name gpt2 --no_critic --particles 10 --output_dir results
```