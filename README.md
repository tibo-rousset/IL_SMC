# Intermediate-Layer Sequential Monte Carlo: Steering LLM Generation via Internal Representations

This repository contains the code for the experiments described in the paper *Intermediate-Layer Sequential Monte Carlo: Steering LLM Generation via Internal Representations*.

For related work, see [genlm/genlm-control](https://github.com/genlm/genlm-control). Our experiments leverage the [genlm/genlm-eval](https://github.com/genlm/genlm-eval) library.

## Setup

### Requirements
- Python >= 3.11 (required for GenLM support)
- A GPU with CUDA support
- The dependencies in `pyproject.toml`

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/tibo-rousset/IL_SMC.git
    ```

2. Install the dependencies:
    ```bash
    cd IL_SMC
    pip install -e .
    ```

   **Note**: We recommend using a virtual environment to manage dependencies. For example, using `conda`:

    ```bash
    conda create -n il_smc python=3.11
    conda activate il_smc
    pip install -e .
    ```

### Model Setup

IL_SMC builds on the [**GenLM**](https://github.com/genlm) stack. Ensure that the compatible model adapters are set up. The necessary integrations can be found in `src/il_smc/llms.py`. Add your preferred transformer model and the Tuned Lens implementation to the Python path if required.

## Running Evaluation

Use `run_eval.py` to evaluate on [**TruthfulQA**](https://github.com/sylinrl/TruthfulQA) or [**GSM8K**](https://github.com/openai/grade-school-math).

### Example Commands:

- **TruthfulQA**:
    ```bash
    python scripts/run_eval.py --task truthfulqa --model_name gpt2 --particles 5 --max_tokens 50 --output_dir results
    ```
  
- **GSM8K**:
    ```bash
    python scripts/run_eval.py --task gsm8k --model_name gpt2 --particles 5 --max_tokens 50 --output_dir results
    ```

### Key Arguments:
- `--task`: Select either `truthfulqa` or `gsm8k`.
- `--layer_idx`: Intermediate layer to derive the signals from (default to last layer).
- `--model_name`: Model to use (e.g., `gpt2`).
- `--particles`: Number of SMC particles (default `5`).
- `--max_tokens`: Maximum tokens to generate (default `50`).
- `--greedy`: Use greedy decoding instead of SMC.
- `--output_dir`: Directory for saving results.

For a full description of the arguments, please refer to [this guide](scripts/README.md).

### Evaluation Output
Results are saved in `--output_dir` as `summary.csv` and other metrics files.

## Acknowledgments

This project relies heavily on the following libraries:

- **[GenLM Eval](https://github.com/genlm/genlm-eval)**: A library for evaluation scripts and tools for large language models, licensed under the **Apache 2.0 License**.
- **[GenLM Control](https://github.com/genlm/genlm-control)**: A library providing controlled generation methods for large language models, licensed under the **Apache 2.0 License**.

These libraries form the foundation of the IL_SMC framework and are integral to running the experiments described in the paper. We would like to thank the authors and contributors for their valuable work.

## Contributing
Issues and PRs are welcome. Please include reproducible examples when possible.

## License
See `pyproject.toml` for licensing information.
