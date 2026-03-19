import asyncio
import argparse
import logging
import torch
import sys
import os
import json
import pandas as pd
from genlm.control.sampler import DirectTokenSampler
from genlm.eval import ModelOutput, ModelResponse, run_evaluation
from genlm.control import InferenceVisualizer

from il_smc import (
    TunedLensLLM, 
    ActivationPotential, 
    MonitoredDirectTokenSampler,
    truthful_qa_prompt_formatter,
    gsm8k_prompt_formatter,
    TruthfulQADataset, 
    GSM8KDataset,
    TruthfulQAEvaluator, 
    GSM8KEvaluator,
)

from il_smc.metrics import entropy_score, kl_divergence_score

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

logger = logging.getLogger("eval_script")

def parse_args():
    parser = argparse.ArgumentParser(description="Eval Script (TruthfulQA & GSM8K)")
    
    # --- TASK SELECTION ---
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=["truthfulqa", "gsm8k"],
        help="Which benchmark to run."
    )

    # Model Args
    parser.add_argument("--model_name", type=str, default="gpt2", help="HF Model name")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index (for Tuned Lens)")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Temperature (for SMC/Sampling)")
    
    parser.add_argument("--offline", action="store_true", default=False, help="Offline mode (load local files)")
    parser.add_argument("--cache_dir", type=str, default="lens_cache", help="Cache directory")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to local CSV (for TruthfulQA offline)")

    # Inference Method
    parser.add_argument("--greedy", action="store_true", help="Run Standard Greedy Decoding (No SMC)")
    
    # SMC Args 
    parser.add_argument("--particles", type=int, default=5, help="Number of SMC particles")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument("--no_critic", action="store_true", help="Disable Tuned Lens potential (Standard SMC)")
    parser.add_argument("--weight", type=float, default=1.0, help="Potential weight")
    parser.add_argument("--ess_threshold", type=float, default=0.5, help="ESS threshold for resampling")

    # Eval Args
    parser.add_argument("--max_instances", type=int, default=0, help="Num instances (0=all)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--viz", action="store_true", help="Enable SMC visualization")
    parser.add_argument("--viz_port", type=int, default=8080, help="Port for viz server")

    parser.add_argument(
        "--metrics", 
        nargs="+", 
        default=[], 
        help="Override default metrics (e.g. --metrics bleu rouge)"
    )

    return parser.parse_args()

def get_task_components(args):
    """
    Returns the Dataset, Evaluator, and Prompt Formatter based on the selected task.
    """
    if args.task == "truthfulqa":
        logger.info("Setting up for TruthfulQA...")
        
        # 1. Dataset
        csv_path = args.csv_path if args.csv_path else "TruthfulQA.csv"
        dataset = TruthfulQADataset(split="validation", offline=args.offline, csv_path=csv_path if args.offline else None)
        
        # 2. Metrics
        metric_list = args.metrics if args.metrics else ["bleu", "rouge", "bleurt"]
        evaluator = TruthfulQAEvaluator(metrics=metric_list)
        
        # 3. Formatter
        formatter = truthful_qa_prompt_formatter
        
    elif args.task == "gsm8k":
        logger.info("Setting up for GSM8K...")
        
        # 1. Dataset
        dataset = GSM8KDataset(split="test", offline=args.offline, csv_path=args.csv_path)
        
        # 2. Metrics
        metric_list = args.metrics if args.metrics else ["accuracy"]
        evaluator = GSM8KEvaluator(metrics=metric_list)
        
        # 3. Formatter
        formatter = gsm8k_prompt_formatter
        
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return dataset, evaluator, formatter

def save_summary_csv(results_nested_list, model_name, output_dir):
    """Aggregates results and saves summary.csv"""
    logger.info("Aggregating results...")
    results_list = []
    if results_nested_list:
        for item in results_nested_list:
            if isinstance(item, list): results_list.extend(item)
            else: results_list.append(item)
    
    data = []
    for res in results_list:
        row = {'Model': model_name}
        if isinstance(res, dict):
            metrics = res.get('metadata', res.get('metrics', {}))
        else:
            metrics = getattr(res, 'metadata', getattr(res, 'metrics', {}))
        
        if metrics: row.update(metrics)
        data.append(row)

    if not data: return

    df = pd.DataFrame(data)
    # Calculate means
    summary = df.groupby('Model').mean(numeric_only=True)
    
    # Save
    csv_path = os.path.join(output_dir, 'summary.csv')
    
    # Pivot for clean reading
    try:
        results_long = summary.stack().reset_index()
        results_long.columns = ['Model', 'Metric', 'Value']
        summary_pivot = pd.pivot_table(results_long, values='Value', index='Model', columns='Metric')
        summary_pivot.to_csv(csv_path)
        print("\n" + "="*40 + "\nRESULTS SUMMARY\n" + "="*40)
        print(summary_pivot)
    except Exception:
        summary.to_csv(csv_path)
        print(summary)
    
    print("="*40)

# --- INFERENCE FUNCTIONS ---

async def run_greedy(instance, args, output_dir, replicate, llm_wrapper, formatter):
    """Greedy Decoding Path"""
    async_transformer = llm_wrapper.model
    tokenizer = async_transformer.tokenizer

    if hasattr(async_transformer, "model"):
        hf_model = async_transformer.model
    else:
        hf_model = async_transformer

    # Format Prompt
    raw_ids = formatter(tokenizer, instance, use_chat_format=False)
    if hasattr(raw_ids, "tolist"): raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list): raw_ids = raw_ids[0]

    input_ids = torch.tensor([raw_ids], dtype=torch.long).to(hf_model.device)

    # Generate
    with torch.no_grad():
        output_ids = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,        # Greedy
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not response_text: response_text = "I have no comment."

    return ModelOutput(responses=[ModelResponse(response=response_text, weight=1.0)])

async def run_smc(instance, args, output_dir, replicate, llm_wrapper, formatter, critic):
    """SMC Sampling Path"""
    # Format
    raw_ids = formatter(llm_wrapper.model.tokenizer, instance, use_chat_format=False)
    if hasattr(raw_ids, "tolist"): raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list): raw_ids = raw_ids[0]

    # Spawn & Sample
    current_llm = llm_wrapper.spawn(prompt_ids=raw_ids)
    sampler = MonitoredDirectTokenSampler(current_llm)
    
    inst_id = getattr(instance, 'instance_id', 'unk')
    json_path = os.path.join(output_dir, f"smc_{inst_id}_rep{replicate}.json")

    # Visualization
    if args.viz: visualizer = InferenceVisualizer(port=args.viz_port)

    # Run SMC
    sequences = await sampler.smc(
        n_particles=args.particles,
        max_tokens=args.max_tokens,
        verbosity=0,
        ess_threshold=args.ess_threshold,
        json_path=json_path,
        critic=critic,
    )

    # Decode
    candidates = sequences.decoded_posterior
    if not candidates:
        prompt_text = current_llm.model.tokenizer.decode(raw_ids, skip_special_tokens=True)
        for seq, weight in zip(sequences.contexts, sequences.normalized_weights):
            full_text = b"".join([b for b in seq if isinstance(b, bytes)]).decode("utf-8", errors="ignore")
            gen_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            candidates[gen_text] = candidates.get(gen_text, 0.0) + weight

    responses = []
    for seq, prob in candidates.items():
        clean = seq.strip().split("\n\n")[0].split("\nQ:")[0].strip() or "I have no comment."
        responses.append(ModelResponse(response=clean, weight=prob))
    
    if args.viz:
        visualizer.visualize(json_path)
        visualizer.shutdown_server()

    return ModelOutput(responses=responses)

# --- MAIN ---

async def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Prepare Output Dir
    safe_model = args.model_name.replace("/", "_")
    if args.greedy:
        run_name = f"{safe_model}_{args.task}_GREEDY"
    else:
        critic_tag = "NoCritic" if args.no_critic else f"W{args.weight}"
        run_name = f"{safe_model}_{args.task}_L{args.layer_idx}_P{args.particles}_{critic_tag}"
    
    final_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(final_output_dir, exist_ok=True)
    logger.info(f"Output Directory: {final_output_dir}")

    # Load Model
    logger.info(f"Loading Model: {args.model_name}")
    llm = TunedLensLLM.from_name(
        args.model_name, 
        backend="hf", 
        target_layer_idx=args.layer_idx,
        temperature=args.temperature,
        offline=args.offline, 
        cache_dir=args.cache_dir
    )

    # Load Task Components
    dataset, evaluator, formatter = get_task_components(args)

    potential = None
    if not args.greedy and not args.no_critic:
        metric_fn = entropy_score 
        potential = ActivationPotential(model=llm, metric=metric_fn, weight=args.weight)
        logger.info(f"Potential initialized (Weight={args.weight})")

    async def bound_model_fn(instance, output_dir, replicate):
        if args.greedy:
            return await run_greedy(instance, args, output_dir, replicate, llm, formatter)
        else:
            return await run_smc(instance, args, output_dir, replicate, llm, formatter, potential)

    # Run Eval
    max_inst = args.max_instances if args.max_instances > 0 else len(dataset)
    logger.info(f"Starting Evaluation on {max_inst} instances...")
    
    results = await run_evaluation(
        dataset=dataset,
        model=bound_model_fn,
        evaluator=evaluator,
        output_dir=final_output_dir,
        overwrite_results=True,
        overwrite_outputs=True,
        verbosity=1,
        max_instances=max_inst,
    )

    if results is not None:
        save_summary_csv(results["all_instance_results"], args.model_name, final_output_dir)

if __name__ == "__main__":
    asyncio.run(main())