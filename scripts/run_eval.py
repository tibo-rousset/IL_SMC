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

# Import your custom modules
from genlm_project.metrics import *
from genlm_project import (
    TunedLensLLM, 
    ActivationPotential, 
    TruthfulQADataset, 
    TruthfulQAEvaluator, 
    MonitoredDirectTokenSampler,
    truthful_qa_prompt_formatter,
)

# Setup Logger
logger = logging.getLogger("eval_script")

def parse_args():
    parser = argparse.ArgumentParser(description="Run TruthfulQA Evaluation")
    
    # Model Args
    parser.add_argument("--model_name", type=str, default="gpt2", help="HF Model name")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Temperature")

    parser.add_argument("--offline", action="store_true", default=False, help="Offline mode")
    parser.add_argument("--cache_dir", type=str, default="lens_cache", help="Cache directory for offline mode")
    
    # Generation Args
    parser.add_argument("--max_tokens", type=int, default=30, help="Max tokens")
    parser.add_argument("--particles", type=int, default=5, help="SMC particles")
    parser.add_argument("--weight", type=float, default=1.0, help="Potential weight")
    parser.add_argument("--ess_threshold", type=float, default=0.5, help="ESS threshold")
    
    # Eval Args
    parser.add_argument("--max_instances", type=int, default=5, help="Num instances (0=all)")
    parser.add_argument("--output_dir", type=str, default="truthfulqa_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--viz_port", type=int, default=8080, help="Port for visualization server")

    parser.add_argument(
        "--metrics", 
        nargs="+", 
        default=["bleu", "rouge", "bleurt"], 
        choices=["bleu", "rouge", "bleurt", "mc"],
        help="List of metrics to include in the summary CSV (e.g. --metrics bleu rouge)"
    )

    return parser.parse_args()

def save_summary_csv(results_nested_list, model_name, output_dir):
    """
    Aggregates evaluation results and saves a summary CSV matching 
    the official TruthfulQA format.
    
    Handles the nested list structure from run_evaluation: [[Res1], [Res2], ...]
    """
    logger.info("Aggregating results into summary CSV...")

    results_list = []
    if results_nested_list:
        for item in results_nested_list:
            if isinstance(item, list):
                results_list.extend(item)
            else:
                results_list.append(item)
    
    data = []
    for res in results_list:
        row = {'Model': model_name}
        
        metrics_dict = {}
        
        # 1. Try accessing as dictionary
        if isinstance(res, dict):
            if 'metadata' in res:
                metrics_dict = res['metadata']
            elif 'metrics' in res:
                metrics_dict = res['metrics']
            elif 'results' in res:
                sub = res['results'][0]
                metrics_dict = sub['metadata']
        
        else:
            metrics_dict = getattr(res, 'metadata', getattr(res, 'metrics', {}))
        
        if metrics_dict:
            row.update(metrics_dict)
        
        data.append(row)

    if not data:
        logger.warning("No data found to aggregate.")
        return

    df = pd.DataFrame(data)

    summary = df.groupby('Model').mean(numeric_only=True)

    results_long = summary.stack().reset_index()
    results_long.columns = ['Model', 'Metric', 'Value']

    summary_pivot = pd.pivot_table(results_long, values='Value', index='Model', columns='Metric')
    
    csv_path = os.path.join(output_dir, 'summary.csv')
    summary_pivot.to_csv(csv_path)
    logger.info(f"Summary CSV saved to: {csv_path}")
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    print(summary_pivot)
    print("="*40)

async def inference_fn(instance, args, output_dir, replicate, llm_wrapper, critic=None):
    # 1. Format Prompt
    raw_ids = truthful_qa_prompt_formatter(
        llm_wrapper.model.tokenizer, instance, use_chat_format=False
    )
    if hasattr(raw_ids, "tolist"): raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list): raw_ids = raw_ids[0]

    # 2. Spawn Model
    current_llm = llm_wrapper.spawn(prompt_ids=raw_ids)

    # 3. Initialize Sampler
    sampler = MonitoredDirectTokenSampler(current_llm)
    
    inst_id = instance.instance_id if hasattr(instance, 'instance_id') else "unk"

    json_filename = f"smc_record_{inst_id}_rep{replicate}.json"
    full_json_path = os.path.join(output_dir, json_filename)

    # 4. Run SMC Sampling
    if args.viz:
        visualizer = InferenceVisualizer(port=args.viz_port)

    sequences = await sampler.smc(
        n_particles=args.particles,
        max_tokens=args.max_tokens,
        verbosity=0,
        ess_threshold=args.ess_threshold,
        json_path=full_json_path,
        critic=critic,
    )

    # 5. Decode
    candidates = sequences.decoded_posterior
    prompt_text = current_llm.model.tokenizer.decode(raw_ids, skip_special_tokens=True)
    responses = []

    if not candidates:
        for seq, weight in zip(sequences.contexts, sequences.normalized_weights):
            full_text = b"".join([b for b in seq if isinstance(b, bytes)]).decode("utf-8", errors="ignore")
            gen_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            candidates[gen_text] = candidates.get(gen_text, 0.0) + weight

    for sequence, prob in candidates.items():
        clean_resp = sequence.strip().split("\n\n")[0].split("\nQ:")[0].strip() or "I have no comment."
        responses.append(ModelResponse(response=clean_resp, weight=prob))
    
    if args.viz:
        logger.info(f"Generating visualization for instance ID {inst_id} on port {args.viz_port}...")
        visualizer.visualize(full_json_path)
        visualizer.shutdown_server()

    return ModelOutput(responses=responses)

async def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    safe_model_name = args.model_name.replace("/", "_")
    
    run_name = (
        f"{safe_model_name}_"
        f"L{args.layer_idx}_"
        f"T{args.temperature}_"
        f"P{args.particles}_"
        f"W{args.weight}"
    )
    
    final_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {final_output_dir}")

    logger.info(f"Loading TunedLensLLM: {args.model_name}...")
    llm = TunedLensLLM.from_name(
        args.model_name, 
        backend="hf", 
        target_layer_idx=args.layer_idx,
        temperature=args.temperature,
        offline=args.offline,
        cache_dir=args.cache_dir
    )

    metric_fn = entropy_score
    potential = ActivationPotential(model=llm, metric=metric_fn, weight=args.weight)
    logger.info("Potential initialized.")

    logger.info("Initializing Dataset & Evaluator...")
    dataset = TruthfulQADataset(split="validation", offline=args.offline, csv_path="TruthfulQA.csv" if args.offline else None)
    evaluator = TruthfulQAEvaluator(metrics = args.metrics)

    async def bound_model_fn(instance, output_dir, replicate):
        return await inference_fn(
            instance, 
            args,
            output_dir,
            replicate,
            llm_wrapper=llm, 
            critic=potential
        )

    max_inst = args.max_instances if args.max_instances > 0 else None
    
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
    else:
        logger.warning("No results returned from evaluation loop.")

if __name__ == "__main__":
    asyncio.run(main())