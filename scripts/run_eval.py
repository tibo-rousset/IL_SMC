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

# Import your custom modules
from genlm_project.metrics import *
from genlm_project import (
    TunedLensLLM, 
    ActivationPotential, 
    TruthfulQADataset, 
    TruthfulQAEvaluator, 
    truthful_qa_prompt_formatter
)

# Setup Logger
logger = logging.getLogger("eval_script")

def parse_args():
    parser = argparse.ArgumentParser(description="Run TruthfulQA Evaluation")
    
    # Model Args
    parser.add_argument("--model_name", type=str, default="gpt2", help="HF Model name")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Temperature")
    
    # Generation Args
    parser.add_argument("--max_tokens", type=int, default=30, help="Max tokens")
    parser.add_argument("--particles", type=int, default=5, help="SMC particles")
    parser.add_argument("--weight", type=float, default=1.0, help="Potential weight")
    
    # Eval Args
    parser.add_argument("--max_instances", type=int, default=5, help="Num instances (0=all)")
    parser.add_argument("--output_dir", type=str, default="truthfulqa_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    return parser.parse_args()

def save_summary_csv(results_list, model_name, output_dir):
    """
    Aggregates evaluation results and saves a summary CSV matching 
    the official TruthfulQA format.
    """
    logger.info("Aggregating results into summary CSV...")
    
    data = []
    for res in results_list:
        row = {'Model': model_name}
        
        metrics_dict = getattr(res["results"], 'metadata', getattr(res["results"], 'metrics', {}))
        
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

    target_metrics = [
        'MC1', 'MC2',
        'bleu acc',
        'rouge1 acc',
        'bleurt acc', 
        'BLEURT acc',
        'GPT-judge acc',
        'GPT-info acc'
    ]
    
    final_df = results_long[results_long['Metric'].isin(target_metrics)]

    if final_df.empty:
        logger.warning("No matching metrics found. Saving all computed metrics instead.")
        final_df = results_long

    summary_pivot = pd.pivot_table(final_df, values='Value', index='Model', columns='Metric')
    
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
    sampler = DirectTokenSampler(current_llm)
    
    # 4. Run SMC Sampling
    sequences = await sampler.smc(
        n_particles=args.particles,
        max_tokens=args.max_tokens,
        verbosity=0,
        ess_threshold=0.5,
        critic=critic 
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
        temperature=args.temperature
    )

    metric_fn = entropy_score
    potential = ActivationPotential(model=llm, metric=metric_fn)
    logger.info("Potential initialized.")

    logger.info("Initializing Dataset & Evaluator...")
    dataset = TruthfulQADataset(split="validation")
    evaluator = TruthfulQAEvaluator()

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