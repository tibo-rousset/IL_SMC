import asyncio
import argparse
import logging
import torch
import sys
from genlm.control.sampler import DirectTokenSampler
from genlm.eval import ModelOutput, ModelResponse, run_evaluation

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
    parser = argparse.ArgumentParser(description="Run TruthfulQA Evaluation with Custom Potentials")
    
    # Model Args
    parser.add_argument("--model_name", type=str, default="gpt2", help="HF Model name (must have tuned-lens available)")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index to extract activations from")
    parser.add_argument("--temperature", type=float, default=0.0001, help="Sampling temperature")
    
    # Generation Args
    parser.add_argument("--max_tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--particles", type=int, default=5, help="Number of SMC particles")
    
    # Potential Args
    parser.add_argument("--weight", type=float, default=1.0, help="Weight scaling factor for the potential score")
    
    # Eval Args
    parser.add_argument("--max_instances", type=int, default=5, help="Number of examples to evaluate (set 0 for all)")
    parser.add_argument("--output_dir", type=str, default="truthfulqa_results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()

def create_metric_fn(weight):
    """
    Creates a metric function with closure over the weight and logger.
    """
    def my_metric(activations):
        """Example: favor activations with higher norm"""
        score = 0.0
        if isinstance(activations, torch.Tensor):
            raw_val = torch.norm(activations).item()
            score = raw_val * weight
            
            logger.debug(f"Potential Activation | Norm: {raw_val:.4f} | Weighted Score: {score:.4f}")
            
        return score
    return my_metric

async def inference_fn(instance, args, replicate, llm_wrapper, critic=None):
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
    
    logger.info(f"Starting Evaluation with args: {args}")

    logger.info(f"Loading TunedLensLLM: {args.model_name}...")
    llm = TunedLensLLM.from_name(
        args.model_name, 
        backend="hf", 
        target_layer_idx=args.layer_idx,
        temperature=args.temperature
    )

    metric_fn = create_metric_fn(args.weight)
    potential = ActivationPotential(model=llm, metric=metric_fn)
    logger.info("Potential initialized and linked to TunedLensLLM.")

    logger.info("Initializing Dataset & Evaluator...")
    dataset = TruthfulQADataset(split="validation")
    evaluator = TruthfulQAEvaluator()

    async def bound_model_fn(instance, output_dir, replicate):
        return await inference_fn(
            instance, 
            args,
            replicate, 
            llm_wrapper=llm, 
            critic=potential
        )

    max_inst = args.max_instances if args.max_instances > 0 else None
    logger.info(f"Starting Evaluation Loop for {max_inst if max_inst else 'ALL'} instances...")
    
    await run_evaluation(
        dataset=dataset,
        model=bound_model_fn,
        evaluator=evaluator,
        output_dir=args.output_dir,
        overwrite_results=True,
        overwrite_outputs=True,
        verbosity=1,
        max_instances=max_inst,
    )

if __name__ == "__main__":
    asyncio.run(main())