import asyncio
import logging
import torch
from genlm.control import direct_token_sampler
from genlm.eval import ModelOutput, ModelResponse, run_evaluation

from genlm_project import (
    TunedLensLLM, 
    ActivationPotential, 
    TruthfulQADataset, 
    TruthfulQAEvaluator, 
    truthful_qa_prompt_formatter
)

# --- CONFIGURATION ---
MODEL_NAME = "gpt2" # Ensure a tuned lens exists for this model
LAYER_IDX = -1
MAX_TOKENS = 30
PARTICLES = 1

# Define a simple metric function for the Potential
def my_metric(activations):
    """Example: favor activations with higher norm (just for testing)"""
    if isinstance(activations, torch.Tensor):
        return torch.norm(activations).item() * 0.1
    return 0.0

async def inference_fn(instance, output_dir, replicate, llm_wrapper, potential):
    # 1. Format Prompt
    raw_ids = truthful_qa_prompt_formatter(
        llm_wrapper.model.tokenizer, instance, use_chat_format=False
    )
    if hasattr(raw_ids, "tolist"): raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list): raw_ids = raw_ids[0]

    # 2. Spawn Model
    current_llm = llm_wrapper.spawn(prompt_ids=raw_ids)

    # 3. Create Sampler with POTENTIAL
    # Note: Depending on your GenLM version, you might pass potential to sampler or .smc()
    sampler = direct_token_sampler(current_llm, potential=potential)
    
    sequences = await sampler.smc(
        n_particles=PARTICLES,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        ess_threshold=0.5
    )

    # 4. Decode
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
    print(f"Loading TunedLensLLM: {MODEL_NAME}...")
    llm = TunedLensLLM.from_name(
        MODEL_NAME, 
        backend="hf", 
        target_layer_idx=LAYER_IDX,
        temperature=0.0001
    )

    # Initialize Potential with the Model and Metric
    potential = ActivationPotential(model=llm, metric=my_metric)

    print("Initializing Dataset & Evaluator...")
    dataset = TruthfulQADataset(split="validation")
    evaluator = TruthfulQAEvaluator()

    async def bound_model_fn(inst, out, rep):
        return await inference_fn(inst, out, rep, llm_wrapper=llm, potential=potential)

    print("Starting Evaluation Loop...")
    await run_evaluation(
        dataset=dataset,
        model=bound_model_fn,
        evaluator=evaluator,
        output_dir="truthfulqa_results",
        overwrite_results=True,
        overwrite_outputs=True,
        verbosity=1,
        max_instances=5,
    )

if __name__ == "__main__":
    asyncio.run(main())