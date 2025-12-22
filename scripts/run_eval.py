import asyncio
import logging
from genlm.control import direct_token_sampler
from genlm.eval import ModelOutput, ModelResponse, run_evaluation
from genlm.backend.llm import load_model_by_name

# Import from your new package
from genlm_project.llms import RepresentationPromptedLLM
from genlm_project.data import TruthfulQADataset
from genlm_project.metrics import TruthfulQAEvaluator
from genlm_project.utils import truthful_qa_prompt_formatter

# Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- CONFIGURATION ---
MODEL_NAME = "openai-community/gpt2"
BACKEND = "hf"  # Must be 'hf' for your custom class
LAYER_IDX = -1
MAX_TOKENS = 30
PARTICLES = 1

async def inference_fn(instance, output_dir, replicate, llm_wrapper):
    """
    The main inference logic used by genlm.eval
    """
    # 1. Format Prompt
    raw_ids = truthful_qa_prompt_formatter(
        llm_wrapper.model.tokenizer, instance, use_chat_format=False
    )
    
    # Handle list wrapping issues common in tokenizers
    if hasattr(raw_ids, "tolist"): raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list): raw_ids = raw_ids[0]

    # 2. Spawn Model for this specific prompt
    # Note: spawn() handles prompt_ids assignment
    current_llm = llm_wrapper.spawn(prompt_ids=raw_ids)

    # 3. Create Sampler (SMC, Beam, or Direct)
    sampler = direct_token_sampler(current_llm)
    
    # 4. Run Generation
    sequences = await sampler.smc(
        n_particles=PARTICLES,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        ess_threshold=0.5
    )

    # 5. Decode Responses
    candidates = sequences.decoded_posterior
    prompt_text = current_llm.model.tokenizer.decode(raw_ids, skip_special_tokens=True)
    responses = []

    # If decoding didn't happen automatically in sampler, do it manually (fallback)
    if not candidates:
        for seq, weight in zip(sequences.contexts, sequences.normalized_weights):
            full_text = b"".join([b for b in seq if isinstance(b, bytes)]).decode("utf-8", errors="ignore")
            # Strip prompt
            gen_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            candidates[gen_text] = candidates.get(gen_text, 0.0) + weight

    # 6. Clean Text
    for sequence, prob in candidates.items():
        clean_resp = sequence.strip().split("\n\n")[0].split("\nQ:")[0].strip()
        if not clean_resp:
            clean_resp = "I have no comment."
        responses.append(ModelResponse(response=clean_resp, weight=prob))

    return ModelOutput(responses=responses)

async def main():
    print(f"Loading Model: {MODEL_NAME}...")
    
    # Initialize your custom LLM
    # We load via the class method to ensure backend setup is correct
    llm = RepresentationPromptedLLM.from_name(
        MODEL_NAME, 
        backend=BACKEND, 
        target_layer_idx=LAYER_IDX,
        temperature=0.0001
    )

    print("Initializing Dataset & Evaluator...")
    dataset = TruthfulQADataset(split="validation")
    evaluator = TruthfulQAEvaluator()

    # Create a partial function to pass the loaded LLM to the inference loop
    async def bound_model_fn(inst, out, rep):
        return await inference_fn(inst, out, rep, llm_wrapper=llm)

    print("Starting Evaluation Loop...")
    results = await run_evaluation(
        dataset=dataset,
        model=bound_model_fn,
        evaluator=evaluator,
        output_dir="truthfulqa_results",
        overwrite_results=True,
        overwrite_outputs=True,
        verbosity=1,
        max_instances=10, # Set to None for full run
    )
    print("Evaluation Complete.")

if __name__ == "__main__":
    asyncio.run(main())