import torch
import numpy as np
from tuned_lens import TunedLens
from genlm.backend.llm import load_model_by_name
from genlm.control.potential.llm import PromptedLLM  # Adjusted import path for standard genlm

class TunedLensLLM(PromptedLLM):
    """
    A PromptedLLM that uses a Tuned Lens to project hidden states into vocabulary space.
    """

    def __init__(
        self,
        llm,
        lens,
        target_layer_idx: int,
        prompt_ids=None,
        eos_tokens=None,
        temperature=1.0,
        token_maps=None,
    ):
        super().__init__(
            llm, 
            prompt_ids=prompt_ids, 
            eos_tokens=eos_tokens, 
            temperature=temperature, 
            token_maps=token_maps
        )
        self.target_layer_idx = target_layer_idx
        self.lens = lens

    @classmethod
    def from_name(
        cls,
        name,
        target_layer_idx,
        backend=None,
        eos_tokens=None,
        prompt_ids=None,
        temperature=1.0,
        **kwargs,
    ):
        # Default to vllm if available, else hf
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        
        # Load base model
        model = load_model_by_name(name, backend=backend, **kwargs)

        # Load Tuned Lens (requires model to be available)
        # Note: 'model.model' usually accesses the raw HF model in the backend wrapper
        raw_hf_model = getattr(model, "model", getattr(model, "_model", None))
        if raw_hf_model is None:
             raise ValueError("Could not access raw Hugging Face model for Tuned Lens.")

        lens = TunedLens.from_model_and_pretrained(raw_hf_model)
        
        return cls(
            model,
            lens=lens,
            target_layer_idx=target_layer_idx,
            prompt_ids=prompt_ids, 
            eos_tokens=eos_tokens, 
            temperature=temperature
        )

    async def get_activations(self, context):
        """
        Computes the lens output for the LAST token of the context.
        """
        # 1. Sanitize Context
        safe_context = [t for t in context if isinstance(t, (str, bytes, int))]
        if not safe_context:
            return None

        # 2. Prepare inputs
        if isinstance(safe_context, str):
            safe_context = self.tokenize(safe_context)
            
        try:
            context_ids = self.encode_tokens(safe_context)
        except Exception:
            return None
        
        full_ids = self.prompt_ids + context_ids
        
        # 3. Access Internal Model
        raw_model = getattr(self.model, "model", getattr(self.model, "_model", None))
        
        if raw_model and hasattr(raw_model, "forward"):
            input_tensor = torch.tensor([full_ids], device=self.model.device)
            
            with torch.no_grad():
                outputs = raw_model(
                    input_tensor, 
                    output_hidden_states=True,
                    return_dict=True
                )
            
            try:
                target_layer_output = outputs.hidden_states[self.target_layer_idx]
            except IndexError:
                raise ValueError(f"Layer index {self.target_layer_idx} out of bounds.")
            
            # Apply Lens
            lens_output = self.lens(target_layer_output, self.target_layer_idx)
            
            # Return last token activation [hidden_size]
            return lens_output[0, -1, :]
            
        else:
            return np.zeros(768)

    def spawn(self, prompt_ids=None, eos_tokens=None, temperature=None):
        prompt_ids = prompt_ids if prompt_ids is not None else self.prompt_ids.copy()
        temperature = temperature if temperature is not None else self.temperature
        
        reuse_map = (eos_tokens is None) or (eos_tokens == self.token_maps.eos_tokens)

        if reuse_map:
            return TunedLensLLM(
                self.model,
                lens=self.lens, # Pass the lens reference
                target_layer_idx=self.target_layer_idx,
                prompt_ids=prompt_ids,
                temperature=temperature,
                token_maps=self.token_maps,
            )
        
        return TunedLensLLM(
            self.model,
            lens=self.lens,
            target_layer_idx=self.target_layer_idx,
            prompt_ids=prompt_ids,
            eos_tokens=eos_tokens,
            temperature=temperature,
        )