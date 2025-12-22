import torch
from genlm.control.potential.llm import PromptedLLM

class RepresentationPromptedLLM(PromptedLLM):
    """
    A PromptedLLM that also exposes internal hidden states at a specific layer.
    """
    def __init__(
        self,
        llm,
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

    async def get_activations(self, context):
        """
        Computes the activation vector for the LAST token of the context.
        """
        if isinstance(context, str):
            context = self.tokenize(context)
            
        context_ids = self.encode_tokens(context)
        full_ids = self.prompt_ids + context_ids
        
        # Access Internal Model
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
            
            # Return last token activation
            return target_layer_output[0, -1, :]
        else:
            raise NotImplementedError("Could not find raw HuggingFace model.")

    def spawn(self, prompt_ids=None, eos_tokens=None, temperature=None):
        """Preserves target_layer_idx during spawning."""
        prompt_ids = prompt_ids if prompt_ids is not None else self.prompt_ids.copy()
        temperature = temperature if temperature is not None else self.temperature
        
        reuse_map = (eos_tokens is None) or (eos_tokens == self.token_maps.eos_tokens)

        if reuse_map:
            return RepresentationPromptedLLM(
                self.model,
                target_layer_idx=self.target_layer_idx,
                prompt_ids=prompt_ids,
                temperature=temperature,
                token_maps=self.token_maps,
            )
        
        return RepresentationPromptedLLM(
            self.model,
            target_layer_idx=self.target_layer_idx,
            prompt_ids=prompt_ids,
            eos_tokens=eos_tokens,
            temperature=temperature,
        )