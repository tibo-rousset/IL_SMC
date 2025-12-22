from genlm.control.potential.base import Potential
from .llms import TunedLensLLM

class ActivationPotential(Potential):
    """
    Filters sequences based on a metric applied to the TunedLens output.
    """
    def __init__(self, model: TunedLensLLM, metric):
        # IMPORTANT: Pass vocab to super() to satisfy Potential contract
        super().__init__(model.vocab)
        self.model = model
        self.metric = metric

    async def _check_score(self, context) -> float:
        # 1. Get activations (async)
        activations = await self.model.get_activations(context)
        
        if activations is None:
            return 0.0

        # Apply the custom metric function
        score = self.metric(activations)
        return score

    # Implement required abstract methods
    async def score(self, context) -> float:
        return await self._check_score(context)

    async def prefix(self, context) -> float:
        return await self._check_score(context)

    async def complete(self, context) -> float:
        return await self._check_score(context)