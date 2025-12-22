from genlm.control.potential.base import Potential
from .llms import TunedLensLLM

class ActivationPotential(Potential):
    """
    Filters sequences based on a metric applied to the TunedLens output.
    """
    def __init__(self, model: TunedLensLLM, metric):
        super().__init__(model.vocab)
        self.model = model
        self.metric = metric

    async def _check_score(self, context) -> float:
        activations = await self.model.get_activations(context)
        
        if activations is None:
            return 0.0

        score = self.metric(activations)
        return score

    async def score(self, context) -> float:
        return await self._check_score(context)

    async def prefix(self, context) -> float:
        return await self._check_score(context)

    async def complete(self, context) -> float:
        return await self._check_score(context)
    
class DualActivationPotential(Potential):
    """
    A generic potential that steers based on the relationship between 
    two sets of activations (e.g., Mid-Layer vs. Final-Layer).
    """
    def __init__(self, model: TunedLensLLM, metric):
        """
        Args:
            model: The TunedLensLLM instance.
            metric: A function `f(logits_mid, logits_final) -> float`.
        """
        super().__init__(model.vocab)
        self.model = model
        self.metric = metric

    async def _check_score(self, context) -> float:
        # 1. Fetch BOTH sets of logits
        logits_mid, logits_final = await self.model.get_dual_activations(context)
        
        if logits_mid is None:
            return 0.0

        # 2. Apply the generic metric function
        score = self.metric(logits_mid, logits_final)
        
        return score

    async def score(self, context) -> float:
        return await self._check_score(context)

    async def prefix(self, context) -> float:
        return await self._check_score(context)

    async def complete(self, context) -> float:
        return await self._check_score(context)