import torch
from genlm.control.sampler import DirectTokenSampler

class MonitoredDirectTokenSampler(DirectTokenSampler):
    """
    A subclass of DirectTokenSampler that tracks the Effective Sample Size (ESS)
    of the batch during generation without modifying the original sampling logic.
    """
    def __init__(self, potential):
        super().__init__(potential)
        self.running_log_weights = None
        self.ess_history = []

    def _compute_ess(self, log_weights):
        """Computes ESS in log-space."""
        log_sum_w = torch.logsumexp(log_weights, dim=0)
        log_sum_w_sq = torch.logsumexp(2 * log_weights, dim=0)
        
        log_ess = 2 * log_sum_w - log_sum_w_sq
        return torch.exp(log_ess).item()

    async def sample(self, context, draw=None):
        token, step_weight, logp = await super().sample(context, draw)

        if self.running_log_weights is None:
            if hasattr(step_weight, 'shape') and len(step_weight.shape) > 0:
                self.running_log_weights = torch.zeros_like(step_weight)
            else:
                self.running_log_weights = torch.tensor(0.0)

        self.running_log_weights += step_weight

        current_ess = self._compute_ess(self.running_log_weights)
        self.ess_history.append(current_ess)

        return token, step_weight, logp

    async def cleanup(self):
        """Resets tracking state and calls parent cleanup."""
        self.running_log_weights = None
        self.ess_history = []
        await super().cleanup()