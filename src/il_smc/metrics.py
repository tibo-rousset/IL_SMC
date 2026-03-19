import torch
import torch.nn.functional as F
import logging

# Setup module-level logger
logger = logging.getLogger(__name__)

def entropy_score(logits: torch.Tensor) -> float:
    """
    Calculates the entropy of the distribution defined by logits.
    Formula: H(P) = - sum(p_i * log(p_i))
    
    Args:
        logits: Tensor of shape (batch, vocab_size) or (vocab_size,)
    
    Returns:
        The entropy value (float).
    """
    
    log_probs = F.log_softmax(logits, dim=-1)       # log p_i
    probs = torch.exp(log_probs)                    # p_i
    
    entropy = -(probs * log_probs).sum(dim=-1)      # shape: (batch,) or scalar

    val = entropy.mean().item() if entropy.ndim > 0 else entropy.item()

    logger.debug(f"Potential Activation | Entropy: {val:.4f}")

    return val

def kl_divergence_score(logits_mid: torch.Tensor, logits_final: torch.Tensor) -> float:
    """
    Calculates KL(P_mid || P_final).
    Returns the raw positive KL value. 
    """
    p_probs = F.softmax(logits_mid, dim=-1)
    
    q_log_probs = F.log_softmax(logits_final, dim=-1)

    kl_val = F.kl_div(q_log_probs, p_probs, reduction='batchmean')
    val = kl_val.item()
    
    # Debug log
    logger.debug(f"Dual Activation | KL: {val:.4f}")
    
    return val