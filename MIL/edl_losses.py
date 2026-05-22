"""
Evidential Deep Learning (EDL) loss functions.

Implements:
    1. EDL Cross-Entropy Loss (Type II MLE): L = sum_k y_k * (digamma(S) - digamma(alpha_k))
    2. KL Divergence Regularization: KL(Dir(alpha) || Dir(1,...,1))
    3. Combined EDL Loss with annealed KL term
"""

import torch
import torch.nn.functional as F
import math


def edl_crossentropy_loss(alpha, target, num_classes=2, class_weights=None):
    """
    EDL Cross-Entropy Loss (Type II Maximum Likelihood Estimation).
    
    L = sum_k y_k * (digamma(S) - digamma(alpha_k))
    
    Args:
        alpha: (B, K) Dirichlet parameters
        target: (B,) integer class labels (0 or 1 for binary)
        num_classes: number of classes (default 2)
        class_weights: optional tensor/list with one weight per class
    
    Returns:
        scalar loss value
    """
    # One-hot encode target
    y = F.one_hot(target.long(), num_classes=num_classes).float()  # (B, K)
    
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)
    
    # digamma(S) - digamma(alpha_k) for each class k
    loss = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)  # (B,)
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=loss.dtype, device=loss.device)
        sample_weights = class_weights[target.long()]
        loss = loss * sample_weights
    
    return loss.mean()


def kl_divergence_dirichlet(alpha, num_classes=2):
    """
    KL divergence between Dir(alpha) and Dir(1, 1, ..., 1) (uniform Dirichlet).
    
    This regularization term penalizes evidence when predictions are incorrect,
    encouraging the model to "reduce" evidence for wrong classes.
    
    KL(Dir(alpha) || Dir(1,...,1)) = 
        log_gamma(sum(alpha)) - sum(log_gamma(alpha_k))
        - log_gamma(K) + K * log_gamma(1)
        + sum((alpha_k - 1) * (digamma(alpha_k) - digamma(sum(alpha))))
    
    Args:
        alpha: (B, K) Dirichlet parameters (alpha >= 1)
        num_classes: K
    
    Returns:
        scalar KL divergence value
    """
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)
    
    # log_gamma(S) - sum(log_gamma(alpha_k))
    term1 = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    
    # sum((alpha_k - 1) * (digamma(alpha_k) - digamma(S)))
    term2 = torch.sum(
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S)),
        dim=-1, keepdim=True
    )
    
    # -log_gamma(K) + K * log_gamma(1) = -log_gamma(K) + 0 = -log_gamma(K)
    # This is a constant, but we include it for correctness
    term3 = -math.lgamma(num_classes)
    
    kl = term1 + term2 + term3  # (B, 1)
    
    return kl.squeeze(-1)  # (B,)


def edl_mse_loss(alpha, target, num_classes=2):
    """
    EDL Mean Squared Error Loss (alternative to cross-entropy).
    
    L = sum_k (y_k - alpha_k / S)^2 + sum_k (alpha_k * (S - alpha_k)) / (S^2 * (S + 1))
    
    Args:
        alpha: (B, K) Dirichlet parameters
        target: (B,) integer class labels
        num_classes: number of classes
    
    Returns:
        scalar loss value
    """
    y = F.one_hot(target.long(), num_classes=num_classes).float()
    S = torch.sum(alpha, dim=-1, keepdim=True)
    prob = alpha / S
    
    # Data fit term
    err = torch.sum((y - prob) ** 2, dim=-1)
    
    # Variance term (Dirichlet variance)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)),
        dim=-1
    )
    
    return (err + var).mean()


class EDLCombinedLoss(torch.nn.Module):
    """
    Combined EDL Loss: EDL Cross-Entropy + annealed KL regularization.
    
    L_total = L_edl_ce + lambda_kl * annealing_coeff * KL
    
    The annealing coefficient increases from 0 to 1 over training epochs,
    gradually introducing the KL regularization.
    
    Args:
        num_classes: number of classes (default 2 for binary)
        kl_weight: weight for KL divergence term (default 1.0)
        annealing_start: epoch to start annealing (default 0)
        annealing_epochs: number of epochs for full annealing (default 10)
        class_weights: optional class weights for the EDL cross-entropy term
    """
    def __init__(self, num_classes=2, kl_weight=1.0, annealing_start=0, annealing_epochs=10, class_weights=None):
        super(EDLCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_start = annealing_start
        self.annealing_epochs = annealing_epochs
        self.class_weights = class_weights
    
    def get_annealing_coeff(self, epoch):
        """Get current annealing coefficient (0 to 1)."""
        if epoch < self.annealing_start:
            return 0.0
        coeff = (epoch - self.annealing_start) / max(1, self.annealing_epochs)
        return min(1.0, coeff)
    
    def forward(self, alpha, target, epoch=0):
        """
        Args:
            alpha: (B, K) Dirichlet parameters
            target: (B,) integer class labels
            epoch: current epoch number for annealing
        
        Returns:
            loss: scalar combined loss
            dict: individual loss components for logging
        """
        # EDL cross-entropy loss
        ce_loss = edl_crossentropy_loss(alpha, target, self.num_classes, self.class_weights)
        
        # KL regularization. Keep evidence for the true class, and regularize
        # only non-target evidence toward the uniform Dirichlet prior.
        y = F.one_hot(target.long(), num_classes=self.num_classes).float()
        alpha_for_kl = y + (1.0 - y) * alpha
        annealing = self.get_annealing_coeff(epoch)
        kl_loss = kl_divergence_dirichlet(alpha_for_kl, self.num_classes).mean()
        
        # Combined
        total_loss = ce_loss + self.kl_weight * annealing * kl_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'annealing': annealing,
            'total_loss': total_loss.item(),
        }
