"""
Evidential Deep Learning (EDL) loss functions.

Implements:
    1. EDL Cross-Entropy Loss (Type II MLE): L = sum_k y_k * (digamma(S) - digamma(alpha_k))
    2. KL Divergence Regularization: KL(Dir(alpha) || Dir(1,...,1))
    3. Combined EDL Loss with annealed KL term
"""

import math

import torch
import torch.nn.functional as F


def _parse_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"y", "yes", "true", "1"}:
            return True
        if normalized in {"n", "no", "false", "0"}:
            return False
        raise ValueError(f"{name} must be y/n or true/false.")
    return bool(value)


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
        return value.tolist()
    return float(value)


def _edl_crossentropy_per_sample(alpha, target, num_classes=2):
    y = F.one_hot(target.long(), num_classes=num_classes).float()
    y = y.to(device=alpha.device, dtype=alpha.dtype)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    return torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)


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
        focal_gamma: optional focal modulation; 0 keeps the original behavior
        wrong_evidence_penalty_weight: weight for wrong-direction evidence penalty
        wrong_evidence_margin: margin used by max wrong-class probability violation
        wrong_evidence_class_balanced: average wrong-evidence penalty by class first
        loss_weight_normalization: "legacy_mean" preserves the old weighted mean;
            "weighted_mean" divides by the weighted denominator
    """
    def __init__(
        self,
        num_classes=2,
        kl_weight=1.0,
        annealing_start=0,
        annealing_epochs=10,
        class_weights=None,
        focal_gamma=0.0,
        wrong_evidence_penalty_weight=0.0,
        wrong_evidence_margin=0.05,
        wrong_evidence_class_balanced=True,
        loss_weight_normalization="legacy_mean",
    ):
        super(EDLCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_start = annealing_start
        self.annealing_epochs = annealing_epochs
        self.class_weights = class_weights
        self.focal_gamma = float(focal_gamma or 0.0)
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        self.wrong_evidence_penalty_weight = float(wrong_evidence_penalty_weight or 0.0)
        if self.wrong_evidence_penalty_weight < 0:
            raise ValueError("wrong_evidence_penalty_weight must be non-negative.")
        self.wrong_evidence_margin = float(wrong_evidence_margin)
        if self.wrong_evidence_margin < 0:
            raise ValueError("wrong_evidence_margin must be non-negative.")
        self.wrong_evidence_class_balanced = _parse_bool(
            wrong_evidence_class_balanced,
            "wrong_evidence_class_balanced",
        )
        self.loss_weight_normalization = str(loss_weight_normalization or "legacy_mean").strip().lower()
        if self.loss_weight_normalization not in {"legacy_mean", "weighted_mean"}:
            raise ValueError("loss_weight_normalization must be 'legacy_mean' or 'weighted_mean'.")
        self.last_stats = {}
    
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
        target_indices = target.long().to(alpha.device)
        per_sample_ce = _edl_crossentropy_per_sample(alpha, target_indices, self.num_classes)
        unweighted_ce_loss = per_sample_ce.mean()

        if self.class_weights is None:
            sample_weights = torch.ones_like(per_sample_ce)
        else:
            class_weights = torch.as_tensor(
                self.class_weights,
                dtype=per_sample_ce.dtype,
                device=per_sample_ce.device,
            )
            sample_weights = class_weights[target_indices]

        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        p_true = prob.gather(1, target_indices.view(-1, 1)).squeeze(1).clamp(1e-8, 1.0)
        if self.focal_gamma > 0:
            focal_factor = (1.0 - p_true).pow(self.focal_gamma)
        else:
            focal_factor = torch.ones_like(per_sample_ce)

        weighted_ce = per_sample_ce * sample_weights
        class_weighted_ce_loss = weighted_ce.mean()
        focal_ce_loss = (per_sample_ce * focal_factor).mean()
        focal_weighted_ce = weighted_ce * focal_factor
        if self.loss_weight_normalization == "weighted_mean":
            focal_weighted_denominator = (sample_weights * focal_factor).sum().clamp_min(1e-8)
            ce_loss = focal_weighted_ce.sum() / focal_weighted_denominator
        else:
            focal_weighted_denominator = torch.as_tensor(
                per_sample_ce.numel(),
                dtype=per_sample_ce.dtype,
                device=per_sample_ce.device,
            )
            ce_loss = focal_weighted_ce.mean()
        
        # KL regularization. Keep evidence for the true class, and regularize
        # only non-target evidence toward the uniform Dirichlet prior.
        y = F.one_hot(target_indices, num_classes=self.num_classes).float()
        y = y.to(device=alpha.device, dtype=alpha.dtype)
        alpha_for_kl = y + (1.0 - y) * alpha
        annealing = self.get_annealing_coeff(epoch)
        kl_loss = kl_divergence_dirichlet(alpha_for_kl, self.num_classes).mean()

        evidence = torch.clamp(alpha - 1.0, min=0.0)
        true_class_mask = y.bool()
        wrong_probs = prob.masked_fill(true_class_mask, float("-inf"))
        p_wrong = wrong_probs.max(dim=1).values
        margin_violation = F.relu(p_wrong - p_true + self.wrong_evidence_margin)
        total_evidence = evidence.sum(dim=1)
        per_sample_wrong_evidence_penalty = margin_violation * total_evidence
        if self.wrong_evidence_class_balanced:
            penalty_means = []
            for class_idx in range(self.num_classes):
                class_mask = target_indices == class_idx
                if class_mask.any():
                    penalty_means.append(per_sample_wrong_evidence_penalty[class_mask].mean())
            if penalty_means:
                wrong_evidence_penalty = torch.stack(penalty_means).mean()
            else:
                wrong_evidence_penalty = per_sample_wrong_evidence_penalty.mean()
        else:
            wrong_evidence_penalty = per_sample_wrong_evidence_penalty.mean()
        
        # Combined
        total_loss = ce_loss + self.kl_weight * annealing * kl_loss
        if self.wrong_evidence_penalty_weight > 0:
            total_loss = (
                total_loss
                + self.wrong_evidence_penalty_weight * annealing * wrong_evidence_penalty
            )

        stats = {
            'ce_loss': _safe_float(ce_loss),
            'data_loss': _safe_float(ce_loss),
            'unweighted_ce_loss': _safe_float(unweighted_ce_loss),
            'class_weighted_ce_loss': _safe_float(class_weighted_ce_loss),
            'focal_ce_loss': _safe_float(focal_ce_loss),
            'kl_loss': _safe_float(kl_loss),
            'annealing': float(annealing),
            'wrong_evidence_penalty': _safe_float(wrong_evidence_penalty),
            'margin_violation_mean': _safe_float(margin_violation.mean()),
            'total_evidence_mean': _safe_float(total_evidence.mean()),
            'focal_factor_mean': _safe_float(focal_factor.mean()),
            'sample_weight_mean': _safe_float(sample_weights.mean()),
            'focal_weighted_denominator': _safe_float(focal_weighted_denominator),
            'total_loss': _safe_float(total_loss),
        }

        for class_idx in range(self.num_classes):
            class_mask = target_indices == class_idx
            stats[f'class{class_idx}_n'] = int(class_mask.detach().sum().cpu())
            if class_mask.any():
                stats[f'class{class_idx}_ce_loss_mean'] = _safe_float(per_sample_ce[class_mask].mean())
                stats[f'class{class_idx}_weighted_ce_loss_mean'] = _safe_float(weighted_ce[class_mask].mean())
                stats[f'class{class_idx}_focal_weighted_ce_loss_mean'] = _safe_float(
                    focal_weighted_ce[class_mask].mean()
                )
                stats[f'class{class_idx}_focal_factor_mean'] = _safe_float(focal_factor[class_mask].mean())
                stats[f'class{class_idx}_wrong_evidence_penalty_mean'] = _safe_float(
                    per_sample_wrong_evidence_penalty[class_mask].mean()
                )
            else:
                stats[f'class{class_idx}_ce_loss_mean'] = float("nan")
                stats[f'class{class_idx}_weighted_ce_loss_mean'] = float("nan")
                stats[f'class{class_idx}_focal_weighted_ce_loss_mean'] = float("nan")
                stats[f'class{class_idx}_focal_factor_mean'] = float("nan")
                stats[f'class{class_idx}_wrong_evidence_penalty_mean'] = float("nan")

        self.last_stats = stats

        return total_loss, stats
