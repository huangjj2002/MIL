import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dst_activation_init_gamma(gamma_init):
    return math.sqrt(max(float(gamma_init), 1e-6))


def dst_activation_init_alpha(alpha_init):
    return float(alpha_init)


def pignistic(mass, num_classes):
    class_mass = mass[..., :num_classes]
    omega = mass[..., num_classes]
    prob = class_mass + omega.unsqueeze(-1) / float(num_classes)
    return prob, omega


class PrototypeLayer(nn.Module):
    def __init__(self, n_prototypes, n_features):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_prototypes, n_features))

    def forward(self, x):
        return torch.cdist(x, self.w, p=2).pow(2)


class DistanceActivationLayer(nn.Module):
    def __init__(self, n_prototypes, init_alpha=0.0, init_gamma=1.0):
        super().__init__()
        self.xi = nn.Parameter(torch.full((n_prototypes,), float(init_alpha)))
        self.eta = nn.Parameter(torch.full((n_prototypes,), float(init_gamma)))

    def forward(self, distances):
        reliability = torch.sigmoid(self.xi).unsqueeze(0)
        gamma = self.eta.pow(2).unsqueeze(0).clamp_min(1e-8)
        return reliability * torch.exp(-gamma * distances)


class DempsterShaferModule(nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes, eps=1e-8):
        super().__init__()
        if n_prototypes % n_classes != 0:
            raise ValueError("n_prototypes must be divisible by n_classes.")
        self.n_feature_maps = int(n_feature_maps)
        self.n_classes = int(n_classes)
        self.n_prototypes = int(n_prototypes)
        self.prototypes_per_class = self.n_prototypes // self.n_classes
        self.eps = float(eps)

        self.ds1 = PrototypeLayer(self.n_prototypes, self.n_feature_maps)
        self.ds1_activate = DistanceActivationLayer(self.n_prototypes)

        class_ids = torch.arange(self.n_prototypes) // self.prototypes_per_class
        self.register_buffer("prototype_classes", class_ids.long())

    def ds2(self, activations):
        mass = activations.new_zeros(
            activations.size(0),
            self.n_prototypes,
            self.n_classes,
        )
        class_index = self.prototype_classes.view(1, -1, 1).expand(
            activations.size(0),
            -1,
            1,
        )
        mass.scatter_(2, class_index, activations.unsqueeze(-1))
        return mass

    def ds2_omega(self, mass_prototypes):
        assigned_mass = mass_prototypes.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        omega = (1.0 - assigned_mass).clamp_min(self.eps)
        return torch.cat([mass_prototypes, omega], dim=-1)

    def _combine_pair(self, current, incoming):
        current_class = current[:, : self.n_classes]
        incoming_class = incoming[:, : self.n_classes]
        current_omega = current[:, self.n_classes : self.n_classes + 1]
        incoming_omega = incoming[:, self.n_classes : self.n_classes + 1]

        same_class = current_class * incoming_class
        incoming_unknown = current_class * incoming_omega
        current_unknown = incoming_class * current_omega
        class_mass = same_class + incoming_unknown + current_unknown
        omega = current_omega * incoming_omega

        conflict = (
            current_class.sum(dim=1, keepdim=True) * incoming_class.sum(dim=1, keepdim=True)
            - same_class.sum(dim=1, keepdim=True)
        )
        normalizer = (1.0 - conflict).clamp_min(self.eps)
        return torch.cat([class_mass, omega], dim=-1) / normalizer

    def ds3_dempster(self, mass_prototypes_omega):
        combined = mass_prototypes_omega[:, 0, :]
        for proto_idx in range(1, mass_prototypes_omega.size(1)):
            combined = self._combine_pair(combined, mass_prototypes_omega[:, proto_idx, :])
        return combined

    def ds3_normalize(self, mass):
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, x, normalize=False):
        prototypes = self.ds1.w
        x_for_distance = x
        if normalize:
            x_for_distance = F.normalize(x_for_distance, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
            distances = (x_for_distance[:, None, :] - prototypes[None, :, :]).pow(2).sum(dim=-1)
        else:
            distances = self.ds1(x_for_distance)
        activations = self.ds1_activate(distances)
        mass_prototypes = self.ds2(activations)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_dempster = self.ds3_dempster(mass_prototypes_omega)
        mass = self.ds3_normalize(mass_dempster)
        return mass, distances, activations, mass_prototypes


DistanceActivation_layer = DistanceActivationLayer
Dempster_Shafer_Module = DempsterShaferModule
