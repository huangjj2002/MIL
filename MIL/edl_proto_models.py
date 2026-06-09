

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from MIL.edl_models import MIL_EDL_Wrapper


def _inverse_softplus(value):
    value = float(value)
    if value > 20.0:
        return value
    return math.log(math.expm1(max(value, 1e-6)))


class PrototypeEDLHead(nn.Module):
 

    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        normalize=True,
        gamma_init=1.0,
        dropout=0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeEDLHead currently expects binary EDL output.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.drop = nn.Dropout(p=dropout)

        self.prototypes = nn.Parameter(
            torch.empty(self.num_classes, self.prototypes_per_class, self.in_features)
        )
        self.proto_strength = nn.Parameter(
            torch.full(
                (self.num_classes, self.prototypes_per_class),
                _inverse_softplus(1.0),
            )
        )
        self.raw_gamma = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init)))
        self.reset_parameters()

    def reset_parameters(self):
        flat_prototypes = self.prototypes.view(-1, self.in_features)
        nn.init.xavier_uniform_(flat_prototypes)

    def forward(self, x):
        x = self.drop(x)
        if self.normalize:
            x_for_distance = F.normalize(x, dim=-1)
            prototypes = F.normalize(self.prototypes, dim=-1)
        else:
            x_for_distance = x
            prototypes = self.prototypes

        distances = (
            x_for_distance[:, None, None, :] - prototypes[None, :, :, :]
        ).pow(2).sum(dim=-1)
        gamma = F.softplus(self.raw_gamma) + 1e-6
        similarity = torch.exp(-gamma * distances)
        strengths = F.softplus(self.proto_strength) + 1e-6
        prototype_evidence = similarity * strengths.unsqueeze(0)

        evidence = prototype_evidence.sum(dim=-1)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = alpha.shape[-1] / S.squeeze(-1)

        out = {
            "evidence": evidence,
            "alpha": alpha,
            "S": S,
            "prob": prob,
            "uncertainty": uncertainty,
            "prototype_distances": distances,
            "prototype_similarity": similarity,
            "prototype_evidence": prototype_evidence,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_evidence, top_idx = torch.topk(prototype_evidence, k=topk, dim=-1)
            top_similarity = torch.gather(similarity, dim=-1, index=top_idx)
            top_distances = torch.gather(distances, dim=-1, index=top_idx)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_evidence,
                    "topk_proto_similarity": top_similarity,
                    "topk_proto_distances": top_distances,
                }
            )

        return out

    def initialize_from_embeddings(self, embeddings, labels, random_state=0):

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().float().numpy()
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels).astype(int)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_features:
            raise ValueError(
                f"Expected embeddings with shape (N, {self.in_features}), "
                f"got {embeddings.shape}."
            )
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must contain the same number of rows.")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot initialize prototypes from an empty embedding set.")

        working_embeddings = embeddings
        if self.normalize:
            norms = np.linalg.norm(working_embeddings, axis=1, keepdims=True)
            working_embeddings = working_embeddings / np.clip(norms, 1e-12, None)

        global_center = working_embeddings.mean(axis=0, keepdims=True)
        centers_by_class = []
        warnings = []

        for class_idx in range(self.num_classes):
            class_embeddings = working_embeddings[labels == class_idx]
            if len(class_embeddings) == 0:
                centers = np.repeat(global_center, self.prototypes_per_class, axis=0)
                warnings.append(
                    f"class {class_idx} has no samples; using global mean prototypes"
                )
            elif len(class_embeddings) >= self.prototypes_per_class:
                kmeans = KMeans(
                    n_clusters=self.prototypes_per_class,
                    n_init=10,
                    random_state=random_state + class_idx,
                )
                centers = kmeans.fit(class_embeddings).cluster_centers_.astype(np.float32)
            else:
                centers = class_embeddings.astype(np.float32)
                repeat_idx = 0
                while len(centers) < self.prototypes_per_class:
                    centers = np.concatenate(
                        [centers, class_embeddings[[repeat_idx % len(class_embeddings)]]],
                        axis=0,
                    )
                    repeat_idx += 1
                warnings.append(
                    f"class {class_idx} has fewer samples than prototypes; repeated centers"
                )

            centers_by_class.append(centers[: self.prototypes_per_class])

        centers = np.stack(centers_by_class, axis=0).astype(np.float32)
        with torch.no_grad():
            self.prototypes.copy_(
                torch.as_tensor(centers, dtype=self.prototypes.dtype, device=self.prototypes.device)
            )

        return warnings


class MIL_EDL_Prototype_Wrapper(MIL_EDL_Wrapper):


    def __init__(
        self,
        mil_model,
        edl_dropout=0.0,
        proto_k=4,
        proto_topk=3,
        proto_normalize=True,
        proto_gamma_init=1.0,
    ):
        super().__init__(mil_model, edl_dropout=edl_dropout)

        self.proto_k = int(proto_k)
        self.proto_topk = int(proto_topk)
        self.proto_normalize = bool(proto_normalize)
        self.proto_gamma_init = float(proto_gamma_init)

        if hasattr(mil_model, "classifier"):
            in_features = self._get_classifier_in_features(mil_model.classifier)
            self.edl_head = self._make_proto_head(in_features, edl_dropout)

        self.edl_side_heads = nn.ModuleDict()
        if hasattr(mil_model, "side_classifiers"):
            for key, clf in mil_model.side_classifiers.items():
                in_features = self._get_classifier_in_features(clf)
                self.edl_side_heads[key] = self._make_proto_head(in_features, edl_dropout)

    def _make_proto_head(self, in_features, edl_dropout):
        return PrototypeEDLHead(
            in_features=in_features,
            num_classes=2,
            prototypes_per_class=self.proto_k,
            topk=self.proto_topk,
            normalize=self.proto_normalize,
            gamma_init=self.proto_gamma_init,
            dropout=edl_dropout,
        )

    def prototype_heads(self):
        heads = {}
        if hasattr(self, "edl_head") and isinstance(self.edl_head, PrototypeEDLHead):
            heads["edl_head"] = self.edl_head
        for name, head in self.edl_side_heads.items():
            if isinstance(head, PrototypeEDLHead):
                heads[f"edl_side_heads.{name}"] = head
        return heads
