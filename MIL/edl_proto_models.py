import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from MIL.dst_layers import (
    DempsterShaferModule,
    DistanceActivationLayer,
    dst_activation_init_alpha,
    dst_activation_init_gamma,
    pignistic,
)
from MIL.edl_models import MIL_EDL_Wrapper


class PrototypeEDLHead(nn.Module):
    """Prototype head built on Dempster-Shafer evidence combination."""

    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        normalize=True,
        gamma_init=1.0,
        alpha_init=0.0,
        dropout=0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeEDLHead currently expects binary output.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.n_prototypes = self.num_classes * self.prototypes_per_class
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.drop = nn.Dropout(p=dropout)

        self.ds_module = DempsterShaferModule(
            n_feature_maps=self.in_features,
            n_classes=self.num_classes,
            n_prototypes=self.n_prototypes,
        )
        self.ds_module.ds1_activate = DistanceActivationLayer(
            n_prototypes=self.n_prototypes,
            init_alpha=dst_activation_init_alpha(alpha_init),
            init_gamma=dst_activation_init_gamma(gamma_init),
        )
        self.reset_parameters()

    @property
    def prototypes(self):
        return self.ds_module.ds1.w.view(
            self.num_classes,
            self.prototypes_per_class,
            self.in_features,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ds_module.ds1.w)

    def set_prototypes_from_embeddings(self, prototypes):
        if isinstance(prototypes, torch.Tensor):
            prototype_array = prototypes.detach().cpu().float().numpy()
        else:
            prototype_array = np.asarray(prototypes, dtype=np.float32)

        expected_shape = (self.num_classes, self.prototypes_per_class, self.in_features)
        if prototype_array.shape == (self.n_prototypes, self.in_features):
            prototype_array = prototype_array.reshape(expected_shape)
        if prototype_array.shape != expected_shape:
            raise ValueError(
                f"Expected prototypes with shape {expected_shape} or "
                f"({self.n_prototypes}, {self.in_features}), got {prototype_array.shape}."
            )

        with torch.no_grad():
            self.ds_module.ds1.w.copy_(
                torch.as_tensor(
                    prototype_array.reshape(self.n_prototypes, self.in_features),
                    dtype=self.ds_module.ds1.w.dtype,
                    device=self.ds_module.ds1.w.device,
                )
            )

    def forward(self, x):
        x = self.drop(x.float())
        mass, distances, similarity, mass_prototypes = self.ds_module(
            x,
            normalize=self.normalize,
        )
        prob, uncertainty = pignistic(mass, self.num_classes)

        prototype_mass = torch.zeros(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
            device=x.device,
            dtype=x.dtype,
        )
        for class_idx in range(self.num_classes):
            start = class_idx * self.prototypes_per_class
            end = start + self.prototypes_per_class
            prototype_mass[:, class_idx, :] = mass_prototypes[:, start:end, class_idx]

        distances_by_class = distances.view(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
        )
        similarity_by_class = similarity.view(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
        )

        out = {
            "prob": prob,
            "uncertainty": uncertainty,
            "dst_mass": mass,
            "prototype_distances": distances_by_class,
            "prototype_similarity": similarity_by_class,
            "prototype_evidence": prototype_mass,
            "prototype_mass": prototype_mass,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_mass, top_idx = torch.topk(prototype_mass, k=topk, dim=-1)
            top_similarity = torch.gather(similarity_by_class, dim=-1, index=top_idx)
            top_distances = torch.gather(distances_by_class, dim=-1, index=top_idx)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_mass,
                    "topk_proto_mass": top_mass,
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
        flat_centers = centers.reshape(self.n_prototypes, self.in_features)
        with torch.no_grad():
            self.ds_module.ds1.w.copy_(
                torch.as_tensor(
                    flat_centers,
                    dtype=self.ds_module.ds1.w.dtype,
                    device=self.ds_module.ds1.w.device,
                )
            )

        return warnings


class BagEmbeddingPrototypeDSTModel(nn.Module):
    def __init__(
        self,
        in_features,
        edl_dropout=0.0,
        proto_k=4,
        proto_topk=3,
        proto_normalize=True,
        proto_gamma_init=1.0,
        proto_alpha_init=0.0,
    ):
        super().__init__()
        self.is_training = True
        self.edl_head = PrototypeEDLHead(
            in_features=in_features,
            num_classes=2,
            prototypes_per_class=proto_k,
            topk=proto_topk,
            normalize=proto_normalize,
            gamma_init=proto_gamma_init,
            alpha_init=proto_alpha_init,
            dropout=edl_dropout,
        )

    def forward(self, x, bag_mask=None):
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        out = self.edl_head(x)
        out["type"] = "bag_embedding"
        return out

    def prototype_heads(self):
        return {"edl_head": self.edl_head}


class MIL_EDL_Prototype_Wrapper(MIL_EDL_Wrapper):
    def __init__(
        self,
        mil_model,
        edl_dropout=0.0,
        proto_k=4,
        proto_topk=3,
        proto_normalize=True,
        proto_gamma_init=1.0,
        proto_alpha_init=0.0,
    ):
        super().__init__(
            mil_model,
            edl_dropout=edl_dropout,
            dst_k=proto_k,
            dst_topk=proto_topk,
            dst_normalize=proto_normalize,
            dst_gamma_init=proto_gamma_init,
            dst_alpha_init=proto_alpha_init,
        )

        self.proto_k = int(proto_k)
        self.proto_topk = int(proto_topk)
        self.proto_normalize = bool(proto_normalize)
        self.proto_gamma_init = float(proto_gamma_init)
        self.proto_alpha_init = float(proto_alpha_init)

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
            alpha_init=self.proto_alpha_init,
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
